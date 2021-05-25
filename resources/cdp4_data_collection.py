import os
import time
import rospy
import numpy as np
import cameratransform as ct
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from dvs_msgs.msg import EventArray
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates, LinkStates
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetLinkState, GetModelState, SetModelState,\
    GetWorldProperties, SpawnEntity, DeleteModel, SetVisualProperties,\
    SetVisualPropertiesRequest
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class CDP4DataCollection:

    def __init__(self, camera_params, object_names):

        self.bridge = CvBridge()
        for camera_id in range(camera_params['n_cameras']):
            setattr(self, 'last_image_%02i' % (camera_id,), [[None, None, None]])
        self.last_model_states = ModelStates()
        self.spawned_objects = []

        # Camera IDs and corresponding variables holding images
        self.camera_type = camera_params['camera_type']
        self.update_rate = camera_params['update_rate']
        self.camera_step = 1.0/self.update_rate  # in secdons
        self.image_dims = camera_params['camera_resolution'] + (3,)
        self.camera_dict = {'%02i' % (camera_id,): getattr(self, 'last_image_%02i'\
            % (camera_id,)) for camera_id in range(camera_params['n_cameras'])}
    
        # ROS Subscribers, Publishers, Services
        rospy.init_node('cdp4_data_collection')
        for camera_id in range(camera_params['n_cameras']):
            if self.camera_type in ['rgb', 'both']:
                img_callback_rgb = self.__image_callback_wrapper(camera_id, 'rgb')
                topic_name_rgb = '/robot/camera_rgb_%02i' % (camera_id,)
                setattr(self, '__image_rgb_%02i_sub' % (camera_id,), rospy.Subscriber(
                    topic_name_rgb, Image, img_callback_rgb, queue_size=1))
            if self.camera_type in ['dvs', 'both']:
                img_callback_dvs = self.__image_callback_wrapper(camera_id, 'dvs')
                topic_name_dvs = '/robot/camera_dvs_%02i/events' % (camera_id,)
                setattr(self, '__image_dvs_%02i_sub' % (camera_id,), rospy.Subscriber(
                    topic_name_dvs, EventArray, img_callback_dvs, queue_size=1))

        rospy.wait_for_service('gazebo/set_model_state', 10.0)  # get_model_state not used for now
        self.__set_model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)   
        rospy.wait_for_service('gazebo/get_model_state', 10.0)  # get_model_state not used for now
        self.__get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('gazebo/set_model_state', 10.0)
        self.__set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        rospy.wait_for_service('gazebo/get_link_state', 10.0)
        self.__get_link_state_srv = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        
        # 3D voxel to 2D camera pixel projection parameters (useful for segmentation dataset)
        self.cam_transform = []
        for camera_id in range(camera_params['n_cameras']):
            cam_pose = self.get_object_pose('camera_%02i' % (camera_id))
            cam_pos = cam_pose.position
            cam_ori = cam_pose.orientation
            roll, pitch, yaw = euler_from_quaternion([cam_ori.x, cam_ori.y, cam_ori.z, cam_ori.w])
            (roll, pitch, yaw) = (angle_in_rad*180/np.pi for angle_in_rad in (roll, pitch, yaw))
            cam_projection = ct.RectilinearProjection(
                focallength_px=camera_params['focal_length_px'],
                image=camera_params['camera_resolution'])
            cam_orientation = ct.SpatialOrientation(
                pos_x_m=cam_pos.x, pos_y_m=cam_pos.y, elevation_m=cam_pos.z,
                roll_deg=roll, tilt_deg=90-pitch, heading_deg=90-yaw)
            cam_lens = ct.BrownLensDistortion(
                k1=0.0, k2=0.0, k3=0.0, projection=cam_projection)
            self.cam_transform.append(ct.Camera(
                projection=cam_projection, orientation=cam_orientation, lens=cam_lens))

        # rospy.init_node('cdp4_data_collection')  # moved abovey
        # rospy.wait_for_service('/gazebo/get_world_properties')
        # self.__physics_state = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        # while self.__physics_state().sim_time < 2:
        #     print('Waiting for simulation to be started')
        #     rospy.sleep(2)

        camera_names = ['camera_%02i' % (cam_id) for cam_id in range(camera_params['n_cameras'])]
        camera_names = [name + '::eye_vision_camera' for name in camera_names]  # not working for now
        subcribed_link_names = camera_names + object_names
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_callback, subcribed_link_names)
        self.link_poses = {name: None for name in subcribed_link_names}  # initialization
        rospy.sleep(0.1)  # to avoid the subscriber being not ready
        self.last_frame_timestamp = [rospy.Time.now() for camera_id in range(camera_params['n_cameras'])]
    
    def link_callback(self, data, args):
        for link_name in args:
            array_index = data.name.index(link_name)
            self.link_poses[link_name] = data.pose[array_index]

    def __image_callback_wrapper(self, camera_id, mode):
        """
        Returns correct image saving function

        :param camera_id: The id of the camera to be used 
        """
        last_image = getattr(self, 'last_image_%02i' % (camera_id,))
        def image_callback(msg):
            """
            Saves the last published image to last_image

            :param msg: The ROS message
            """
            try:
                if mode == 'rgb':
                    timestamp = (msg.header.stamp.secs, msg.header.stamp.nsecs)
                    rgb_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
                if mode == 'dvs':
                    dvs_img = np.zeros(self.image_dims, dtype=np.uint8)
                    for event in msg.events:
                        dvs_img[event.y][event.x] = (event.polarity*255, 255, 0)
            except CvBridgeError, e:
                print e
            else:
                if mode == 'rgb':
                    last_image[0][0] = rgb_img
                    last_image[0][2] = timestamp
                if mode == 'dvs':
                    last_image[0][1] = dvs_img
        return image_callback

    def get_object_pose(self, object_name, reference_frame='world'):
        """
        Gets the current pose of an object relative to the world's coordinate frame

        :param object_name: the model name of the object
        :param reference_frame: the reference frame from which the pose will be calculated
        """
        if '::' in object_name:
            return self.link_poses[object_name]
            # return self.__get_link_state_srv(object_name, reference_frame).link_state.pose
        else:
            return self.__get_model_state_srv(object_name, reference_frame).pose

    def move_camera(self, camera_id, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        """
        Moves the robot camera by a desired amount of distance (meter) and angle (rad)
        """
        # model_name = 'camera_%02i::eye_vision_camera' % (camera_id)
        model_name = 'camera_%02i' % (camera_id)
        cur_pose = self.get_object_pose(model_name)
        cur_orientation = euler_from_quaternion([cur_pose.orientation.x,
                                                 cur_pose.orientation.y,
                                                 cur_pose.orientation.z,
                                                 cur_pose.orientation.w])
        orientation = quaternion_from_euler(cur_orientation[0] + roll,
                                            cur_orientation[1] + pitch,
                                            cur_orientation[2] + yaw)
        msg = ModelState()
        msg.model_name = model_name
        msg.reference_frame = 'world'
        msg.scale.x = 1
        msg.scale.y = 1
        msg.scale.z = 1
        msg.pose.position.x = cur_pose.position.x + x
        msg.pose.position.y = cur_pose.position.y + y
        msg.pose.position.z = cur_pose.position.z + z
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        self.__set_model_state_srv(msg)  # takes 0.35 seconds

        # Update 3D world to 2D camera array projection state
        self.cam_transform[camera_id].pos_x_m += x
        self.cam_transform[camera_id].pos_y_m += y
        self.cam_transform[camera_id].elevation_m += z
        self.cam_transform[camera_id].roll_deg += roll*180/np.pi
        self.cam_transform[camera_id].tilt_deg -= pitch*180/np.pi
        self.cam_transform[camera_id].heading_deg -= yaw*180/np.pi
       
    def capture_image(self, camera_id, ensure_timing=True):
        """
        Captures an image with a time stamp greater than the current time. This helps us overcome
        ROS synchronization issues and ensures that we don't get images from the past

        :param camera_id: The id of the camera
        :param ensure_timing: Do the time check or not (slower)?
        """
        if ensure_timing:
            # now = rospy.get_rostime() - rospy.Time(secs=0)
            # time_step = 0.01  # 0.5/self.update_rate  # 0.01           
            # while (now + rospy.Duration(0, time_step)) > rospy.Duration(
            #         self.camera_dict['%02i' % (camera_id,)][0][2][0],
            #         self.camera_dict['%02i' % (camera_id,)][0][2][1]):
            #     rospy.sleep(time_step/10)
            step = rospy.Duration.from_sec(self.camera_step)  # 1.0 / frame_rate
            while (rospy.Time.now() - self.last_frame_timestamp[camera_id] < step):
                rospy.sleep(step/10)
            self.last_frame_timestamp[camera_id] = rospy.Time.now()
            
        rgb_img = self.camera_dict['%02i' % (camera_id,)][0][0]
        dvs_img = self.camera_dict['%02i' % (camera_id,)][0][1]
        return rgb_img, dvs_img
