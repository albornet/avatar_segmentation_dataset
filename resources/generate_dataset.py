import imageio
import numpy as np
import random
import h5py
import time
import sys
import re
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from cdp4_data_collection import CDP4DataCollection
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation
from cv2 import fillConvexPoly, fillPoly
from glob import glob
eps = sys.float_info.epsilon

# Some global variables
class TrainingRoom():

	# May be this function should also build the walls and create the camera object
	def __init__(self, camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence=20):
		
		# Camera parameters
		self.image_dimensions = image_dimensions
		self.n_frames_per_sequence = n_frames_per_sequence
		self.n_cameras = camera_params['n_cameras']
		self.camera_type = camera_params['camera_type']
		self.camera_near_distance = camera_params['near_distance']
		self.n_sequences_per_scene = n_sequences_per_scene
		self.n_sequences_per_camera = n_sequences_per_scene/self.n_cameras
		self.camera_speed_states = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for c in range(self.n_cameras)]
		self.camera_speed_range_t = [0.00, 0.03]  # tangential speed
		self.camera_speed_range_r = [0.50, 2.00]  # radial speed, > 1 for inward spiral, < 1 for outward spiral
		self.camera_speed_t = np.random.uniform(*self.camera_speed_range_t)
		self.camera_speed_r = np.random.uniform(*self.camera_speed_range_r)
		self.robot_position = [-0.46, 0.10, 1.30]  # torso potition (z = 1.60 for head position)
		self.camera_look_at = self.robot_position  # initialization
		self.camera_heights = [0.0, 1.0]  # above self.camera_look_at[-1]
		self.min_camera_radius = 1.0
		self.max_camera_radius = 2.0
		self.camera_angle_range = [-np.pi/2.0, np.pi/2.0]

		# Objects in the scene variables
		self.object_dict = {
			'Head': '_012',
			'LeftArm': '_013', 'LeftForeArm': '_014', 'LeftHand': '_015',
			'RightArm': '_031', 'RightForeArm': '_032', 'RightHand': '_033',

			'LeftHandMiddle1': '_020', 'LeftHandMiddle2': '_025', 'LeftHandMiddle3': '_028',
			'LeftHandThumb1': '_016', 'LeftHandThumb2': '_017', 'LeftHandThumb3': '_018',
			'LeftHandIndex1': '_019', 'LeftHandIndex2': '_026', 'LeftHandIndex3': '_027',
			'LeftHandRing1': '_021', 'LeftHandRing2': '_024', 'LeftHandRing3': '_029',
			'LeftHandPinky1': '_022', 'LeftHandPinky2': '_023', 'LeftHandPinky3': '_030',
			
			'RightHandMiddle1': '_038', 'RightHandMiddle2': '_043', 'RightHandMiddle3': '_047',
			'RightHandThumb1': '_034', 'RightHandThumb2': '_035', 'RightHandThumb3': '_036',
			'RightHandIndex1': '_037', 'RightHandIndex2': '_044', 'RightHandIndex3': '_048',
			'RightHandRing1': '_039', 'RightHandRing2': '_042', 'RightHandRing3': '_046',
			'RightHandPinky1': '_040', 'RightHandPinky2': '_041', 'RightHandPinky3': '_045'}

		self.object_shapes = ['avatar_ybot::mixamorig_' + name
			if 'link' not in name else 'iiwa14::iiwa_' + name for name in self.object_dict.keys() ]
		self.object_dae_paths = ['VIS_Alpha_Surface' + path + '.dae' for path in self.object_dict.values()]
		self.object_presence_prob = {name: 1.0 for name in self.object_shapes}
		self.object_numbers = {name: 1 for name in self.object_shapes}
		self.vertex_index_meshes, self.triangle_index_meshes, self.posed_meshes, self.physical_scales = self.load_object_meshes()

		# Segmentation mode?
		# All meshes different
		# self.object_instances = [i+1 for i, shape in enumerate(self.object_shapes) for n in range(self.object_numbers[shape])]  # 0 is no object
		# Left, right, head
		self.object_instances = [1 + int(int(n[1:]) > 12) + int(int(n[1:]) > 30) for n in self.object_dict.values()]
		# Robot or none
		# self.object_instances = [0 for _ in self.object_dict.keys()]

		# Classification and segmentation dataset variables
		self.data_collector = CDP4DataCollection(camera_params, self.object_shapes)
		data_images_shape = (self.n_sequences_per_scene, self.n_frames_per_sequence) + self.image_dimensions
		self.segment_image = np.zeros(self.image_dimensions[:-1], dtype=np.uint8)
		self.segment_layer = np.zeros(self.segment_image.shape, dtype=np.bool)
		self.distance_image = np.zeros(self.segment_image.shape)
		self.distance_layer = np.zeros(self.segment_image.shape)
		self.data_rgb = np.zeros(data_images_shape, dtype=np.uint8)
		self.data_dvs = np.zeros(data_images_shape, dtype=np.uint8)
		self.data_lbl = {'segments': np.zeros(data_images_shape[:-1], dtype=np.uint8)}

	# Select new random position and angle for the camera
	def choose_new_camera_pose(self):

		return dest_x, dest_y, dest_z, dest_roll, dest_pitch, dest_yaw

	# For each caemra, initiate new sequences of frames in a given scene
	def reset_cameras(self):
		for camera_id in range(self.n_cameras):

			# Choose new camera position and angle
			self.camera_look_at = self.robot_position + 1.0*(np.random.random(size=(3,))-0.5)  # add some noise
			radius = np.random.uniform(self.min_camera_radius, self.max_camera_radius)
			angle = np.random.uniform(*self.camera_angle_range)  # 2*np.pi*np.random.random()
			dest_x = self.camera_look_at[0] + radius*np.cos(angle)
			dest_y = self.camera_look_at[1] + radius*np.sin(angle)
			dest_z = self.camera_look_at[2] + np.random.uniform(*self.camera_heights)
			dest_roll = 0.0
			dest_pitch = np.arctan2(dest_z - self.camera_look_at[2], radius)  # points towards table plane
			dest_yaw = np.pi + angle

			# Set camera position and angle
			pose = self.data_collector.get_object_pose('camera_%02i::eye_vision_camera' % (camera_id,))
			ori3_pose = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
			x = dest_x - pose.position.x
			y = dest_y - pose.position.y
			z = dest_z - pose.position.z
			roll = dest_roll - ori3_pose[0]
			pitch = dest_pitch - ori3_pose[1]
			yaw = dest_yaw - ori3_pose[2]
			self.data_collector.move_camera(camera_id, x, y, z, roll, pitch, yaw)

			# Compute the correct velocities to move around the center
			self.camera_speed_t = np.random.uniform(*self.camera_speed_range_t) if np.random.random() > 0.25 else eps
			self.camera_speed_r = np.random.uniform(*self.camera_speed_range_r) if np.random.random() > 0.25 else 1.0
			sense = np.random.choice([-1, 1])
			r_x = dest_x - self.camera_look_at[0]
			r_y = dest_y - self.camera_look_at[1]
			v_x = self.camera_speed_t*sense*r_y
			v_y = -self.camera_speed_t*sense*r_x
			v_z = 0.0
			v_roll = 0.0
			v_pitch = 0.0
			norm_r = (r_x**2 + r_y**2)**(0.5)
			norm_v = (v_x**2 + v_y**2)**(0.5)
			v_yaw = -sense*(norm_v/norm_r)

			# Update speed variables of each camera
			self.camera_speed_states[camera_id] = [v_x, v_y, v_z, v_roll, v_pitch, v_yaw]

	# Circular motion around the center
	def update_cameras_positions_and_speeds(self):
		for camera_id in range(self.n_cameras):

			# Update camera position and orientation according to its speeds
			v_x, v_y, v_z, v_roll, v_pitch, v_yaw = self.camera_speed_states[camera_id]
			self.data_collector.move_camera(camera_id, v_x, v_y, v_z, v_roll, v_pitch, v_yaw)  # takes time (*2)

			# Update the speeds to obtain a circular (or spiralic) motion
			pose = self.data_collector.get_object_pose('camera_%02i::eye_vision_camera' % (camera_id,))  # takes time (*1)
			r_x = self.camera_look_at[0] - pose.position.x
			r_y = self.camera_look_at[1] - pose.position.y
			norm2_r = r_x**2 + r_y**2
			norm2_v = v_x**2 + v_y**2
			factor = norm2_v/norm2_r  # a_c = (v**2/r)*r_unit = (v**2/r**2)*r_vect
			v_x += self.camera_speed_r*factor*r_x
			v_y += self.camera_speed_r*factor*r_y
			if self.camera_speed_r != 1.0:  # keep same v_norm
				new_norm2_v = v_x**2 + v_y**2
				v_x = v_x*(norm2_v/new_norm2_v)**(0.5)
				v_y = v_y*(norm2_v/new_norm2_v)**(0.5)
			self.camera_speed_states[camera_id][0:2] = v_x, v_y

	# Create lists containing all voxel positions (and other useful arrays) inside every object
	def load_object_meshes(self, subdir='centered/'):
		all_vertices_meshes = {shape: np.zeros((0, 3), dtype=float) for shape in self.object_shapes}
		all_triangles_meshes = {shape: np.zeros((0, 3), dtype=int) for shape in self.object_shapes}
		all_physical_scales = {shape: None for shape in self.object_shapes}
		dae_dir = 'dae_files/' + subdir
		for shape, dae_path in zip(self.object_shapes, self.object_dae_paths):
			with open(dae_dir + dae_path) as f:
				s = f.read()
				vertices_info = re.findall(r'<float_array.+?mesh-positions-array.+?>(.+?)</float_array>', s)
				transform_info = re.findall(r'<matrix sid="transform">(.+?)</matrix>.+?<instance_geometry', s, flags=re.DOTALL)  # better way?
				triangles_info = re.findall(r'<triangles.+?<p>(.+?)</p>.+?</triangles>', s, flags=re.DOTALL)
				if len(triangles_info) == 0:
					triangles_info = re.findall(r'<polylist.+?<p>(.+?)</p>.+?</polylist>', s, flags=re.DOTALL)
				for part_id in range(len(vertices_info)):
					transform_matrix = np.array([float(n) for n in transform_info[part_id].split(' ')]).reshape((4, 4))
					vertices_temp = np.array([float(n) for n in vertices_info[part_id].split(' ')])
					vertices_temp = np.reshape(vertices_temp, (vertices_temp.shape[0]/3, 3))
					vertices_temp = np.dot(transform_matrix, np.c_[vertices_temp, np.ones(vertices_temp.shape[0])].T)[:-1].T
					triangles_temp = np.array([int(n) for n in triangles_info[part_id].split(' ')])[::3]
					triangles_temp = np.reshape(triangles_temp, (triangles_temp.shape[0]/3, 3))
					triangles_temp = triangles_temp + all_vertices_meshes[shape].shape[0]  # shift triangle indices
					all_vertices_meshes[shape] = np.vstack((all_vertices_meshes[shape], vertices_temp))
					all_triangles_meshes[shape] = np.vstack((all_triangles_meshes[shape], triangles_temp))
				min_pos = [all_vertices_meshes[shape][:, d].min() for d in range(3)]
				max_pos = [all_vertices_meshes[shape][:, d].max() for d in range(3)]
				all_physical_scales[shape] = [(max_pos[d] - min_pos[d]) for d in range(3)]

		# Better way? Very ugly...
		vertices_meshes_list = []
		triangles_meshes_list = []
		physical_scales_list = []
		posed_meshes_list = [None for shape in self.object_shapes]
		for shape in self.object_shapes:
			vertices_meshes_list.append(all_vertices_meshes[shape])
			triangles_meshes_list.append(all_triangles_meshes[shape])
			physical_scales_list.append(all_physical_scales[shape])
		return (np.array(lst) for lst in [vertices_meshes_list, triangles_meshes_list, posed_meshes_list, physical_scales_list])

	# Transform the basic mesh coordinates with actual object psotion, scale and orientation
	def update_object_meshes(self):

		# Line below takes a lot of time
		poses = [self.data_collector.get_object_pose(shape) for shape in self.object_shapes]
		positions = [[p.position.x, p.position.y, p.position.z] for p in poses]
		orientations = [euler_from_quaternion([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in poses]
		for i, (pos, ori) in enumerate(zip(positions, orientations)):  #, self.physical_scales)):
			rot = Rotation.from_euler('xyz', ori)
			scl = 1.0
			self.posed_meshes[i] = rot.apply(scl*self.vertex_index_meshes[i]) + pos

	# Compute distance of all object vertices to the camera
	def compute_distances_to_cam(self, camera_id, vertices):
		cam = self.data_collector.cam_transform[camera_id]
		cam_pos = [cam.pos_x_m, cam.pos_y_m, cam.elevation_m]
		cam_vector = np.array([l-p for (p,l) in zip(cam_pos, self.camera_look_at)])
		distances = vertices - cam_pos
		proj_dist = np.dot(cam_vector/np.linalg.norm(cam_vector), distances.T)
		norm_dist = np.linalg.norm(distances, axis=1)
		norm_dist[proj_dist < self.camera_near_distance] = np.nan
		return norm_dist

	# Move camera around the scene and take screenshots
	def generate_data_subset(self):
		for sequence_id in range(self.n_sequences_per_camera):
			self.reset_cameras()
			for frame_id in range(self.n_frames_per_sequence):
				self.update_cameras_positions_and_speeds()  # takes a lot of time (*1)
				self.update_object_meshes()                 # takes a lot of time (*2)
				for camera_id in range(self.n_cameras):
					sample_id = sequence_id*self.n_cameras + camera_id
					sequence_sample = self.data_collector.capture_image(camera_id)
					self.data_rgb[sample_id, frame_id] = sequence_sample[0]
					self.data_dvs[sample_id, frame_id] = sequence_sample[1]
					self.segment_image[:] = 0
					self.distance_image[:] = np.inf
					for (vertices, triangles_idx, segment_idx) in zip(self.posed_meshes, self.triangle_index_meshes, self.object_instances):
						self.segment_layer[:] = 0
						self.distance_layer[:] = np.inf
						vertices_2D = np.array(self.data_collector.cam_transform[camera_id].imageFromSpace(vertices))
						if len(vertices_2D) > 0:
							triangles_2D = np.take(vertices_2D, triangles_idx, axis=0).astype(int)
							distances_to_cam_vertices = self.compute_distances_to_cam(camera_id, vertices)
							distances_to_cam_triangles = np.take(distances_to_cam_vertices, triangles_idx, axis=0).max(axis=1)
							for distance, triangle_2D in sorted(zip(distances_to_cam_triangles, triangles_2D), key=lambda x: x[0])[::-1]:
								if not np.isnan(distance):
									fillConvexPoly(self.distance_layer, triangle_2D, distance)
							self.segment_image[self.distance_layer < self.distance_image] = segment_idx
							self.distance_image = np.minimum(self.distance_image, self.distance_layer)
					self.segment_image[:, 0] = self.segment_image[:, 1]  # erase annoying bug
					self.data_lbl['segments'][sample_id, frame_id] = self.segment_image
		self.data_dvs[:, 0] = 0  # difference between 1st frame and previous one does not exist
		if plot_gifs:  # with last camera output (just to plot an example)
			self.record_sequence_gif()

	# Save an example gif of object segmentation labelling (uses 1sr camera only)
	def record_sequence_gif(self):
		gif_frames = []
		for frame_id in range(self.n_frames_per_sequence):
			fig, ax = plt.subplots(dpi=150)
			fig.subplots_adjust(hspace=0.5)
			plt.subplot(1,3,1)
			plt.title('Sample\nrgb frame')
			plt.imshow(self.data_rgb[0, frame_id]/255.0)
			plt.axis('off')
			plt.subplot(1,3,2)
			plt.title('Segmentation\nlabelling')
			plt.imshow(self.data_lbl['segments'][0, frame_id], vmin=0, vmax=max(self.object_instances))
			plt.axis('off')
			plt.subplot(1,3,3)
			plt.title('Sample\ndvs frame')
			plt.imshow(self.data_dvs[0, frame_id]/255.0)
			plt.axis('off')
			fig.canvas.draw()  # draw the canvas, cache the renderer
			gif_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			gif_frames.append(gif_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
			plt.close()
		imageio.mimsave('./segment_examples/sample_%02i.gif' % (scene_index+1,), gif_frames, fps=24)

# Generate the whole dataset
if __name__ == '__main__':

	# Camera and scene parameters
	camera_params = {'name': 'robot'}
	camera_params['n_cameras'] = 1
	camera_params['camera_type'] = 'both'  # 'rgb', 'dvs', 'both'
	with open('../virtual_room.sdf') as sdf:  # !!!all camera parameters should be the same!!!
		text = sdf.read()
		first_cam_text = text.split('<sensor name="camera" type="camera">')[1].split('</sensor>')[0]
		height = int(first_cam_text.split('<height>')[1].split('</height>')[0])
		width = int(first_cam_text.split('<width>')[1].split('</width>')[0])
		h_fov = float(first_cam_text.split('<horizontal_fov>')[1].split('</horizontal_fov>')[0])
		v_fov = h_fov*float(height)/float(width)
		n_cameras_max = int(text.split('<model name="camera_')[-1].split('">')[0])+1
		update_rate = float(first_cam_text.split('<update_rate>')[1].split('</update_rate>')[0])
		near_distance = float(first_cam_text.split('<near>')[1].split('</near>')[0])
	camera_params['camera_resolution'] = (height, width)
	camera_params['focal_length_px'] = 0.5003983220157445*width  # ??? not sure.. but it works only for w=h
	camera_params['update_rate'] = update_rate
	camera_params['near_distance'] = near_distance
	if camera_params['n_cameras'] > n_cameras_max:
		print('Too many cameras selected: number of cameras set to %s.' % (n_cameras_max,))
		camera_params['n_cameras'] = n_cameras_max

	# Dataset parameters
	plot_gifs = True  # record segmentation labelling .gif examples
	dataset_output_dir = './datasets'
	dataset_output_name = 'training_room_dataset'  # a number is added to avoid overwriting
	n_color_channels = 3
	image_dimensions = camera_params['camera_resolution'] + (n_color_channels,)
	n_frames_per_sequence = 20  # 20
	n_sequences_per_scene = 1  # 16
	assert n_sequences_per_scene % camera_params['n_cameras'] == 0,\
		'Error: n_sequences_per_scene must be a multiple of n_cameras.'
	n_samples_per_dataset = 250  # 1000  # ~16 GB for uncompressed np.array((64000, 20, 64, 64, 3), dtype=np.uint8)
	n_scenes_per_dataset = int(n_samples_per_dataset/n_sequences_per_scene)
	if float(n_samples_per_dataset)/n_sequences_per_scene - n_scenes_per_dataset > 0:
		n_scenes_per_dataset += 1  # 1 partial run to finish the sequence samples
	training_room = TrainingRoom(camera_params, image_dimensions, n_sequences_per_scene, n_frames_per_sequence)

	# Create datasets to be filled by the NRP simulation
	starting_time = time.time()
	dataset_dims_images = (n_samples_per_dataset, n_frames_per_sequence,) + image_dimensions
	dataset_dims_lbl_seg = dataset_dims_images[:-1]
	chunk_dims_image = (1,) + dataset_dims_images[1:]
	chunk_dims_seg = (1,) + dataset_dims_images[1:-1]
	file_name_index = len(glob('%s/%s_*.h5' % (dataset_output_dir, dataset_output_name,))) + 1
	dataset_output_name = '%s/%s_%02i.h5' % (dataset_output_dir, dataset_output_name, file_name_index)
	with h5py.File(dataset_output_name, 'w') as f:
		f.create_dataset('rgb_samples', shape=dataset_dims_images, dtype='uint8', chunks=chunk_dims_image, compression='gzip')
		f.create_dataset('dvs_samples', shape=dataset_dims_images, dtype='uint8', chunks=chunk_dims_image, compression='gzip')
		f.create_dataset('lbl_segments', shape=dataset_dims_lbl_seg, chunks=chunk_dims_seg, dtype='uint8', compression='gzip')

		# Fill the dataset with the generated sequences of frames and corresponding labels
		remaining_indexes = np.array(range(n_samples_per_dataset))
		for scene_index in range(n_scenes_per_dataset):
			first_id = scene_index*n_sequences_per_scene
			last_id = min((scene_index+1)*n_sequences_per_scene, n_samples_per_dataset)
			indexes_to_fill = np.random.choice(remaining_indexes, size=(last_id-first_id,), replace=False)
			remaining_indexes = np.delete(remaining_indexes, [np.where(remaining_indexes==idx) for idx in indexes_to_fill])			
			sys.stdout.write('\rCreating dataset (%i/%i sequences generated)' % (first_id, n_samples_per_dataset))
			sys.stdout.flush()
			training_room.generate_data_subset()
			for i, sample_id in enumerate(indexes_to_fill):
				f['rgb_samples'][sample_id] = training_room.data_rgb[i]
				f['dvs_samples'][sample_id] = training_room.data_dvs[i]
				f['lbl_segments'][sample_id] = training_room.data_lbl['segments'][i]

	# Goodbye message
	n_minutes = int((time.time() - starting_time)/60) + 1
	print('\rDataset created in %i minutes (%i/%i sequences generated)' % (n_minutes, n_samples_per_dataset, n_samples_per_dataset))
