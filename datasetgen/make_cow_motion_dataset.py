from utils import *

cam_dist = 1.75
device = torch.device("cpu")

# # Set paths
date = 'dec_19_gens'
mesh_dir = os.path.join("data", "meshes")
obj_filename = os.path.join(mesh_dir, "cow_mesh/cow.obj")
mask_path = f'{mesh_dir}/cow_mesh/grass.png'

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)

###### METHOD 1: SPIN ######
filename = 'test'
data_dir = f'data/datasets/{date}/{filename}/'
video_data_dir = os.path.join(data_dir, 'videos/')

# Generate translations and animate the mesh
num_frames = 5
verts = [1234, 1349]
mesh_T = generate_left_right_translations(num_frames)
mesh_R = generate_continuous_spin_rotations(num_frames, rotation_speed=10)
renderer, R, T = setup_renderer(device, num_frames=num_frames, cam_dist=cam_dist)

# produce animation and save as video
animated_images, animated_depths, animated_vert_images, animated_visibilities, animated_vert_coords = animate_mesh_vert_translation(mesh, renderer, mesh_T, mesh_R, verts)
masked_images = np.array([add_mask_from_png_to_image_frame(frame, mask_path) for frame in animated_images])

# save as video etc etc 
# save_images_as_video(np.array(animated_images), data_dir = video_data_dir, filename=filename, fps=15)
save_images_as_video(np.array(animated_vert_images), data_dir = video_data_dir, filename=filename+'_pointvert_visibilities', fps=15)
save_images_as_video(np.array(masked_images), data_dir = video_data_dir, filename=filename+'_masked', fps=15)
save_depths_as_video(np.array(animated_depths), data_dir = video_data_dir, filename=filename+'_depth', fps=15, depth_max=3)

save_mesh_R_T(mesh_R, mesh_T, data_dir = data_dir+'GT/', filename=filename)
save_camera_R_T(R, T, data_dir = data_dir+'GT/', filename=filename)
save_vis(verts, animated_visibilities, animated_vert_coords, data_dir = data_dir+'GT/', filename=filename)

# ####### METHOD 2: SQUARE PATH ######
# filename = 'square'
# data_dir = f'data/datasets/{date}/{filename}/'
# video_data_dir = os.path.join(data_dir, 'videos/')

# # Generate translations and animate the mesh
# num_frames = 40
# verts = [1234, 1349]
# mesh_T = generate_square_path_translations_ranged(num_frames, (-0.5, 0.5), (-0.5, 0.5))
# mesh_R = generate_continuous_spin_rotations(num_frames, rotation_speed=0)
# renderer, R, T = setup_renderer(device, num_frames=num_frames, cam_dist=cam_dist)

# # produce animation and save as video
# animated_images, animated_depths, animated_vert_images, animated_visibilities, animated_vert_coords = animate_mesh_vert_translation(mesh, renderer, mesh_T, mesh_R, verts)
# masked_images = np.array([add_mask_from_png_to_image_frame(frame, mask_path) for frame in animated_images])

# # save as video etc etc 
# # save_images_as_video(np.array(animated_images), data_dir = video_data_dir, filename=filename, fps=15)
# save_images_as_video(np.array(animated_vert_images), data_dir = video_data_dir, filename=filename+'_pointvert_visibilities', fps=15)
# save_images_as_video(np.array(masked_images), data_dir = video_data_dir, filename=filename+'_masked', fps=15)
# save_depths_as_video(np.array(animated_depths), data_dir = video_data_dir, filename=filename+'_depth', fps=15, depth_max=3)

# save_mesh_R_T(mesh_R, mesh_T, data_dir = data_dir+'GT/', filename=filename)
# save_camera_R_T(R, T, data_dir = data_dir+'GT/', filename=filename)
# save_vis(verts, animated_visibilities, animated_vert_coords, data_dir = data_dir+'GT/', filename=filename)

# ###### METHOD 2: SQUARE PATH ######
# filename = 'square_path_zoom'
# data_dir = f'data/{date}/{filename}/'
# video_data_dir = data_dir + 'videos/'

# # Generate translations and animate the mesh
# num_frames = 40
# verts = [1234, 1349]
# mesh_T = generate_square_path_translations_ranged(num_frames, (-0.5, 0.5), (-0.5, 0.5))
# mesh_R = generate_continuous_spin_rotations(num_frames, rotation_speed=0)
# renderer, R, T = setup_renderer(device, num_frames=num_frames, cam_dist=cam_dist)

# # produce animation and save as video
# animated_images, animated_depths, animated_vert_images, animated_visibilities, animated_vert_coords = animate_mesh_vert_translation(mesh, renderer, mesh_T, mesh_R, verts)
# masked_images = np.array([add_mask_from_png_to_image_frame(frame, mask_path) for frame in animated_images])

# # save as video etc etc 
# save_images_as_video(np.array(animated_images), data_dir = video_data_dir, filename=filename, fps=15)
# save_images_as_video(np.array(animated_vert_images), data_dir = video_data_dir, filename=filename+'_verts', fps=15)
# save_images_as_video(np.array(masked_images), data_dir = video_data_dir, filename=filename+'_masked', fps=15)
# save_images_as_video(np.array(animated_depths), data_dir = video_data_dir, filename=filename+'_depth', fps=15)

# save_mesh_R_T(mesh_R, mesh_T, data_dir = data_dir, filename=filename)
# save_camera_R_T(R, T, data_dir = data_dir, filename=filename)
# save_vis(verts, animated_visibilities, animated_vert_coords, data_dir = data_dir, filename=filename)

# ###### METHOD 3: SQUARE PATH ######
# filename = 'square_path_spin_wide_zoom'
# data_dir = f'data/{date}/{filename}/'
# video_data_dir = data_dir + 'videos/'

# # Generate translations and animate the mesh
# num_frames = 40
# verts = [1234, 1349]
# mesh_T = generate_square_path_translations(num_frames)
# mesh_R = generate_continuous_spin_rotations(num_frames, rotation_speed=10)
# renderer, R, T = setup_renderer(device, num_frames=num_frames, cam_dist=cam_dist)

# # produce animation and save as video
# animated_images, animated_depths, animated_vert_images, animated_visibilities, animated_vert_coords = animate_mesh_vert_translation(mesh, renderer, mesh_T, mesh_R, verts)
# masked_images = np.array([add_mask_from_png_to_image_frame(frame, mask_path) for frame in animated_images])

# # save as video etc etc 
# save_images_as_video(np.array(animated_images), data_dir = video_data_dir, filename=filename, fps=15)
# save_images_as_video(np.array(animated_vert_images), data_dir = video_data_dir, filename=filename+'_verts', fps=15)
# save_images_as_video(np.array(masked_images), data_dir = video_data_dir, filename=filename+'_masked', fps=15)
# save_images_as_video(np.array(animated_depths), data_dir = video_data_dir, filename=filename+'_depth', fps=15)

# save_mesh_R_T(mesh_R, mesh_T, data_dir = data_dir, filename=filename)
# save_camera_R_T(R, T, data_dir = data_dir, filename=filename)
# save_vis(verts, animated_visibilities, animated_vert_coords, data_dir = data_dir, filename=filename)
