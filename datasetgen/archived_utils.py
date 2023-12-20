import os 
import torch
import matplotlib.pyplot as plt
import math

import numpy as np
from tqdm.notebook import tqdm
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    MeshRendererWithFragments,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d
from pytorch3d.renderer.blending import BlendParams

def make_mesh_motion(mesh, R, T, device=torch.device("cpu"), image_size=128, fov=120):
    """
    Make a video of a mesh rotating around the y-axis.
    """
    # the number of different viewpoints from which we want to render the mesh.
    # num_views = 100

    # Get a batch of viewing angles. 
    # elev = torch.linspace(0, 360, num_views)
    # elev = torch.linspace(0, 1, num_views)
    # dist = torch.linspace(2.8, 2.4, num_views)
    # azim = torch.linspace(0, 180, num_views)

    # Place a point light in front of the object. As mentioned above, the front of 
    # the cow is facing the -z direction.
    num_views = R.shape[0]

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    # We arbitrarily choose one particular view that will be used to visualize 
    # results
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                    T=T[None, 1, ...], fov=fov) 
    
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # Define the settings for rasterization and shading. Here we set the output 
    # image to be of size 128X128. As we are rendering images for visualization 
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to 
    # rasterize_meshes.py for explanations of these parameters.  We also leave 
    # bin_size and max_faces_per_bin to their default values of None, which sets 
    # their values using heuristics and ensures that the faster coarse-to-fine 
    # rasterization method is used.  Refer to docs/notes/renderer.md for an 
    # explanation of the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured 
    # Phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            lights=lights,
            blend_params=blend_params
        )
        
    ) 

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    
    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                            T=T[None, i, ...]) for i in range(num_views)]

    images = target_images.cpu().numpy()
    

    return renderer, images, mesh

def create_rotation_matrix(rx, ry, rz, convention="XYZ"):
    """
    Create a rotation matrix from Euler angles.

    Args:
        rx, ry, rz: Rotation angles in radians around x, y, and z axes.
        convention: The convention for Euler angles. Default is "XYZ".

    Returns:
        Rotation matrix.
    """
    # Convert angles to a tensor
    angles = torch.tensor([rx, ry, rz])

    # Create the rotation matrix
    rotation_matrix = euler_angles_to_matrix(angles, convention)

    return rotation_matrix

def animate_two_meshes(mesh1, mesh2, num_frames=30, translation_speed1=0.1, translation_speed2=0.05, device=torch.device("cpu")):
    renderer = setup_renderer(device)
    rendered_images = []

    for i in range(num_frames):
        # Compute translations for each mesh
        translation1 = torch.tensor([i * translation_speed1, 0, 0], device=device)
        translation2 = torch.tensor([0, i * translation_speed2, 0], device=device)

        # Apply translations
        translated_verts1 = mesh1.verts_packed() + translation1
        translated_verts2 = mesh2.verts_packed() + translation2

        # Combine vertices and faces for both meshes
        verts = torch.cat([translated_verts1, translated_verts2], dim=0)
        faces = torch.cat([mesh1.faces_packed(), mesh2.faces_packed() + len(translated_verts1)], dim=0)

        # Create a new Meshes object with combined data
        combined_mesh = Meshes(verts=[verts], faces=[faces])

        # Render the combined mesh
        image = renderer(combined_mesh)
        rendered_images.append(image)

    return rendered_images

# ###### METHOD 1: TRANSLATE MESH ######
# filename = 'scratchpad'
# data_dir = f'data/dec_12_gens/'

# # Generate translations and animate the mesh
# num_frames = 20
# mesh_T = generate_left_right_translations(num_frames)
# mesh_R = generate_continuous_spin_rotations(num_frames, rotation_speed=10)
# renderer, R, T = setup_renderer(device, num_frames=num_frames)

# # produce animation and save as video
# animated_images, animated_depths = animate_mesh_translation(mesh, renderer, mesh_T, mesh_R, device=device)
# animated_images = np.squeeze(np.array(animated_images), axis=1)
# masked_images = np.array([add_mask_to_image_frame(frame, (1,0,0)) for frame in animated_images])

# # save as video etc etc 
# save_images_as_video(np.array(animated_images), data_dir = data_dir, filename=filename, fps=15)
# save_images_as_video(np.array(masked_images), data_dir = data_dir, filename=filename+'_masked', fps=15)
# save_images_as_video(np.array(animated_depths), data_dir = data_dir, filename=filename+'_depth', fps=15)
# save_mesh_R_T(mesh_R, mesh_T, data_dir = data_dir, filename=filename)
# save_camera_R_T(R, T, data_dir = data_dir, filename=filename)

# ok last thing!!! save a particle trackkkkk


# todo - this is out of date ! 
# data_filename = 'data/simple_sq'
# # Generate square path translations and continuous spin rotations
# num_frames = 40
# translations = generate_square_path_translations(num_frames, side_length=2.0)
# rotations = generate_continuous_spin_rotations(num_frames, rotation_speed=0)
# # Animate the mesh with these translations and rotations
# renderer = setup_renderer(device)
# animated_images = animate_mesh_translation(mesh, renderer, translations, rotations, device=device)
# # Convert to numpy array and save as video
# animated_images = np.squeeze(np.array(animated_images), axis=1)
# save_images_as_video(np.array(animated_images), filename=data_filename, fps=25)


###### METHOD 2: CHANGE R's and T's ######

# create motion params
# todo: clean dis up
# num_views = 40

# # dist_lin = torch.linspace(1.5, 3, num_views)
# dist_there_back = torch.cat((torch.linspace(1.5, 3, int(num_views/2)), 
#                              torch.linspace(1.5, 3, int(num_views/2)).flip(0)), 0)
# dist_const = torch.linspace(2, 2, num_views)
# azim_spin = torch.linspace(0, 180, num_views)
# azim_const = torch.linspace(0, 0, num_views)
# elev_spin = torch.linspace(-89, 89, num_views)
# elev_const = torch.linspace(0, 0, num_views)
# elev = torch.linspace(-89, 89, num_views)
# dist = torch.linspace(1.5, 3, num_views)
# azim = torch.linspace(0, 180, num_views)

# # testcow
# cow_filename = 'data/all_motion'
# R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
# renderer, images, meshes = make_mesh_motion(mesh, R, T, image_size=250)
# save_images_as_video(images, filename=cow_filename, fps=25)
# save_R_T(R, T, cow_filename)

# # azim spin
# cow_filename = 'data/azim_spin'
# R, T = look_at_view_transform(dist=dist_const, elev=elev_const, azim=azim_spin)
# renderer, images, meshes = make_mesh_motion(mesh, R, T, image_size=250)
# save_images_as_video(images, filename=cow_filename, fps=25)
# save_R_T(R, T, cow_filename)

# # there and back
# cow_filename = 'data/there_back'
# R, T = look_at_view_transform(dist=dist_there_back, elev=elev_const, azim=azim_const)
# renderer, images, meshes = make_mesh_motion(mesh, R, T, image_size=250)
# save_images_as_video(images, filename=cow_filename, fps=25)
# save_R_T(R, T, cow_filename)

# # elev spin
# cow_filename = 'data/elev_spin'
# R, T = look_at_view_transform(dist=dist_const, elev=elev_spin, azim=azim_const)
# renderer, images, meshes = make_mesh_motion(mesh, R, T, image_size=250)
# save_images_as_video(images, filename=cow_filename, fps=25)
# save_R_T(R, T, cow_filename)


############################################################################################################
# # TABLED: TWO MESHES
# mesh1 = load_objs_as_meshes([obj_filename], device=device)
# mesh2 = load_objs_as_meshes([obj_filename], device=device)

# # Animate the meshes
# animated_images = animate_two_meshes(mesh1, mesh2, num_frames=10, translation_speed1=0.05, translation_speed2=0.02, device=device)
# animated_images = np.squeeze(np.array(animated_images), axis=1)
# save_images_as_video(np.array(animated_images), filename='data/animated_meshes', fps=25)