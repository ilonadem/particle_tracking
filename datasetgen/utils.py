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

from PIL import Image

# saving and loading functions
def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

def save_depths_as_video(images, data_dir=None, filename='output', fps=30, depth_max=2):
    """
    Save a numpy array of images as an MP4 video.

    Parameters:
    - images: numpy array of shape (num_frames, height, width, 4), representing 100 RGBA images.
    - filename: name of the output MP4 file.
    - fps: frames per second for the video.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Get the height and width from the image shape
    height, width = images.shape[1:3]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also use 'XVID'
    print("writing video to: ", filename+'.mp4')
    video = cv2.VideoWriter(data_dir + filename+'.mp4', fourcc, fps, (width, height))

    for img in images:
        # todo: normalize depth image from 0-10
        img = np.clip(img, 0, depth_max)
        img = (255/depth_max)*img
        img = img.astype(np.uint8)
        # Convert RGBA to BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        # Write the frame
        video.write(bgr_img)

    # Release everything when job is finished
    video.release()

def save_images_as_video(images, data_dir=None, filename='output', fps=30):
    """
    Save a numpy array of images as an MP4 video.

    Parameters:
    - images: numpy array of shape (num_frames, height, width, 4), representing 100 RGBA images.
    - filename: name of the output MP4 file.
    - fps: frames per second for the video.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Get the height and width from the image shape
    height, width = images.shape[1:3]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also use 'XVID'
    print("writing video to: ", filename+'.mp4')
    video = cv2.VideoWriter(data_dir + filename+'.mp4', fourcc, fps, (width, height))

    for img in images:
        img = 255*img
        img = img.astype(np.uint8)
        # Convert RGBA to BGR
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        # Write the frame
        video.write(bgr_img)

    # Release everything when job is finished
    video.release()

def save_mesh_R_T(mesh_r, mesh_t, data_dir=None, filename='output'):
    os.makedirs(data_dir, exist_ok=True)
    save_R_T(mesh_r, mesh_t, data_dir=data_dir, filename=filename+'_mesh')

def save_camera_R_T(R, T, data_dir=None, filename='output'):
    os.makedirs(data_dir, exist_ok=True)
    save_R_T(R, T, data_dir=data_dir, filename=filename+'_camera')

def save_R_T(R, T, data_dir=None, filename='output'):
    os.makedirs(data_dir, exist_ok=True)
    torch.save(R, data_dir + filename + '_R.pt')
    torch.save(T, data_dir + filename + '_T.pt')

def load_R_T(filename):
    R = torch.load(filename + '_R.pt')
    T = torch.load(filename + '_T.pt')
    return R, T

# Main functions for mesh manipulation and rendering
def setup_renderer(device, num_frames=1, image_size=512, cam_dist=2.7):
    """
    Set up the PyTorch3D renderer.

    Args:
        device: PyTorch device.
        image_size: Size of the output image.

    Returns:
        Configured renderer object.
    """
    R, T = look_at_view_transform(cam_dist, 0, 180) 

    # assuming a still camera, so R, T are constant in each frame
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=120)
    lights = PointLights(device=device, location=[[0, 0, -3]])
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0, faces_per_pixel=1)
    R = R.repeat(num_frames, 1, 1)
    T = T.repeat(num_frames, 1, 1)
    
    return MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    ), R, T

def animate_mesh_translation(mesh, renderer, translations, rotations, device=torch.device("cpu")):
    """
    Animate mesh translation and rotation, and render each frame using PyTorch3D's Transform3d.

    Args:
        mesh: PyTorch3D Mesh object.
        translations: Tensor of translation vectors for each frame.
        rotations: Tensor of rotation angles (in radians) around the Y-axis for each frame.
        device: PyTorch device.

    Returns:
        List of rendered images.
    """
    rendered_images = []
    rendered_depths = []

    for translation, rotation in zip(translations, rotations):
        # Create a Transform3d object for translation and rotation
        transform = Transform3d().translate(*translation).rotate_axis_angle(angle=rotation, axis="Y")

        # Apply the transformation to the mesh
        transformed_verts = transform.transform_points(mesh.verts_packed())

        # Create a new mesh with the transformed vertices
        transformed_mesh = Meshes(verts=[transformed_verts], faces=[mesh.faces_packed()], textures=mesh.textures)

        # Render the transformed mesh
        image, fragments = renderer(transformed_mesh)
        depth = fragments.zbuf.squeeze().numpy()
        rendered_images.append(image)
        rendered_depths.append(depth)

    return rendered_images, rendered_depths

# vertex visibility functions
def get_visibility_of_single_vert(mesh_instance, renderer, vert_idx, h, w, eps=1e-2):
    # Step 1: Transform vertex to camera space
    world_to_view_transform = renderer.rasterizer.cameras.get_world_to_view_transform()
    view_verts = world_to_view_transform.transform_points(mesh_instance.verts_packed())
    zcam = view_verts[vert_idx, 2]

    # Get pixel coordinates corresponding to the vertex of interest
    S_cam = renderer.rasterizer.cameras.transform_points_screen(mesh_instance.verts_packed(), image_size=(h, w))
    p_cam = S_cam[vert_idx]
    p_x, p_y = int(p_cam[0]), int(p_cam[1])

    # Step 2: Render the scene to extract z-buffer
    _, fragments = renderer(mesh_instance)
    zbuf = fragments.zbuf[0, p_y, p_x, 0]  # Ensure that indices p_x and p_y are not swapped

    # Step 3: Determine visibility
    is_visible = abs(zbuf-zcam)<eps
    
    return is_visible, [p_x, p_y]

def get_vert_visibilities(mesh_instance, renderer, verts, h, w, eps=1e-2):
    # Transform vertices to camera space
    world_to_view_transform = renderer.rasterizer.cameras.get_world_to_view_transform()
    view_verts = world_to_view_transform.transform_points(mesh_instance.verts_packed())
    S_cam = renderer.rasterizer.cameras.transform_points_screen(mesh_instance.verts_packed(), image_size=(h, w))

    visibilities, vert_coords = [], []
    for vert_idx in verts:
        
        # get vertex in camera space
        zcam = view_verts[vert_idx, 2]
        
        # Get pixel coordinates corresponding to the vertex of interest
        p_cam = S_cam[vert_idx]
        p_x, p_y = int(p_cam[0]), int(p_cam[1])

        

        # Step 2: Render the scene to extract z-buffer
        # Check if coordinates are within the frame
        if p_x < 0 or p_x >= w or p_y < 0 or p_y >= h:
            visibilities.append(False)
            vert_coords.append([p_x, p_y])
        else:
            _, fragments = renderer(mesh_instance)
            zbuf = fragments.zbuf[0, p_y, p_x, 0]  # Ensure that indices p_x and p_y are not swapped

            # Step 3: Determine visibility
            is_visible = abs(zbuf-zcam)<eps

            visibilities.append(is_visible)
            vert_coords.append([p_x, p_y])
    
    return np.array(visibilities), np.array(vert_coords)

def change_neighborhood_color(image, x, y, color, neighborhood_size=3):
    """
    Change the color of a neighborhood of pixels around a point (x, y) in the image.
    """
    h, w, channels = image.shape
    # Check if the central point is within the bounds of the image
    if x < 0 or x >= w or y < 0 or y >= h:
        return  # Do not change the color if the central point is out of bounds
    
    for i in range(-neighborhood_size, neighborhood_size + 1):
        for j in range(-neighborhood_size, neighborhood_size + 1):
            if 0 <= x + i < w and 0 <= y + j < h:
                image[y + j, x + i, :3] = color  # Update only the first three channels (RGB)

def make_vert_animation(rendered_images, rendered_visibilities, rendered_vert_coords):
        rendered_vert_images = []
        for image, visibilities, vert_coords in zip(rendered_images, rendered_visibilities, rendered_vert_coords):
            vert_image = image.copy()
            for is_visible, p_coord in zip(visibilities, vert_coords):
                color = [0, 1, 0] if is_visible else [1, 0, 0]  # Green if visible, Red if not
                change_neighborhood_color(vert_image, p_coord[0], p_coord[1], color)
            rendered_vert_images.append(vert_image)
        
        return rendered_vert_images

def animate_mesh_vert_translation(mesh, renderer, translations, rotations, verts, device=torch.device("cpu")):
    rendered_images = []
    rendered_depths = []
    # rendered_vert_images = []
    rendered_visibilities = []
    rendered_vert_coords = []

    for translation, rotation in zip(translations, rotations):
        # Create a new mesh with the transformed vertices
        transform = Transform3d().translate(*translation).rotate_axis_angle(angle=rotation, axis="Y")
        transformed_verts = transform.transform_points(mesh.verts_packed())
        transformed_mesh = Meshes(verts=[transformed_verts], faces=[mesh.faces_packed()], textures=mesh.textures)

        # Render the transformed mesh
        image = np.squeeze(renderer(transformed_mesh)[0], axis=0).numpy()  # Convert to numpy array
        depth = renderer(transformed_mesh)[1].zbuf.squeeze().numpy()
        rendered_images.append(image.copy())
        rendered_depths.append(depth)

        # get vertex coords + visibilities
        visibilities, vert_coords = get_vert_visibilities(transformed_mesh, renderer, verts, h=depth.shape[0], w=depth.shape[1])
        rendered_visibilities.append(visibilities)
        rendered_vert_coords.append(vert_coords)

        # vert_image = image.copy()
        # for is_visible, p_coord in zip(visibilities, vert_coords):
        #     color = [0, 1, 0] if is_visible else [1, 0, 0]  # Green if visible, Red if not
        #     change_neighborhood_color(vert_image, p_coord[0], p_coord[1], color)
        # rendered_vert_images.append(vert_image)
    print("rendered_images len: ", len(rendered_images), "rendered_visibilities len", len(rendered_visibilities), "len rendered_vert_coords: ", len(rendered_vert_coords))
    print("rendered_images[0] shape ", rendered_images[0].shape, "rendered_visibilities[0] shape", rendered_visibilities[0].shape, "rendered_vert_coords[0] shape: ", rendered_vert_coords[0].shape)
    print(type(rendered_images[0]), type(rendered_visibilities[0]), type(rendered_vert_coords[0]))
    rendered_vert_images = make_vert_animation(rendered_images, rendered_visibilities, rendered_vert_coords)

    return rendered_images, rendered_depths, rendered_vert_images, rendered_visibilities, rendered_vert_coords

def save_vis(verts, visibilities, vert_coords, data_dir=None, filename='vert'):
    os.makedirs(data_dir, exist_ok=True)
    torch.save(verts, os.path.join(data_dir, filename+'_pointvert_ids.pt'))
    torch.save(visibilities, os.path.join(data_dir, filename+'_pointvert_vis.pt'))
    torch.save(vert_coords, os.path.join(data_dir, filename+'_pointvert_coords.pt'))

# mesh spin / rotation functions
def generate_left_right_translations(num_frames, range_x=1.0, height_variation=0.1):
    """
    Generate translations for left-right motion as a tensor.

    Args:
        num_frames: Number of frames in the animation.
        range_x: Range of motion along the X-axis.
        height_variation: Variation in height (Y-axis) during the motion.

    Returns:
        Tensor of translation vectors.
    """
    translations = torch.zeros((num_frames, 3))
    for i in range(num_frames):
        # Oscillate back and forth along the X-axis
        x = range_x * math.sin(2 * math.pi * i / num_frames)

        # Optionally, vary the height
        y = height_variation * math.sin(4 * math.pi * i / num_frames)

        # Set the translations for each frame
        translations[i] = torch.tensor([x, y, 0])  # Z-axis is constant

    return translations

def generate_rotations(num_frames, rotation_speed=0.1):
    """
    Generate rotation angles for each frame as a tensor.

    Args:
        num_frames: Number of frames in the animation.
        rotation_speed: Rotation speed (radians per frame).

    Returns:
        Tensor of rotation angles.
    """
    return torch.tensor([rotation_speed * i for i in range(num_frames)])

def generate_circular_translations(num_frames, radius=1.0, height_variation=0.1):
    """
    Generate translations for circular motion as a tensor.

    Args:
        num_frames: Number of frames in the animation.
        radius: Radius of the circular path.
        height_variation: Variation in height (Y-axis) during the motion.

    Returns:
        Tensor of translation vectors.
    """
    translations = torch.zeros((num_frames, 3))
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        translations[i] = torch.tensor([radius * math.cos(angle), 
                                        height_variation * math.sin(2 * angle), 
                                        radius * math.sin(angle)])

    return translations

def generate_square_path_translations(num_frames, side_length=2.0):
    """
    Generate translations for a square path.

    Args:
        num_frames: Total number of frames in the animation.
        side_length: Length of each side of the square.

    Returns:
        List of translation vectors.
    """
    translations = []
    # Ensure at least one frame per side
    frames_per_side = max(1, num_frames // 4)

    # Define the four corners of the square
    corners = [
        (0, 0), (side_length, 0),
        (side_length, side_length), (0, side_length), (0, 0)
    ]

    # Generate translations for each side of the square
    for i in range(4):
        start_x, start_y = corners[i]
        end_x, end_y = corners[i + 1]
        for j in range(frames_per_side):
            t = j / (frames_per_side - 1) if frames_per_side > 1 else 0
            x = (1 - t) * start_x + t * end_x
            y = (1 - t) * start_y + t * end_y
            translations.append([x, y, 0])  # Z-coordinate is 0 for planar motion

    # Trim the translations to the desired number of frames
    return translations[:num_frames]

def generate_square_path_translations_ranged(num_frames, horizontal_range=(-2.0, 2.0), vertical_range=(-2.0, 2.0)):
    """
    Generate translations for a square path within specified horizontal and vertical ranges.

    Args:
        num_frames: Total number of frames in the animation.
        horizontal_range: Tuple (min, max) specifying the horizontal range of the square.
        vertical_range: Tuple (min, max) specifying the vertical range of the square.

    Returns:
        List of translation vectors.
    """
    translations = []
    frames_per_side = max(1, num_frames // 4)

    min_x, max_x = horizontal_range
    min_y, max_y = vertical_range

    # Define the four corners of the square within the specified ranges
    corners = [
        (min_x, min_y), (max_x, min_y),
        (max_x, max_y), (min_x, max_y), (min_x, min_y)
    ]

    # Generate translations for each side of the square
    for i in range(4):
        start_x, start_y = corners[i]
        end_x, end_y = corners[i + 1]
        for j in range(frames_per_side):
            t = j / (frames_per_side - 1) if frames_per_side > 1 else 0
            x = (1 - t) * start_x + t * end_x
            y = (1 - t) * start_y + t * end_y
            translations.append([x, y, 0])  # Z-coordinate is 0 for planar motion

    # Trim the translations to the desired number of frames
    return translations[:num_frames]

# def generate_square_path_translations(num_frames, side_length=2.0):
#     """
#     Generate translations for a square path.

#     Args:
#         num_frames: Total number of frames in the animation.
#         side_length: Length of each side of the square.

#     Returns:
#         List of translation vectors.
#     """
#     translations = []
#     frames_per_side = num_frames // 4

#     # Define the four corners of the square
#     corners = [
#         (0, 0), (side_length, 0),
#         (side_length, side_length), (0, side_length), (0, 0)
#     ]

#     # Generate translations for each side of the square
#     for i in range(4):
#         start_x, start_y = corners[i]
#         end_x, end_y = corners[i + 1]
#         for j in range(frames_per_side):
#             t = j / frames_per_side
#             x = (1 - t) * start_x + t * end_x
#             y = (1 - t) * start_y + t * end_y
#             translations.append([x, y, 0])  # Z-coordinate is 0 for planar motion

#     return translations

def generate_continuous_spin_rotations(num_frames, rotation_speed=0.1):
    """
    Generate rotation angles for continuous spin around Y-axis.

    Args:
        num_frames: Number of frames in the animation.
        rotation_speed: Rotation speed (radians per frame).

    Returns:
        List of rotation angles.
    """
    return [rotation_speed * i for i in range(num_frames)]

# adding masks stuff
def create_rgb_image_mask(height, width, color):
    """
    Create a RGB color mask as a NumPy array.

    :param height: Height of the image
    :param width: Width of the image
    :param color: A tuple (R, G, B) specifying the color
    :return: NumPy array representing the image
    """
    # Create a NumPy array with the given color
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color

    return image

def add_mask_to_image_frame(image_frame, color=(1, 0, 0)):
    height, width = image_frame.shape[:2]
    background = create_rgb_image_mask(height, width, color)  # Red color mask

    cow_mask = image_frame[:,:,-1]>0
    background_mask = np.logical_not(cow_mask)

    # Expand the dimensions of the masks
    background_mask_expanded = np.expand_dims(background_mask, axis=-1)
    cow_mask_expanded = np.expand_dims(cow_mask, axis=-1)

    new_im = background_mask_expanded * background + cow_mask_expanded * image_frame[:,:,:3]
    return new_im

def add_mask_from_png_to_image_frame(image_frame, mask_image_path):
    # Read the mask image and convert it to a numpy array
    mask_image = Image.open(mask_image_path)
    mask_image = mask_image.resize((image_frame.shape[1], image_frame.shape[0]))
    mask_image_np = np.array(mask_image).astype(float) / 255.0  # Normalize to 0-1 range

    # Extract only the RGB channels (assuming mask is RGBA or RGB)
    mask_rgb = mask_image_np[:, :, :3]  # Use only the RGB channels

    cow_mask = image_frame[:, :, -1] > 0
    background_mask = np.logical_not(cow_mask)

    # Expand the dimensions of the cow mask
    cow_mask_expanded = np.expand_dims(cow_mask, axis=-1)

    # Apply the masking to RGB channels only
    new_rgb = np.where(cow_mask_expanded, image_frame[:, :, :3], mask_rgb)
    
    # Combine new RGB with the original depth channel
    new_image = np.concatenate((new_rgb, image_frame[:, :, -1:]), axis=-1)
    return new_image