import numpy as np
import cv2
from functools import reduce
import math
import src.py3d_tools as p3d
import torch
from einops import rearrange


def anim_frame_warp_3d(device, prev_img_cv2, depth, anim_args, frame_idx):
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -anim_args.translation_x_series[frame_idx] * TRANSLATION_SCALE,
        anim_args.translation_y_series[frame_idx] * TRANSLATION_SCALE,
        -anim_args.translation_z_series[frame_idx] * TRANSLATION_SCALE,
    ]
    rotate_xyz = [
        math.radians(anim_args.rotation_3d_x_series[frame_idx]),
        math.radians(anim_args.rotation_3d_y_series[frame_idx]),
        math.radians(anim_args.rotation_3d_z_series[frame_idx]),
    ]
    rot_mat = p3d.euler_angles_to_matrix(
        torch.tensor(rotate_xyz, device=device), "XYZ"
    ).unsqueeze(0)
    result = transform_image_3d(
        device, prev_img_cv2, depth, rot_mat, translate_xyz, anim_args
    )
    torch.cuda.empty_cache()
    return result


def transform_image_3d(
    device, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args
):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w) / float(h)
    near, far, fov_deg = anim_args.near_plane, anim_args.far_plane, anim_args.fov
    persp_cam_old = p3d.FoVPerspectiveCameras(
        near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device
    )
    persp_cam_new = p3d.FoVPerspectiveCameras(
        near,
        far,
        aspect_ratio,
        fov=fov_deg,
        degrees=True,
        R=rot_mat,
        T=torch.tensor([translate]),
        device=device,
    )

    # range of [-1,1] is important to torch grid_sample's padding handling
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=device),
        torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=device),
    )
    if depth_tensor is None:
        z = torch.ones_like(x)
    else:
        z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(
        xyz_old_world
    )[:, 0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(
        xyz_old_world
    )[:, 0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    ).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(
        identity_2d_batch, [1, 1, h, w], align_corners=False
    )
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

    image_tensor = rearrange(
        torch.from_numpy(prev_img_cv2.astype(np.float32)), "h w c -> c h w"
    ).to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=anim_args.sampling_mode,
        padding_mode=anim_args.padding_mode,
        align_corners=False,
    )

    # convert back to cv2 style numpy array
    result = (
        rearrange(new_image.squeeze().clamp(0, 255), "c h w -> h w c")
        .cpu()
        .numpy()
        .astype(prev_img_cv2.dtype)
    )
    return result


# TODO: move to right place
class AnimArgs:
    def __init__(
        self,
        near_plane,
        far_plane,
        fov,
        sampling_mode,
        padding_mode,
        translation_x_series,
        translation_y_series,
        translation_z_series,
        rotation_3d_x_series,
        rotation_3d_y_series,
        rotation_3d_z_series,
    ):
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov = fov
        self.sampling_mode = sampling_mode
        self.padding_mode = padding_mode
        self.translation_x_series = translation_x_series
        self.translation_y_series = translation_y_series
        self.translation_z_series = translation_z_series
        self.rotation_3d_x_series = rotation_3d_x_series
        self.rotation_3d_y_series = rotation_3d_y_series
        self.rotation_3d_z_series = rotation_3d_z_series
