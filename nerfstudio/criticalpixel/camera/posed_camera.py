import math
from typing import Optional

import numpy as np
import torch
from tensordict import tensorclass

from nerfstudio.criticalpixel.geometry.ray import Ray
from nerfstudio.criticalpixel.geometry.transform import CoordinateType

from .camera import Camera, CameraModel


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.eye(4)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W).float()
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


@tensorclass
class PosedCamera:
    cameras: Camera
    pose_c2w: torch.Tensor
    coord_type: CoordinateType
    unique_ids: Optional[torch.Tensor]

    def rescale(self, scale: float):
        self.cameras.rescale(scale)

    def transorm_to_current_coordinates(self, directions: torch.Tensor):
        if self.coord_type == CoordinateType.OpenGL:
            directions = directions * torch.tensor([1, -1, -1], device=directions.device)
        elif self.coord_type == CoordinateType.OpenCV:
            pass
        else:
            pass
        return directions

    def get_directions(self, uv: torch.Tensor) -> torch.Tensor:
        directions = self.cameras.backproject_to_3d(uv)
        return self.transorm_to_current_coordinates(directions)

    def get_pixelwise_rays(self):
        directions = self.cameras.pixelwise_directions()
        directions = self.transorm_to_current_coordinates(directions)
        ray_dir = (self.pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = self.pose_c2w[..., :3, 3].expand_as(ray_dir)

        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
        return Ray(origin=ray_origin, direction=ray_dir, batch_size=ray_origin.shape[:-1])

    def get_rays(self, uv: torch.Tensor) -> Ray:
        directions = self.get_directions(uv)

        ray_dir = (self.pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = self.pose_c2w[..., :3, 3]
        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
        return Ray(origin=ray_origin, direction=ray_dir, batch_size=uv.shape[:-1])

    def get_transforms(self, near: float, far: float):
        # for gsplat
        # because the specific matrix storage of glm library, the transform may by confused.
        # todo: change glm to eigen

        assert self.cameras.model == CameraModel.Pinhole
        world_view_transform = torch.inverse(self.pose_c2w).transpose(1, 2).float()
        projection_matrix = self.cameras.projection_matrix(near=near, far=far).transpose(1, 2)
        full_proj_transform = world_view_transform.bmm(projection_matrix)
        return world_view_transform, projection_matrix, full_proj_transform
