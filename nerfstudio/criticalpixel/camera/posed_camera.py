from .camera import Camera
import torch
from enum import Enum
from nerfstudio.criticalpixel.geometry.ray import Ray
from nerfstudio.criticalpixel.geometry.transform import CoordinateType


class PosedCamera:
    def __init__(self, cameras: Camera, type: CoordinateType) -> None:
        self._cameras = cameras
        self._pose_type = type

    def transorm_to_current_coordinates(self, directions: torch.Tensor):
        if self._pose_type == CoordinateType.OpenGL:
            directions = directions * torch.tensor([1, -1, -1], device=directions.device)
        elif self._pose_type == CoordinateType.OpenCV:
            pass
        else:
            pass
        return directions

    def get_directions(self, indices: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        device = uv.device

        cameras: Camera = self._cameras[indices].to(device)
        directions = cameras.backproject_to_3d(uv)
        return self.transorm_to_current_coordinates(directions)

    def get_pixelwise_rays(self, indices: torch.Tensor, pose_c2w: torch.Tensor):
        cam = self._cameras[indices].to(pose_c2w.device)

        directions = cam.pixelwise_directions()

        directions = self.transorm_to_current_coordinates(directions)
        ray_dir = (pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = pose_c2w[..., :3, 3].expand_as(ray_dir)

        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

        return Ray(origin=ray_origin, direction=ray_dir, batch_size=ray_origin.shape[:-1])

    def get_rays(self, indices: torch.Tensor, uv: torch.Tensor, pose_c2w: torch.Tensor) -> Ray:
        prefix = indices.shape[:-1]
        device = uv.device
        directions = self.get_directions(indices, uv)

        ray_dir = (pose_c2w[..., :3, :3] @ directions.unsqueeze(-1)).squeeze(-1)
        ray_origin = pose_c2w[..., :3, 3]
        ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
        return Ray(origin=ray_origin, direction=ray_dir, batch_size=uv.shape[:-1])
