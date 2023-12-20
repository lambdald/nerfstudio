import torch
from nerfstudio.criticalpixel.camera.camera import Camera, CameraModel
from nerfstudio.criticalpixel.backend import c_backend


class OpenCVCamera(Camera):
    model = CameraModel.OpenCV

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        f = self.params[..., :2]
        c = self.params[..., 2:4]
        d = self.params[..., 4:9]

        xy = (uv - c) / f

        xy = c_backend.Undistort(c_backend.CameraModel.OpenCV, xy, d)

        z = torch.ones(self.batch_size, dtype=torch.float32, device=uv.device).unsqueeze(-1)

        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        f = self.params[..., :2]
        c = self.params[..., 2:4]
        d = self.params[..., 4:9]
        xy = points[..., :2] / points[..., 2:]
        xy = c_backend.Distort(c_backend.CameraModel.OpenCV, xy, d)
        uv = f * xy + c
        return uv
