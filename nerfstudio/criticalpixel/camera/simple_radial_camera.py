import torch
from nerfstudio.criticalpixel.camera.camera import Camera, CameraModel
from nerfstudio.criticalpixel.backend import c_backend


class SimpleRadialCamera(Camera):
    model: CameraModel = CameraModel.SimpleRadial

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        f = self.params[..., 0:1]
        c = self.params[..., 1:3]
        d = self.params[..., 3:4]
        print(self.params)
        xy = (uv - c) / f

        print(xy)

        xy = c_backend.Undistort(c_backend.CameraModel.SimpleRadial, xy, d)

        z = torch.ones(self.batch_size, dtype=torch.float32, device=uv.device).unsqueeze(-1)

        xyz = torch.cat([xy, z], dim=-1)
        return xyz

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        f = self.params[..., 0:1]
        c = self.params[..., 1:3]
        d = self.params[..., 3:4]
        xy = points[..., :2] / points[..., 2:]

        xy = c_backend.Distort(c_backend.CameraModel.SimpleRadial, xy, d)

        uv = f * xy + c

        return uv