import torch
from nerfstudio.criticalpixel.camera.camera import Camera, CameraModel
from nerfstudio.criticalpixel.backend import c_backend
import math


class PanoramicCamera(Camera):
    model = CameraModel.Panoramic

    def backproject_to_3d(self, uv: torch.Tensor) -> torch.Tensor:
        assert torch.equal(self.hws[..., 1], self.hws[..., 0] * 2.0)

        h, w = self.hws.unbind(dim=-1)
        f = h / math.pi

        cx = w * 0.5
        cy = h * 0.5

        u, v = uv.unbind(dim=-1)

        p = (u - cx) / f
        t = (v - cy) / f
        cos_t = torch.cos(t)
        x = cos_t * torch.sin(p)
        y = torch.sin(t)
        z = cos_t * torch.cos(p)
        return torch.stack([x, y, z], dim=-1)

    def project_to_2d(self, points: torch.Tensor) -> torch.Tensor:
        assert torch.equal(self.hws[..., 1], self.hws[..., 0] * 2.0)

        x, y, z = points.unbind(dim=-1)
        lon = torch.atan2(x, z)
        lat = torch.atan2(y, torch.hypot(x, z))

        h, w = self.hws.unbind(dim=-1)
        f = h / math.pi

        cx = w * 0.5
        cy = h * 0.5

        u = lon * f + cx
        v = lat * f + cy
        return torch.stack([u, v], dim=-1)

    def rescale(self, scale: float):
        self.params[..., :3] *= scale
        self.hws = torch.round(self.hws * scale).int()
