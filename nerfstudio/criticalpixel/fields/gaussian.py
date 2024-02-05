from nerfstudio.criticalpixel.fields.base_field import BaseFieldConfig, BaseField
from dataclasses import dataclass, field
from typing import Type, Optional
import torch
from nerfstudio.criticalpixel.geometry.point_cloud import PointCloud
from nerfstudio.criticalpixel.geometry.gaussian import Gaussian3D
from torch import nn
from nerfstudio.criticalpixel.appearance.sphere_harmonics import RGB2SH

@dataclass
class GaussianFieldConfig(BaseFieldConfig):
    _target: Type = field(default_factory=lambda: GaussianField)
    sh_degree: int = 4
    '''color of gaussian'''


class GaussianField(BaseField):
    geometry_type = "gaussian"
    config: GaussianFieldConfig

    def populate_modules(self):
        pass
        
    def init_from_pointcloud(self, pointcloud: PointCloud):
        pointcloud = pointcloud.view(-1)
        N = pointcloud.shape[0]

        fused_point_cloud = pointcloud.points.float()
        appearance_features = torch.zeros((N, 3, self.config.sh_degree ** 2), dtype=torch.float32, device='cpu')
        if pointcloud.colors is not None:
             appearance_features[:, :3, 0 ] = RGB2SH(pointcloud.colors)


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = pointcloud.knn(3)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def init_pointcloud(self, pointcloud: Optional[PointCloud] = None):

        if pointcloud is None:
            pass
        else:
            self.init_from_pointcloud(pointcloud)