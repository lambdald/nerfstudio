from dataclasses import dataclass, field
from typing import Literal, Type

import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.v3d.core.fields.v3d_neus_field import (V3dNeusField,
                                                       V3dNeusFieldConfig)


@dataclass
class V3dHfneusFieldConfig(V3dNeusFieldConfig):
    _target: Type=field(default_factory=lambda: V3dHfneusField)

class V3dHfneusField(V3dNeusField):
    geometry_type: Literal['density', 'sdf'] = 'sdf'
    config: V3dHfneusFieldConfig

    def get_alpha(self, ray_samples: RaySamples, sdf: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        device = sdf.device
        batch_size, n_samples = sdf.shape[:2]
        view_directions = ray_samples.frustums.directions
        cos_anneal_ratio = self.cos_anneal_ratio
        ray_steps = ray_samples.deltas.squeeze(-1)
        sdf = sdf.squeeze(-1)
        # ### HF-NEUS Alpha Rendering
        true_cos = (view_directions * gradients).sum(-1)
        s = self.deviation_network().clip(1e-6, 1e3).detach()
        inv_s = self.deviation_network().clip(1e-6, 1e6)
        sigmoid_sdf = torch.sigmoid(s * sdf)
        # [B, n_samples]
        weight_sdf = s * sigmoid_sdf * (1 - sigmoid_sdf)
        # [B, 1]
        weight_sdf_sum = weight_sdf.sum(dim=-1, keepdim=True) + 1e-6
        # [B, n_samples]
        weight_sdf = weight_sdf / weight_sdf_sum
        weight_sdf[weight_sdf_sum.squeeze(-1) < 0.2] = 1.0 / n_samples

        # print('weight_sdf', weight_sdf.shape, weight_sdf.detach().min(), weight_sdf.detach().max())
        # print('gradients', gradients.shape)

        inv_s = (inv_s * torch.exp((gradients.norm(dim=-1).clip(1e-3, 5) * weight_sdf.detach()).sum(dim=-1, keepdim=True) - 1))
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(torch.relu(-true_cos * 0.5 + 0.5) *
                    (1.0 - cos_anneal_ratio) + torch.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        # sigma
        cdf = torch.sigmoid(sdf * inv_s)
        # print('cdf:', cdf.shape)
        # print('inv_s:', inv_s.shape)
        # print('iter_cos:', iter_cos.shape)
        # print('ray_steps:', ray_steps.shape)
        e = inv_s * (1 - cdf) * (-iter_cos) * ray_steps
        alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)
        # print('alpha:', alpha.min(), alpha.max())
        return alpha.unsqueeze(-1)
