from dataclasses import dataclass, field
from typing import Dict, Literal, Type, Union

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.v3d.core.fields.v3d_volsdf_field import (V3dVolsdfField,
                                                         V3dVolsdfFieldConfig)


class UnbiasedLaplaceDensity(nn.Module):
    """Laplace density from Unbiased VolSDF"""
    # https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Towards_Unbiased_Volume_Rendering_of_Neural_Implicit_Surfaces_With_Geometry_CVPR_2023_paper.html
    alpha: torch.Tensor
    beta: torch.Tensor

    def __init__(self, init_alpha, init_beta, min_value=0.0001):
        super().__init__()
        self.min_value = min_value
        # 两个同时学比只学beta效果好
        self.register_parameter("alpha", nn.Parameter(torch.tensor(init_alpha), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(torch.tensor(init_beta), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], gradient: torch.Tensor, view_direction: torch.Tensor, beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()
        alpha = self.get_alpha()

        gradient_t =  (view_direction * gradient).norm(dim=-1, keepdim=True)

        scaled_sdf = sdf/(gradient_t+1e-5)

        alpha = self.get_alpha()
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-scaled_sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.min_value
        return beta

    def get_alpha(self):
        alpha = self.alpha.abs() + self.min_value
        return alpha


@dataclass
class V3dUnbiasedVolsdfFieldConfig(V3dVolsdfFieldConfig):
    _target: Type=field(default_factory=lambda: V3dUnbiasedVolsdfField)
    

class V3dUnbiasedVolsdfField(V3dVolsdfField):
    geometry_type: Literal['density', 'sdf'] = 'sdf'
    config: V3dUnbiasedVolsdfFieldConfig

    def populate_geometry_modules(self) -> None:
        super().populate_geometry_modules()
        self.laplace_density_layer = UnbiasedLaplaceDensity(self.config.laplace_density_init_alpha, self.config.laplace_density_init_beta)

    def forward(self, ray_samples: RaySamples, return_alphas: bool = False) -> Dict[FieldHeadNames, torch.Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_samples)
        if return_alphas:
            
            density = self.laplace_density_layer(field_outputs[FieldHeadNames.SDF], field_outputs[FieldHeadNames.GRADIENT], ray_samples.frustums.directions)
            alpha = self.get_alpha(ray_samples, density)
            weight, trans = self.get_weights_and_transmittance_from_alphas(alpha)
            field_outputs[FieldHeadNames.ALPHA] = alpha
            field_outputs[FieldHeadNames.WEIGHT] = weight
            field_outputs[FieldHeadNames.TRANSMITTANCE] = trans
            field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs