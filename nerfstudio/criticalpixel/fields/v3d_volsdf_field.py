from dataclasses import dataclass, field
from typing import Dict, Literal, Type, Union

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.v3d.core.fields.v3d_neus_field import (V3dNeusField,
                                                       V3dNeusFieldConfig)


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    alpha: torch.Tensor
    beta: torch.Tensor

    def __init__(self, init_alpha, init_beta, min_value=0.0001):
        super().__init__()
        self.min_value = min_value
        # 两个同时学比只学beta效果好
        self.register_parameter("alpha", nn.Parameter(torch.tensor(init_alpha), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(torch.tensor(init_beta), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()
        alpha = self.get_alpha()
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.min_value
        return beta

    def get_alpha(self):
        alpha = self.alpha.abs() + self.min_value
        return alpha


@dataclass
class V3dVolsdfFieldConfig(V3dNeusFieldConfig):
    _target: Type=field(default_factory=lambda: V3dVolsdfField)
    
    laplace_density_init_beta: float = 1.0
    laplace_density_init_alpha: float = 1.0

class V3dVolsdfField(V3dNeusField):
    geometry_type: Literal['density', 'sdf'] = 'sdf'
    config: V3dVolsdfFieldConfig

    def populate_geometry_modules(self) -> None:
        super().populate_geometry_modules()
        self.laplace_density_layer = LaplaceDensity(self.config.laplace_density_init_alpha, self.config.laplace_density_init_beta)


    def forward(self, ray_samples: RaySamples, return_alphas: bool = False) -> Dict[FieldHeadNames, torch.Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_samples)
        if return_alphas:
            
            density = self.laplace_density_layer(field_outputs[FieldHeadNames.SDF])
            alpha = self.get_alpha(ray_samples, density)
            weight, trans = self.get_weights_and_transmittance_from_alphas(alpha)
            field_outputs[FieldHeadNames.ALPHA] = alpha
            field_outputs[FieldHeadNames.WEIGHT] = weight
            field_outputs[FieldHeadNames.TRANSMITTANCE] = trans
            field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs

    def get_alpha(self, ray_samples: RaySamples, densities: torch.Tensor) -> torch.Tensor:
        delta_density = ray_samples.deltas * densities
        alphas = 1 - torch.exp(-delta_density)
        return alphas

    def get_metrics_dict(self) -> Dict:
        metrics_dict = super().get_metrics_dict()
        if 's_val' in metrics_dict:
            del metrics_dict['s_val']
        metrics_dict['laplace_density_alpha'] = self.laplace_density_layer.get_alpha()
        metrics_dict['laplace_density_beta'] = self.laplace_density_layer.get_beta()
        return metrics_dict