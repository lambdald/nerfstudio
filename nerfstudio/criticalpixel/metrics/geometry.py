


'''
Description: file content
'''

from dataclasses import dataclass, field
from typing import Dict, Literal, Type, cast

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import Model
from nerfstudio.utils.printing import CONSOLE
from nerfstudio.v3d.core.fields.v3d_base_field import V3dBaseField
from nerfstudio.v3d.core.fields.v3d_neuralangelo_field import \
    V3dNeuralangeloFieldConfig
from nerfstudio.v3d.core.metrics.loss_runner import (LossRunner,
                                                     LossRunnerConfig)
from nerfstudio.v3d.core.models.v3d_base_model import V3dBaseModel


@dataclass
class EikonalLossRunnerConfig(LossRunnerConfig):
    _target: Type=field(default_factory=lambda: EikonalLossRunner)
    loss_name: str = 'eikonal'
    loss_weight: float = 0.1    # weight in neus
    warmup_steps: int = 0
    '''Linearly increase the weight until the specified number of steps'''
    max_steps: int = -1
    '''After exceeding the specified number of steps, the loss becomes invalid.'''

class EikonalLossRunner(LossRunner):

    config: EikonalLossRunnerConfig

    def forward(self, model: Model, batch: Dict[str, torch.Tensor], ray_bundle: RayBundle, ray_sample: RaySamples, model_output: Dict, field_output: Dict) -> Dict[str, torch.Tensor]:
        
        if self.config.loss_weight <= 0. or not self.training:
            return {}
        
        if FieldHeadNames.GRADIENT not in field_output:
            CONSOLE.log('Gradient is not in field_output, and eikonal loss is not available')
            return {}

        gradient = field_output[FieldHeadNames.GRADIENT]
        loss = (gradient.norm(2, dim=-1) - 1) ** 2
        loss = loss.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        factor = 1.0
        if self.config.warmup_steps > 0:
            factor = min(1.0, self.step / self.config.warmup_steps)
        if self.config.max_steps > 0 and self.step > self.config.max_steps:
            factor = 0.0
        return {
            self.config.loss_name: loss.mean() * factor
        }

@dataclass
class CurvatureLossRunnerConfig(LossRunnerConfig):
    _target: Type=field(default_factory=lambda: CurvatureLossRunner)
    loss_name: str = 'curvature'
    loss_weight: float = 5e-4    # weight in neus
    curvature_loss_warmup_steps: int = 2000

class CurvatureLossRunner(LossRunner):

    config: CurvatureLossRunnerConfig

    def forward(self, model: V3dBaseModel , batch: Dict[str, torch.Tensor], ray_bundle: RayBundle, ray_sample: RaySamples, model_output: Dict, field_output: Dict) -> Dict[str, torch.Tensor]:
        
        if self.config.loss_weight <= 0. or not self.training:
            return {}
        
        field_config = cast(V3dNeuralangeloFieldConfig, model.config.field)

        base_res = field_config.position_encoder_config.base_resolution
        per_level_scale = field_config.position_encoder_config.per_level_scale
        max_res = field_config.position_encoder_config.desired_resolution

        init_delta = 1 / base_res

        if self.step < self.config.curvature_loss_warmup_steps:
            factor = self.step / self.config.curvature_loss_warmup_steps
        else:
            delta = 1. / (base_res * per_level_scale ** ( (self.step - self.config.curvature_loss_warmup_steps) / field_config.steps_per_level + field_config.level_init - 1))
            delta = max(1. / max_res, delta)
            factor = delta / init_delta

        curvature: torch.Tensor = field_output[FieldHeadNames.CURVATURE]
        loss = curvature.norm(dim=-1)
        loss = loss.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        loss = loss.mean()
        return {
            self.config.loss_name: loss*factor
        }


@dataclass
class TVLossRunnerConfig(LossRunnerConfig):
    _target: Type=field(default_factory=lambda: TVLossRunner)
    loss_name: str = 'tv'
    loss_weight: float = 1.0    # weight in neus
    resolution: int = 1024
    position_delta: float = 0.01 
    # max_steps: int = -1
    # warmup_steps: int = -1
    criterion: Literal['l1', 'mse', 'huber'] = 'mse'

    position_from: Literal['sampler', 'random'] = 'random'
    num_points: int = 1024
    n_point_dim=3



class TVLossRunner(LossRunner):
    config: TVLossRunnerConfig

    def forward(self, model: V3dBaseModel , batch: Dict[str, torch.Tensor], ray_bundle: RayBundle, ray_sample: RaySamples, model_output: Dict, field_output: Dict) -> Dict[str, torch.Tensor]:

        if not self.training:
            return {}

        nerf_field: V3dBaseField = model.field

        aabb: torch.Tensor = nerf_field.aabb
        device = aabb.device
        size = (aabb[1] - aabb[0]).max().item()
        if self.config.position_delta > 0:
            position_delta = self.config.position_delta
        else:
            position_delta = size / self.config.resolution

        device = aabb.device
        if self.config.position_from == 'sampler':
            xyz = ray_sample.frustums.get_positions()
        else:
            xyz = torch.rand((self.config.num_points, self.config.n_point_dim), device=device) * (aabb[1] - aabb[0]) + aabb[0]


        offsets = position_delta * torch.stack([torch.eye(self.config.n_point_dim, device=device), -torch.eye(self.config.n_point_dim, device=device)], dim=1) # [3, 2, 3]
        positions = xyz.view(xyz.shape[:-1]+(1, 1, self.config.n_point_dim)) + offsets
        # [B, N, 6]
        neighbor_geometry = nerf_field.get_geometry(positions)[0].squeeze(-1)

        if self.config.position_from == 'sampler':
            if nerf_field.geometry_type == 'density':
                geometry = field_output[FieldHeadNames.DENSITY] # [B, N, 1]
            elif nerf_field.geometry_type == 'sdf':
                geometry = field_output[FieldHeadNames.SDF]
            else:
                raise NotImplementedError
            neighbor_geometry = neighbor_geometry.view(xyz.shape[:-1]+(-1, ))
            loss = self.criterion(neighbor_geometry, geometry)
        else:
            neighbor_geometry = neighbor_geometry.squeeze(-1)
            loss = self.criterion(neighbor_geometry[..., 0], neighbor_geometry[..., 1])
        return {
            self.config.loss_name: loss
        }
