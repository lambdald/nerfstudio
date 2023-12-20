'''
Description: file content
'''

from dataclasses import dataclass, field
from typing import Dict, Type
import torch

from nerfstudio.criticalpixel.metrics.loss_runner import LossRunnerConfig, LossRunner
from nerfstudio.models.base_model import Model
from nerfstudio.cameras.rays import RayBundle, RaySamples

@dataclass
class RGBLossRunnerConfig(LossRunnerConfig):
    _target: Type=field(default_factory=lambda: RGBLossRunner)
    loss_name: str = 'rgb'

class RGBLossRunner(LossRunner):

    def __init__(self, config: RGBLossRunnerConfig) -> None:
        super().__init__(config)


    def forward(self, model: Model, batch: Dict[str, torch.Tensor], ray_bundle: RayBundle, ray_sample: RaySamples, model_output: Dict, field_output: Dict) -> Dict[str, torch.Tensor]:
        
        if self.config.loss_weight <= 0.:
            return {}
        
        rgb_gt = batch['image']
        rgb_pred = model_output['rgb']
        loss = self.criterion(rgb_gt.to(rgb_pred.device), rgb_pred)

        return {
            self.config.loss_name: loss
        }
