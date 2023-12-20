from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import (
    distortion_loss, interlevel_loss, interlevel_loss_zip,
    urban_radiance_sky_segment_loss)
from nerfstudio.models.base_model import Model
from nerfstudio.v3d.core.metrics.loss_runner import (LossRunner,
                                                     LossRunnerConfig)


        

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.NLLLoss()

    def forward(self, pred: torch.Tensor, target: torch.LongTensor):
        # https://discuss.pytorch.org/t/nan-loss-with-torch-cuda-amp-and-crossentropyloss/108554
        # https://zhuanlan.zhihu.com/p/98785902
        data_type = pred.dtype
        finfo = torch.finfo(data_type)
        log_softmax = torch.log(torch.softmax(pred, dim=-1) + finfo.tiny)
        return self.criterion(log_softmax, target.squeeze(-1))


@dataclass
class SemanticLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: SemanticLossRunner)
    loss_weight: float = 0.1
    loss_name: str = 'semantic'

class SemanticLossRunner(LossRunner):
    config: SemanticLossRunnerConfig

    def build_criterion(self):
        return CrossEntropyLoss()

    def forward(self, model: Model, batch: Dict[str, torch.Tensor], ray_bundle: RayBundle, ray_sample: RaySamples, model_output: Dict, field_output: Dict) -> Dict[str, torch.Tensor]:
        cross_entropy_loss = self.criterion(model_output['semantic'], batch['semantic'].squeeze(-1))
        return {self.config.loss_name: cross_entropy_loss}
