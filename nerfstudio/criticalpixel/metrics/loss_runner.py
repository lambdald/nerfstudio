"""
Description: file content
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Type
from abc import ABC, abstractmethod

import torch
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.model_components.losses import MSELoss, L1Loss
from nerfstudio.models.base_model import Model
from nerfstudio.cameras.rays import RayBundle, RaySamples


class HuberLoss(torch.nn.Module):
    def __init__(self, delta=0.1):
        super().__init__()
        self.criterion = torch.nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self.criterion(pred, target)


@dataclass
class LossRunnerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: LossRunner)
    loss_weight: float = 1.0
    """loss weight"""
    criterion: Literal["l1", "mse", "huber"] = "mse"
    loss_name: str = "none"


class LossRunner(torch.nn.Module, ABC):
    def __init__(self, config: LossRunnerConfig) -> None:
        """_summary_

        Args:
            config (Dict): data in config {'loss_config':Dict, ...}
        """
        super().__init__()
        self.config = config
        self.step = 0
        self.init(self.config)

    def build_criterion(self):
        if self.config.criterion == "l1":
            return L1Loss()
        elif self.config.criterion == "mse":
            return MSELoss()
        elif self.config.criterion == "huber":
            return HuberLoss()
        else:
            raise NotImplementedError(f"unknown criterion type: {self.config.criterion}")

    def init(self, config):
        self.criterion = self.build_criterion()

    def loss_multiply(self, metric_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.config.loss_name not in metric_dict:
            return {}
        return {self.config.loss_name: metric_dict[self.config.loss_name] * self.config.loss_weight}

    def set_step(self, step):
        self.step = step

    @abstractmethod
    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        pass
