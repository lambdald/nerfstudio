"""
Description: file content
"""

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import DepthLossType, depth_loss
from nerfstudio.models.base_model import Model
from nerfstudio.criticalpixel.metrics.loss_runner import LossRunner, LossRunnerConfig


@dataclass
class DepthLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: DepthLossRunner)
    loss_name: str = "depth"
    is_euclidean: bool = False
    loss_type: DepthLossType = DepthLossType.DS_NERF
    sigma: float = 0.1
    loss_weight: float = 0.1
    max_steps: int = -1
    final_factor: float = 1e-3


class DepthLossRunner(LossRunner):
    config: DepthLossRunnerConfig

    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        if self.config.loss_weight <= 0.0:
            return {}

        loss = 0.0
        if "weights_list" in model_output:
            # proposal sampler
            for i in range(len(model_output["weights_list"])):
                loss += depth_loss(
                    weights=model_output["weights_list"][i],
                    ray_samples=model_output["ray_samples_list"][i],
                    termination_depth=batch["depth_image"],
                    predicted_depth=model_output["depth"],
                    sigma=torch.tensor(self.config.sigma),
                    directions_norm=ray_bundle.metadata["directions_norm"],
                    is_euclidean=self.config.is_euclidean,
                    depth_loss_type=self.config.loss_type,
                ) / len(model_output["weights_list"])

        else:
            if "weights" not in model_output:
                return {}
            loss += depth_loss(
                weights=model_output["weights"],
                ray_samples=model_output["ray_samples"],
                termination_depth=batch["depth_image"],
                predicted_depth=model_output["depth"],
                sigma=torch.tensor(self.config.sigma),
                directions_norm=ray_bundle.metadata["directions_norm"],
                is_euclidean=self.config.is_euclidean,
                depth_loss_type=self.config.loss_type,
            )

        if self.config.max_steps > 0 and self.step > self.config.max_steps:
            factor = self.config.final_factor
        else:
            factor = 1.0

        loss = loss * factor

        return {self.config.loss_name: loss}
