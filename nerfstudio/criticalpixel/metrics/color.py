"""
Description: file content
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Type

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.criticalpixel.metrics.loss_runner import LossRunner, LossRunnerConfig
from nerfstudio.models.base_model import Model


@dataclass
class RGBLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: RGBLossRunner)
    loss_name: str = "rgb"
    criterion: Literal["l1", "mse", "huber"] = "l1"


class RGBLossRunner(LossRunner):
    def __init__(self, config: RGBLossRunnerConfig) -> None:
        super().__init__(config)

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

        rgb_gt = batch["image"]
        rgb_pred = model_output["rgb"]
        loss = self.criterion(rgb_gt.to(rgb_pred.device), rgb_pred)

        return {self.config.loss_name: loss}


@dataclass
class SSIMLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: SSIMLossRunner)
    loss_name: str = "dssim"


class SSIMLossRunner(LossRunner):

    """
    d-ssim(SSIM Dissimilarity)=(1-ssim) * 0.5
    ssim range = (-1, 1)

    loss range = (0.0, 1.0)
    """

    def __init__(self, config: SSIMLossRunnerConfig) -> None:
        super().__init__(config)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0))

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
        # [N, C, H, W]
        rgb_gt = batch["image"]
        rgb_pred = model_output["rgb"]
        ssim: torch.Tensor = self.ssim(rgb_pred, rgb_gt.to(rgb_pred.device))  # ignore: type
        loss = (1 - ssim) * 0.5
        return {self.config.loss_name: loss}


@dataclass
class S3IMLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: S3IMLossRunner)
    loss_name: str = "s3im"
    kernel_size: int = 4
    stride: int = 4
    repeat_time: int = 10
    patch_height: int = 64


class S3IMLossRunner(LossRunner):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """

    config: S3IMLossRunnerConfig

    def __init__(self, config: SSIMLossRunnerConfig) -> None:
        super().__init__(config)
        self.ssim = StructuralSimilarityIndexMeasure(kernel_size=self.config.kernel_size, data_range=(0.0, 1.0))

    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        # [N, 3]
        pred_rgb = model_output["rgb"]
        gt_rgb = batch["image"].to(pred_rgb.device)

        batch_size = pred_rgb.shape[0]

        loss = 0.0
        index_list = []
        for i in range(self.config.repeat_time):
            if i == 0:
                tmp_index = torch.arange(batch_size)
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(batch_size)
                index_list.append(ran_idx)
        # [k, N]
        res_index = torch.cat(index_list)
        gt_patch = gt_rgb[res_index]
        pred_patch = pred_rgb[res_index]
        gt_patch = gt_patch.permute(1, 0).reshape(1, 3, self.config.patch_height, -1)
        pred_patch = pred_patch.permute(1, 0).reshape(1, 3, self.config.patch_height, -1)
        ssim = self.ssim(gt_patch, pred_patch)
        loss = (1 - ssim) * 0.5
        return {self.config.loss_name: loss}
