from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss, ray_samples_to_sdist
from nerfstudio.models.base_model import Model
from nerfstudio.criticalpixel.metrics.loss_runner import LossRunner, LossRunnerConfig


## zip-NeRF losses
def blur_stepfun(x, y, r):
    x_c = torch.cat([x - r, x + r], dim=-1)
    x_r, x_idx = torch.sort(x_c, dim=-1)
    zeros = torch.zeros_like(y[:, :1])
    y_1 = (torch.cat([y, zeros], dim=-1) - torch.cat([zeros, y], dim=-1)) / (2 * r)
    x_idx = x_idx[:, :-1]
    y_2 = torch.cat([y_1, -y_1], dim=-1)[
        torch.arange(x_idx.shape[0]).reshape(-1, 1).expand(x_idx.shape).to(x_idx.device), x_idx
    ]

    y_r = torch.cumsum((x_r[:, 1:] - x_r[:, :-1]) * torch.cumsum(y_2, dim=-1), dim=-1)
    y_r = torch.cat([zeros, y_r], dim=-1)
    return x_r, y_r


def interlevel_loss_zip(weights_list, ray_samples_list):
    """Calculates the proposal loss in the Zip-NeRF paper."""
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()

    # 1. normalize
    w_normalize = w / (c[:, 1:] - c[:, :-1])

    loss_interlevel = 0.0
    for ray_samples, weights, r in zip(ray_samples_list[:-1], weights_list[:-1], [0.03, 0.003]):
        # 2. step blur with different r
        x_r, y_r = blur_stepfun(c, w_normalize, r)
        y_r = torch.clip(y_r, min=0)
        # assert (y_r >= 0.0).all()

        # 3. accumulate
        y_cum = torch.cumsum((y_r[:, 1:] + y_r[:, :-1]) * 0.5 * (x_r[:, 1:] - x_r[:, :-1]), dim=-1)
        y_cum = torch.cat([torch.zeros_like(y_cum[:, :1]), y_cum], dim=-1)

        # 4 loss
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)

        # resample
        inds = torch.searchsorted(x_r, cp, side="right")
        below = torch.clamp(inds - 1, 0, x_r.shape[-1] - 1)
        above = torch.clamp(inds, 0, x_r.shape[-1] - 1)
        cdf_g0 = torch.gather(x_r, -1, below)
        bins_g0 = torch.gather(y_cum, -1, below)
        cdf_g1 = torch.gather(x_r, -1, above)
        bins_g1 = torch.gather(y_cum, -1, above)

        t = torch.clip(torch.nan_to_num((cp - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        w_gt = bins[:, 1:] - bins[:, :-1]

        # TODO here might be unstable when wp is very small
        loss_interlevel += torch.mean(torch.clip(w_gt - wp, min=0) ** 2 / (wp + 1e-5))

    return loss_interlevel


@dataclass
class InterlevelLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: InterlevelLossRunner)
    loss_weight: float = 1.0
    use_zip_interlevel_loss: bool = True
    loss_name: str = "interlevel_loss"


class InterlevelLossRunner(LossRunner):
    config: InterlevelLossRunnerConfig

    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        if not self.training:
            return {}

        loss_dict = {}
        if self.config.use_zip_interlevel_loss:
            interlevel_loss_func = interlevel_loss_zip
        else:
            interlevel_loss_func = interlevel_loss

        loss_dict[self.config.loss_name] = interlevel_loss_func(
            model_output["weights_list"], model_output["ray_samples_list"]
        )
        return loss_dict


@dataclass
class DistortionLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: DistortionLossRunner)
    loss_weight: float = 0.002
    use_zip_interlevel_loss: bool = True
    loss_name: str = "distortion"


class DistortionLossRunner(LossRunner):
    config: DistortionLossRunnerConfig

    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        if not self.training:
            return {}

        if "weights_list" in model_output:
            # proposal sampler
            distortion = distortion_loss(model_output["weights_list"], model_output["ray_samples_list"])
        else:
            if "weights" not in model_output:
                return {}
            distortion = distortion_loss([model_output["weights"]], [model_output["ray_samples"]])

        return {self.config.loss_name: distortion}


@dataclass
class SkyLossRunnerConfig(LossRunnerConfig):
    _target: Type = field(default_factory=lambda: SkyLossRunner)
    loss_name: str = "sky_loss"
    loss_weight: float = 1


class SkyLossRunner(LossRunner):
    config: SkyLossRunnerConfig

    def forward(
        self,
        model: Model,
        batch: Dict[str, torch.Tensor],
        ray_bundle: RayBundle,
        ray_sample: RaySamples,
        model_output: Dict,
        field_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        if not self.training:
            return {}

        sky_loss = 0.0
        termination_depth = batch["depth_image"].to(model.device)
        if "weights_list" in model_output:
            for i in range(len(model_output["weights_list"])):
                sky_loss += urban_radiance_sky_segment_loss(
                    weights=model_output["weights_list"][i],  # depth weighted sum
                    termination_depth=termination_depth,
                ) / len(model_output["weights_list"])
        else:
            if "weights" not in model_output:
                return {}
            sky_loss += urban_radiance_sky_segment_loss(
                weights=model_output["weights"],  # depth weighted sum
                termination_depth=termination_depth,
            )
        return {self.config.loss_name: sky_loss}
