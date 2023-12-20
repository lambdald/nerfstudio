import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.criticalpixel.field_components.mlp import MLPConfig

from nerfstudio.criticalpixel.fields.nerf_field import NerfFieldConfig
from nerfstudio.criticalpixel.metrics.color import RGBLossRunnerConfig
from nerfstudio.criticalpixel.metrics.depth import DepthLossRunnerConfig
from nerfstudio.criticalpixel.metrics.loss_runner import LossRunnerConfig
from nerfstudio.criticalpixel.metrics.weight import (
    DistortionLossRunnerConfig,
    InterlevelLossRunnerConfig,
    SkyLossRunnerConfig,
)
from nerfstudio.criticalpixel.models.base_model import V3dBaseModel, V3dModelConfig
from nerfstudio.criticalpixel.field_components.encodings import TcnnGridEncoderConfig, TcnnSphereHarmonicsEncoderConfig

@dataclass
class ProposalSamplerConfig:
    proposal_nets: List[NerfFieldConfig] = dataclasses.field(
        default_factory=lambda: [
            NerfFieldConfig(
                is_geometry_field=True,
                position_encoder_config=TcnnGridEncoderConfig(log2_hashmap_size=17, n_levels=5, desired_resolution=128),
                geometry_mlp_config=MLPConfig(num_layers=2, layer_width=16, implementation="tcnn"),
            ),
            NerfFieldConfig(
                is_geometry_field=True,
                position_encoder_config=TcnnGridEncoderConfig(log2_hashmap_size=17, n_levels=5, desired_resolution=256),
                geometry_mlp_config=MLPConfig(num_layers=2, layer_width=16, implementation="tcnn"),
            ),
        ]
    )

    proposal_initial_sampler: Literal["piecewise", "uniform"] = "uniform"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""

    use_single_jitter: bool = True
    num_proposal_samples_per_ray: Tuple[int, ...] = dataclasses.field(default_factory=lambda: (256, 512))
    num_nerf_samples_per_ray: int = 48

    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000

    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    def __post_init__(self):
        self.num_proposal_nets = len(self.proposal_nets)


@dataclass
class NerfactoModelConfig(V3dModelConfig):
    _target: Type = dataclasses.field(default_factory=lambda: NerfactoModel)
    """target class to instantiate"""

    loss_runners: List[LossRunnerConfig] = dataclasses.field(
        default_factory=lambda: [
            RGBLossRunnerConfig(),
            DepthLossRunnerConfig(),
            InterlevelLossRunnerConfig(),
            SkyLossRunnerConfig(loss_weight=1),
            DistortionLossRunnerConfig(loss_weight=1),
        ]
    )
    is_unbounded: bool = True
    field: NerfFieldConfig = dataclasses.field(default_factory=lambda: NerfFieldConfig())
    sampler: ProposalSamplerConfig = ProposalSamplerConfig()
    n_semantic_classes: int = -1


class NerfactoModel(V3dBaseModel):
    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        # field
        if self.config.is_unbounded:
            CONSOLE.log(f"Use spatial distortion for unbounded scene.")
            self.config.field.use_spatial_distortion = True
        if self.config.field.with_semantic:
            self.config.field.semantic_mlp_config.out_dim = self.config.n_semantic_classes
            self.semantic_color = torch.rand((self.config.n_semantic_classes, 3))
        self.field = self.config.field.setup(box=self.scene_box, num_images=self.num_train_data)
        self.populate_loss_runners()
        self.populate_samplers()

        CONSOLE.print(self)

    def populate_samplers(self) -> None:
        self.density_fns = []
        num_prop_nets = self.config.sampler.num_proposal_nets
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()

        for i in range(num_prop_nets):
            if self.config.is_unbounded:
                CONSOLE.log(f"Use spatial distortion in proposal density field for unbounded scene.")
                self.config.sampler.proposal_nets[i].use_spatial_distortion = True

            network = self.config.sampler.proposal_nets[i].setup(box=self.scene_box, num_images=self.num_train_data)
            self.proposal_networks.append(network)
            self.density_fns.append(network.get_density)

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.sampler.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.sampler.use_single_jitter)

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(
                    step, [0, self.config.sampler.proposal_warmup], [0, self.config.sampler.proposal_update_every]
                ),
                1,
                self.config.sampler.proposal_update_every,
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.sampler.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.sampler.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.sampler.num_proposal_nets,
            single_jitter=self.config.sampler.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self.config.sampler.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.sampler.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.sampler.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        for loss_runner in self.loss_runners:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=loss_runner.set_step,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights = field_outputs[FieldHeadNames.WEIGHT]

        # field_outputs = self.field(ray_samples)
        # weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_acc(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if FieldHeadNames.SEMANTICS in field_outputs:
            outputs["semantic"] = self.renderer_semantic(field_outputs[FieldHeadNames.SEMANTICS], weights=weights)
            print("semantic", outputs["semantic"])

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.sampler.num_proposal_nets):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if self.training:
            outputs["ray_bundle"] = ray_bundle
            outputs["ray_samples"] = ray_samples
            outputs["field_outputs"] = field_outputs

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        for loss_runner in self.loss_runners:
            loss_dict.update(
                loss_runner(
                    self,
                    batch,
                    outputs.get("ray_bundle", None),
                    outputs.get("ray_samples"),
                    outputs,
                    outputs.get("field_outputs"),
                )
            )
            loss_dict.update(loss_runner.loss_multiply(loss_dict))
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        for i in range(self.config.sampler.num_proposal_nets):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
