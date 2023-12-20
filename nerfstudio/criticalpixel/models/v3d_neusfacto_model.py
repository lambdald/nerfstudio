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
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import (ProposalNetworkSampler,
                                                      UniformSampler)
from nerfstudio.model_components.renderers import NormalsRenderer
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.v3d.core.fields.v3d_base_field import V3dBaseFieldConfig
from nerfstudio.v3d.core.fields.v3d_nerf_field import V3dNerfFieldConfig
from nerfstudio.v3d.core.fields.v3d_neus_field import V3dNeusFieldConfig
from nerfstudio.v3d.core.metrics.color import RGBLossRunnerConfig
from nerfstudio.v3d.core.metrics.depth import DepthLossRunnerConfig
from nerfstudio.v3d.core.metrics.geometry import EikonalLossRunnerConfig
from nerfstudio.v3d.core.metrics.loss_runner import LossRunnerConfig
from nerfstudio.v3d.core.metrics.weight import InterlevelLossRunnerConfig
from nerfstudio.v3d.core.models.v3d_base_model import (V3dBaseModel,
                                                       V3dModelConfig)
from nerfstudio.v3d.core.models.v3d_nerfacto_model import ProposalSamplerConfig


@dataclass
class V3dNeusfactoModelConfig(V3dModelConfig):
    _target: Type = dataclasses.field(default_factory=lambda: V3dNeusfactoModel)
    """target class to instantiate"""

    loss_runners: List[LossRunnerConfig] = dataclasses.field(
        default_factory=lambda: [
            RGBLossRunnerConfig(loss_weight=1.0),
            DepthLossRunnerConfig(loss_weight=0.1),
            InterlevelLossRunnerConfig(loss_weight=1.0),
            EikonalLossRunnerConfig(loss_weight=0.001),
        ]
    )
    is_unbounded: bool = True
    field: V3dBaseFieldConfig = dataclasses.field(default_factory=lambda: V3dNeusFieldConfig())
    sampler: ProposalSamplerConfig = ProposalSamplerConfig()

    cos_annel_iter: int = 50000

    depth_render_method: Literal['median', 'expected'] = 'expected'
    n_semantic_classes: int = -1

    with_background_model: bool = False
    background_field: V3dBaseFieldConfig = dataclasses.field(default_factory=lambda: V3dNerfFieldConfig())



class V3dNeusfactoModel(V3dBaseModel):
    
    config: V3dNeusfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        # field
        if self.config.is_unbounded:
            CONSOLE.log(f'Use spatial distortion for unbounded scene.')
            self.config.field.use_spatial_distortion = True
        if self.config.field.with_semantic:
            self.config.field.semantic_mlp_config.out_dim = self.config.n_semantic_classes
            self.semantic_color = torch.rand((self.config.n_semantic_classes, 3))
        self.field = self.config.field.setup(box=self.scene_box, num_images=self.num_train_data)
        self.populate_loss_runners()
        self.populate_samplers()

        self.renderer_normal = NormalsRenderer()
        if self.config.with_background_model:
            self.populate_background_modules()
        print(self)


    def populate_background_modules(self):        
        self.config.background_field.use_spatial_distortion = True
        self.background_field = self.config.background_field.setup(box=self.scene_box, num_images=self.num_train_data)
        


    def get_foreground_mask(self, ray_samples: RaySamples):
        xyz = ray_samples.frustums.get_positions()
        aabb = self.scene_box.aabb.to(xyz.device)
        xyz_in_bbox_mask = torch.logical_and(xyz > aabb[0], xyz < aabb[1]).all(dim=-1)
        return xyz_in_bbox_mask.to(xyz)


    def forward_background_field_and_merge(self, ray_samples: RaySamples, field_outputs: Dict) -> Dict:
        """_summary_

        Args:
            ray_samples (RaySamples): _description_
            field_outputs (Dict): _description_
        """

        inside_bbox_mask = self.get_foreground_mask(ray_samples).unsqueeze(-1)
        # TODO only forward the points that are outside the sphere if there is a background model

        field_outputs_bg = self.background_field(ray_samples, return_alphas=True)

        field_outputs[FieldHeadNames.ALPHA] = (
            field_outputs[FieldHeadNames.ALPHA] * inside_bbox_mask
            + (1.0 - inside_bbox_mask) * field_outputs_bg[FieldHeadNames.ALPHA]
        )
        field_outputs[FieldHeadNames.RGB] = (
            field_outputs[FieldHeadNames.RGB] * inside_bbox_mask
            + (1.0 - inside_bbox_mask) * field_outputs_bg[FieldHeadNames.RGB]
        )
        if FieldHeadNames.SEMANTICS in field_outputs:
            field_outputs[FieldHeadNames.SEMANTICS] = (            field_outputs[FieldHeadNames.SEMANTICS] * inside_bbox_mask
            + (1.0 - inside_bbox_mask) * field_outputs_bg[FieldHeadNames.SEMANTICS]
        )

        # update weight
        field_outputs[FieldHeadNames.WEIGHT], field_outputs[FieldHeadNames.TRANSMITTANCE] = self.field.get_weights_and_transmittance_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        return field_outputs


    def populate_samplers(self) -> None:

        self.density_fns = []
        num_prop_nets = self.config.sampler.num_proposal_nets
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        
        for i in range(num_prop_nets):
            if self.config.is_unbounded or self.config.with_background_model:
                CONSOLE.log(f'Use spatial distortion in proposal density field for unbounded scene.')
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
                np.interp(step, [0, self.config.sampler.proposal_warmup], [0, self.config.sampler.proposal_update_every]),
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
        if self.config.with_background_model:
            param_groups["background_fields"] = list(self.background_field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self.config.sampler.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.sampler.proposal_weights_anneal_max_num_iters

            def set_sampler_anneal(step):
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
                    func=set_sampler_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        # annel cos
        def set_cos_anneal(step):
            anneal = min([1.0, step / self.config.cos_annel_iter])
            self.field.set_cos_anneal_ratio(anneal)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_cos_anneal,
            )
        )

        return callbacks


    @torch.no_grad()
    def debug_render_info(self, ray_samples: RaySamples, weights, field_outputs: Dict):
        if hasattr(self, 'debug_iter'):
            self.debug_iter += 1
        else:
            self.debug_iter = 0

        if self.debug_iter % 100 != 0:
            return

        import plotext

        # print debug info.

        weights = weights.detach().squeeze(-1)
        sdf = field_outputs[FieldHeadNames.SDF].squeeze(-1).detach()
        sample_distances = (ray_samples.frustums.starts + ray_samples.frustums.starts)*0.5
        sample_distances = sample_distances.squeeze(-1).detach()
        alpha = field_outputs[FieldHeadNames.ALPHA].squeeze(-1).detach()

        print('weight info:')
        print(f'\tweight max={torch.max(weights).item()} min={torch.min(weights).item()}')
        print(weights.shape, sdf.shape, sample_distances.shape)
        # plotext.canvas_color('black')
        # plotext.axes_color('gray')
        plotext.scatter(sample_distances.detach()[0].tolist(), weights[0].tolist(), label=f'distance-weight min={weights[0].min()} max={weights[0].max()}')
        plotext.scatter(sample_distances.detach()[0].tolist(), sdf[0].tolist(), label=f'distance-sdf min={sdf[0].min()} max={sdf[0].max()}')
        plotext.scatter(sample_distances.detach()[0].tolist(), alpha[0].tolist(), label=f'distance-alpha min={alpha[0].min()} max={alpha[0].max()}')
        plotext.show()
        plotext.clear_data()
        print('\n'*5)


    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights = field_outputs[FieldHeadNames.WEIGHT]


        if self.config.with_background_model:
            field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        # if not self.training:
        #     self.debug_render_info(ray_samples, weights, field_outputs)

        # field_outputs = self.field(ray_samples)
        # weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_acc(weights=weights)
        normal = self.renderer_normal(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal
        }

        if FieldHeadNames.SEMANTICS in field_outputs:
            outputs["semantic"] = self.renderer_semantic(field_outputs[FieldHeadNames.SEMANTICS], weights=weights)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list


        for i in range(self.config.sampler.num_proposal_nets):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])


        if self.training:
            outputs['ray_bundle'] = ray_bundle
            outputs['ray_samples'] = ray_samples
            outputs['field_outputs'] = field_outputs

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return outputs


    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)

        metrics_dict.update(self.field.get_metrics_dict())
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        for loss_runner in self.loss_runners:
            loss_dict.update(loss_runner(self, batch, outputs.get('ray_bundle', None), outputs.get('ray_samples'), outputs, outputs.get('field_outputs')))
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

        # normal_vis
        normal = outputs["normal"]
        normal = (normal + 1.0) / 2.0
        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)
        images_dict['normal'] = combined_normal



        return metrics_dict, images_dict


