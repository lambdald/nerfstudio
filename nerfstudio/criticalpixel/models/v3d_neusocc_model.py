import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Type

import nerfacc
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
from nerfstudio.model_components.renderers import NormalsRenderer
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.v3d.core.fields.v3d_base_field import V3dBaseFieldConfig
from nerfstudio.v3d.core.fields.v3d_nerf_field import V3dNerfFieldConfig
from nerfstudio.v3d.core.fields.v3d_neus_field import V3dNeusFieldConfig
from nerfstudio.v3d.core.metrics.color import RGBLossRunnerConfig
from nerfstudio.v3d.core.metrics.depth import DepthLossRunnerConfig
from nerfstudio.v3d.core.metrics.geometry import EikonalLossRunnerConfig
from nerfstudio.v3d.core.metrics.loss_runner import LossRunnerConfig
from nerfstudio.v3d.core.metrics.weight import InterlevelLossRunnerConfig
from nerfstudio.v3d.core.model_components.ray_sampler import \
    VolumetricSamplerConfig
from nerfstudio.v3d.core.models.v3d_base_model import (V3dBaseModel,
                                                       V3dModelConfig)


@dataclass
class V3dNeusoccModelConfig(V3dModelConfig):
    """
    Neus with Occupancy gird.
    """
    _target: Type = dataclasses.field(default_factory=lambda: V3dNeusoccModel)
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
    sampler: VolumetricSamplerConfig = VolumetricSamplerConfig(
        occ_type='alpha',
        occ_resolution=256,
        occ_level=1,
    )
    cos_annel_iter: int = 50000

    depth_render_method: Literal['median', 'expected'] = 'expected'
    n_semantic_classes: int = -1

    with_background_model: bool = False
    background_field: V3dBaseFieldConfig = dataclasses.field(default_factory=lambda: V3dNerfFieldConfig())



class V3dNeusoccModel(V3dBaseModel):
    
    config: V3dNeusoccModelConfig

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
            field_outputs[FieldHeadNames.SEMANTICS] = (field_outputs[FieldHeadNames.SEMANTICS] * inside_bbox_mask
            + (1.0 - inside_bbox_mask) * field_outputs_bg[FieldHeadNames.SEMANTICS]
        )

        # update weight
        field_outputs[FieldHeadNames.WEIGHT], field_outputs[FieldHeadNames.TRANSMITTANCE] = self.field.get_weights_and_transmittance_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        return field_outputs


    def populate_samplers(self) -> None:
        self.sampler = self.config.sampler.setup(aabb=self.scene_box.aabb, occ_fn=self.field.get_alpha_for_sampler)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["fields"] = list(self.field.parameters())
        if self.config.with_background_model:
            param_groups["background_fields"] = list(self.background_field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
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

        def set_occ_ray_marching_step_size(step):
            inv_s = self.field.deviation_network().clip(1e-6, 1e6).item()
            step_size = 2.5 / math.sqrt(inv_s)
            self.sampler.set_render_step_size(step_size)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_occ_ray_marching_step_size,
            )
        )
        
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.update_occupancy_grid,
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

        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(ray_bundle=ray_bundle)

        field_outputs = self.field(ray_samples, return_alphas=True)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        weights = nerfacc.render_weight_from_alpha(
            packed_info=packed_info,
            alphas=field_outputs[FieldHeadNames.ALPHA].squeeze(-1)
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_acc(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        normal = self.renderer_normal(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        # if self.config.with_background_model:
        #     field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        # if not self.training:
        #     self.debug_render_info(ray_samples, weights, field_outputs)


        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "num_samples_per_ray": packed_info[:, 1],
        }

        if FieldHeadNames.SEMANTICS in field_outputs:
            outputs["semantic"] = self.renderer_semantic(field_outputs[FieldHeadNames.SEMANTICS], weights=weights)


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
        metrics_dict['num_samples_per_batch'] = outputs['num_samples_per_ray'].sum()
        depth = outputs['depth']
        CONSOLE.print(f'depth: max={depth.max().item()}, min={depth.min().item()}')

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
