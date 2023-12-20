import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

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
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.v3d.core.fields.v3d_nerf_field import V3dNerfFieldConfig
from nerfstudio.v3d.core.metrics.color import RGBLossRunnerConfig
from nerfstudio.v3d.core.metrics.depth import DepthLossRunnerConfig
from nerfstudio.v3d.core.metrics.loss_runner import LossRunnerConfig
from nerfstudio.v3d.core.metrics.weight import (DistortionLossRunnerConfig,
                                                InterlevelLossRunnerConfig,
                                                SkyLossRunnerConfig)
from nerfstudio.v3d.core.model_components.ray_sampler import \
    NeRFHierarchicalSamplerConfig
from nerfstudio.v3d.core.models.v3d_base_model import (V3dBaseModel,
                                                       V3dModelConfig)


@dataclass
class V3dNerfModelConfig(V3dModelConfig):
    _target: Type = dataclasses.field(default_factory=lambda: V3dNerfModel)
    """target class to instantiate"""

    loss_runners: List[LossRunnerConfig] = dataclasses.field(
        default_factory=lambda: [
            RGBLossRunnerConfig(),
            DistortionLossRunnerConfig(loss_weight=0.001),
        ]
    )
    is_unbounded: bool = True
    field: V3dNerfFieldConfig = dataclasses.field(default_factory=lambda: V3dNerfFieldConfig())
    sampler: NeRFHierarchicalSamplerConfig = NeRFHierarchicalSamplerConfig(
        num_coarse_samples=256,
        num_fine_samples=128,
        num_fine_steps=2,
    )
    n_semantic_classes: int = -1

class V3dNerfModel(V3dBaseModel):
    
    config: V3dNerfModelConfig

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



        print(self)

    def populate_samplers(self) -> None:
        self.sampler = self.config.sampler.setup()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

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
        ray_samples: RaySamples = self.sampler(self.field, ray_bundle)
        field_outputs = self.field(ray_samples, return_alphas=True)
        weights = field_outputs[FieldHeadNames.WEIGHT]
  
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
            print('semantic', outputs['semantic'])

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

        return metrics_dict, images_dict


