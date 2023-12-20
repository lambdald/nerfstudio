from dataclasses import dataclass


from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal
from jinja2 import is_undefined

import torch
from torch import nn
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.criticalpixel.metrics.color import RGBLossRunnerConfig
from nerfstudio.criticalpixel.metrics.loss_runner import LossRunnerConfig
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class V3dModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: V3dBaseModel)
    """target class to instantiate"""

    loss_runners: List[LossRunnerConfig] = field(default_factory=lambda: [RGBLossRunnerConfig()])
    is_unbounded: bool = True
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    collider_type: Literal["near_far", "aabb"] = "near_far"
    depth_render_method: Literal["median", "expected"] = "median"
    near_plane: float = 0.01
    far_plane: float = 100.0


class V3dBaseModel(Model):
    config: V3dModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        if self.config.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        elif self.config.collider_type == "aabb":
            if self.config.is_unbounded:
                CONSOLE.print(f"Warning: you are using aabb in unbounded scene.")
            self.collider = AABBBoxCollider(self.scene_box, self.config.near_plane)
        else:
            raise NotImplementedError(self.config.collider_type)

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_depth = DepthRenderer(method=self.config.depth_render_method)
        self.renderer_acc = AccumulationRenderer()
        self.renderer_semantic = SemanticRenderer()

    def populate_samplers(self) -> None:
        pass

    def populate_loss_runners(self):
        self.loss_runners = torch.nn.ModuleList()
        for loss_runner_config in self.config.loss_runners:
            loss_runner = loss_runner_config.setup()
            self.loss_runners.append(loss_runner)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.field.set_step,
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
        return {}

    def get_metrics_dict(self, outputs, batch):
        return {}

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # depth
        if "depth_image" in batch:
            ground_truth_depth = batch["depth_image"].to(self.device)
            ground_truth_depth = ground_truth_depth * outputs["directions_norm"]
            ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)

            predicted_depth_colormap = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
                near_plane=float(torch.min(ground_truth_depth).cpu()),
                far_plane=float(torch.max(ground_truth_depth).cpu()),
            )
            images_dict["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)

            depth_mask = ground_truth_depth > 0
            metrics_dict["depth_mse"] = float(
                torch.nn.functional.mse_loss(outputs["depth"][depth_mask], ground_truth_depth[depth_mask]).cpu()
            )
        else:
            predicted_depth_colormap = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            images_dict["depth"] = predicted_depth_colormap

        # semantic
        if self.config.field.with_semantic and "semantic" in batch:
            gt_semantic_colormap = self.semantic_color.to(self.device)[batch["semantic"].squeeze(-1)]
            pred_semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantic"], dim=-1), dim=-1)
            pred_semantic_colormap = self.semantic_color.to(self.device)[pred_semantic_labels]
            images_dict["semantic"] = torch.cat([gt_semantic_colormap, pred_semantic_colormap], dim=1)

        return metrics_dict, images_dict
