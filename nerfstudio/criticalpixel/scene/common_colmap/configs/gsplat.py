"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.criticalpixel.data.datamanager.gsplat_data_manager import GSplatDataManagerConfig
from nerfstudio.criticalpixel.data.dataparser.colmap_parser import ColmapDataparserConfig
from nerfstudio.criticalpixel.models.gaussian_splatting import GaussianSplattingModelConfig
from nerfstudio.criticalpixel.pipeline.gsplat_pipeline import GSplatPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig

gsplat_method_configs: Dict[str, TrainerConfig] = {}
gsplat_descriptions = {
    "colmap-gsplat": "gsplat model for render",
}

gsplat_method_configs["colmap-gsplat"] = TrainerConfig(
    method_name="colmap-gsplat",
    steps_per_save=5000,
    max_num_iterations=30000,
    steps_per_eval_batch=1000,
    steps_per_eval_image=1000,
    mixed_precision=False,
    pipeline=GSplatPipelineConfig(
        datamanager=GSplatDataManagerConfig(
            dataparser=ColmapDataparserConfig(),
            train_num_images_to_sample_from=3,
            train_num_times_to_repeat_images=500,
            use_mask=False,
        ),
        model=GaussianSplattingModelConfig(
            near=0.1,
            far=200.0,
            # loss_runners=[
            #     RGBLossRunnerConfig(loss_weight=1.0, criterion="l1"),
            #     DistortionLossRunnerConfig(loss_weight=0.002),
            #     InterlevelLossRunnerConfig(loss_weight=1),
            # ],
        ),
    ),
    optimizers={
        "xyz": {
            "optimizer": AdamOptimizerConfig(lr=0.00016, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=0.0000016,
                max_steps=30000,
            ),
        },
        "feature_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "feature_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scale": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)
