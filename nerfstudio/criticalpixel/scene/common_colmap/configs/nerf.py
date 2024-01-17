"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.criticalpixel.data.dataparser.colmap_parser import ColmapDataparserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components.losses import DepthLossType
from nerfstudio.criticalpixel.pipeline.nerf_pipeline import NeRFPipelineConfig

from nerfstudio.criticalpixel.data.datamanager.nerf_data_manager import NeRFDataManagerConfig

from nerfstudio.criticalpixel.models.nerfacto_model import NerfactoModelConfig


from nerfstudio.criticalpixel.field_components.mlp import MLPConfig
from nerfstudio.criticalpixel.field_components.encodings import TcnnGridEncoderConfig, TcnnSphereHarmonicsEncoderConfig
from nerfstudio.criticalpixel.metrics.color import RGBLossRunnerConfig
from nerfstudio.criticalpixel.metrics.depth import DepthLossRunnerConfig
from nerfstudio.criticalpixel.metrics.weight import DistortionLossRunnerConfig, InterlevelLossRunnerConfig
from nerfstudio.criticalpixel.models.nerfacto_model import (
    ProposalSamplerConfig,
    NerfactoModelConfig,
    NerfFieldConfig,
)

nerf_method_configs: Dict[str, TrainerConfig] = {}
nerf_descriptions = {
    "meta-colmap-nerfacto": "meta nerf model for render",
}

nerf_method_configs["meta-colmap-nerfacto"] = TrainerConfig(
    method_name="meta-colmap-nerfacto",
    steps_per_save=2000,
    max_num_iterations=100000,
    steps_per_eval_batch=50000,
    steps_per_eval_image=50000,
    mixed_precision=True,
    pipeline=NeRFPipelineConfig(
        datamanager=NeRFDataManagerConfig(
            dataparser=ColmapDataparserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            train_num_images_to_sample_from=3,
            train_num_times_to_repeat_images=500,
            use_mask=True,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            collider_type="near_far",
            is_unbounded=False,
            sampler=ProposalSamplerConfig(
                proposal_weights_anneal_max_num_iters=5000,
                num_nerf_samples_per_ray=64,
                num_proposal_samples_per_ray=(512, 256),
                proposal_nets=[
                    NerfFieldConfig(
                        use_spatial_distortion=True,
                        spatial_distortion_bound=1.2,
                        is_geometry_field=True,
                        position_encoder_config=TcnnGridEncoderConfig(
                            type="hashgrid", n_levels=5, log2_hashmap_size=17, desired_resolution=256
                        ),
                        geometry_mlp_config=MLPConfig(num_layers=2, layer_width=16),
                    ),
                    NerfFieldConfig(
                        use_spatial_distortion=True,
                        spatial_distortion_bound=1.2,
                        is_geometry_field=True,
                        position_encoder_config=TcnnGridEncoderConfig(
                            type="hashgrid", n_levels=5, log2_hashmap_size=17, desired_resolution=512
                        ),
                        geometry_mlp_config=MLPConfig(num_layers=2, layer_width=16),
                    ),
                ],
            ),
            field=NerfFieldConfig(
                use_spatial_distortion=True,
                spatial_distortion_bound=1.2,
                use_direction=True,
                use_appearance_embedding=True,
                n_appearance_embedding_dim=8,
                use_average_appearance_embedding=True,
                is_geometry_field=False,
                position_encoder_config=TcnnGridEncoderConfig(
                    type="hashgrid",
                    n_levels=16,
                    n_features_per_level=2,
                    log2_hashmap_size=21,
                    base_resolution=16,
                    desired_resolution=7000,
                ),
            ),
            depth_render_method="expected",
            near_plane=1,
            far_plane=100.0,
            loss_runners=[
                RGBLossRunnerConfig(loss_weight=1.0, criterion="l1"),
                DistortionLossRunnerConfig(loss_weight=0.002),
                InterlevelLossRunnerConfig(loss_weight=1),
            ],
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer+tensorboard",
)
