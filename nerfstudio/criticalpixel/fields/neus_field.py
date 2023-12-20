from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.fields.base_field import (FieldHeadNames,
                                          shift_directions_for_tcnn)
from nerfstudio.v3d.core.fields.v3d_base_field import (V3dBaseField,
                                                       V3dBaseFieldConfig,
                                                       V3dMLPConfig)


class SingleVarianceLayer(torch.nn.Module):
    variance: torch.Tensor
    # for NeuS(SDF NeRF)
    def __init__(self, init_val):
        super().__init__()
        self.register_parameter('variance', torch.nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        return torch.exp(self.variance * 10).clip(1e-6, 1e6)

    def get_variance(self) -> torch.Tensor:
        """return current variance value"""
        return self()


@dataclass
class V3dNeusFieldConfig(V3dBaseFieldConfig):
    _target: Type=field(default_factory=lambda: V3dNeusField)
    is_geometry_field: bool = False
    geometry_mlp_config: V3dMLPConfig = field(default_factory=lambda: V3dMLPConfig(
        num_layers=3,
        layer_width=64,
        activation=torch.nn.Softplus(beta=100),
        geometric_init=None,
        weight_norm=True,
        implementation='torch'))

    variance_init: float = 0.1
    progressive_training: bool = False
    progressive_training_warmup: int = 5000
    steps_per_level: int = 5000
    level_init: int = 8


class V3dNeusField(V3dBaseField):
    geometry_type: Literal['density', 'sdf'] = 'sdf'
    config: V3dNeusFieldConfig

    def populate_geometry_modules(self) -> None:
        super().populate_geometry_modules()
        self.deviation_network = SingleVarianceLayer(self.config.variance_init)
        self.cos_anneal_ratio = 1.0
        if self.config.progressive_training:
            self.hash_encoding_mask = torch.nn.Parameter(torch.ones(
                self.config.position_encoder_config.n_levels * self.config.position_encoder_config.n_features_per_level,
                dtype=torch.float32,
            ), requires_grad=False)

    def set_step(self, step: int):
        super().set_step(step)
        if self.config.progressive_training:
            self.update_encoding_mask(self.step)

    def update_encoding_mask(self, step: int):
        level = max(0, (step - self.config.progressive_training_warmup)) // self.config.steps_per_level + self.config.level_init
        self.progressive_level = min(self.config.position_encoder_config.n_levels, level)
        self.hash_encoding_mask.data.fill_(1.0)
        self.hash_encoding_mask.data[level * self.config.position_encoder_config.n_features_per_level:] = 0.0

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the sdf network."""
        self.cos_anneal_ratio = anneal

    def populate_color_modules(self):
        # space feature + point + gradient
        self.config.color_mlp_config.in_dim = self.config.n_space_feature_dim + 3 + 3
        if self.config.use_direction:
            self.direciton_encoder = self.config.direction_encoder_config.get_encoder()
            self.config.color_mlp_config.in_dim += self.direciton_encoder.n_output_dims

        if self.config.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.config.n_appearance_embedding_dim)
            self.config.color_mlp_config.in_dim += self.embedding_appearance.get_out_dim()
        
        self.color_mlp = self.config.color_mlp_config.get_mlp()

    def get_geometry(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = points.shape[:-1]
        points = self.normalize_points(points.view(-1, 3))
        if self.config.geometry_mlp_config.geometric_init != None and self.config.geometry_mlp_config.geometric_init.init_type == 'sphere':
            points = points*2-1

        feature = self.position_encoder(points)
        feature = feature.to(points)

        if self.config.progressive_training:
            feature = feature * self.hash_encoding_mask

        feature = torch.cat([points, feature], dim=-1)
        feat = self.geometry_mlp(feature).view(*shape, -1)
        sdf = feat[..., 0:1]
        space_feat = feat[..., 1:]
        return sdf, space_feat

    def get_color(self, camera_indices: torch.Tensor, points: torch.Tensor, view_direction: torch.Tensor, space_feat: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        shape = view_direction.shape[:-1]
        features = [space_feat]
        if self.config.use_direction:
            view_direction = shift_directions_for_tcnn(view_direction)
            direction_feat = self.direciton_encoder(view_direction.view(-1, 3)).view(*shape, -1)
            features.append(direction_feat)

        if self.config.use_appearance_embedding:
            camera_indices = camera_indices.squeeze()
            # appearance
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.config.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*shape, self.config.n_appearance_embedding_dim), device=view_direction.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*shape, self.config.n_appearance_embedding_dim), device=view_direction.device
                    )
            features.append(embedded_appearance)



        features.extend([gradients, self.normalize_points(points)])
        features = torch.cat(features, dim=-1)
        color = self.color_mlp(features.view(-1, features.shape[-1])).view(*shape, -1)
        color = torch.sigmoid(color)
        return color

    def get_outputs(
        self, ray_samples: RaySamples
    ) -> Dict[FieldHeadNames, torch.Tensor]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """
        points = ray_samples.frustums.get_positions()
        with torch.enable_grad():
            points_with_grad = torch.nn.Parameter(points)
            sdf, space_feat = self.get_geometry(points_with_grad)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf, inputs=points_with_grad, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]

        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        rgb = self.get_color(ray_samples.camera_indices, points, ray_samples.frustums.directions, space_feat, gradients)
        outputs = {}
        outputs[FieldHeadNames.SDF] = sdf
        outputs[FieldHeadNames.RGB] = rgb
        outputs[FieldHeadNames.GRADIENT] = gradients
        outputs[FieldHeadNames.NORMALS] = normals
        if self.config.with_semantic:
            outputs[FieldHeadNames.SEMANTICS] = self.get_semantic(space_feat)
        return outputs


    def forward(self, ray_samples: RaySamples, return_alphas: bool = False) -> Dict[FieldHeadNames, torch.Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_samples)
        if return_alphas:
            alpha = self.get_alpha(ray_samples, field_outputs[FieldHeadNames.SDF], field_outputs[FieldHeadNames.GRADIENT])
            field_outputs[FieldHeadNames.ALPHA] = alpha
            if alpha.ndim == 3:
                weight, trans = self.get_weights_and_transmittance_from_alphas(alpha)
                field_outputs[FieldHeadNames.WEIGHT] = weight
                field_outputs[FieldHeadNames.TRANSMITTANCE] = trans
        return field_outputs

    def get_alpha(self, ray_samples: RaySamples, sdf: Optional[torch.Tensor]=None, gradients: Optional[torch.Tensor]=None) -> torch.Tensor:

        if sdf is None or gradients is None:
            sdf, gradients, _ = self.get_geometry_and_gradients(ray_samples.frustums.get_positions())

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self.cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha


    def get_alpha_for_sampler(self,
        ray_samples: RaySamples, sdf: Optional[torch.Tensor]=None, inv_s: int=0
    ) -> torch.Tensor:
        """
        rendering given a fixed inv_s as NeuS

        Args:
            ray_samples: samples along ray
            sdf: sdf values along ray
            inv_s: fixed variance value
        Returns:
            alpha value
        """

        if sdf is None:
            sdf = self.get_geometry(ray_samples.frustums.get_positions())[0]
        if inv_s == 0:
            inv_s = self.deviation_network()

        if sdf.ndim == 2:
            # Shape of sdf is [N, 1], so there is no continuous sampling information.
            sdf = sdf.squeeze(-1)
            deltas = ray_samples.deltas.squeeze(-1).to(sdf)
            # single point alpha
            estimated_next_sdf = sdf - deltas * 0.5
            estimated_prev_sdf = sdf + deltas * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            
            p = prev_cdf - next_cdf
            c = prev_cdf
            
            alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
            return alpha.unsqueeze(-1)     # [n_sample, 1]



        sdf = sdf.squeeze(-1)
        batch_size = ray_samples.shape[0]
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        assert ray_samples.deltas is not None
        deltas = ray_samples.deltas[:, :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (deltas + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = deltas
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

        return alpha.unsqueeze(-1)


    def get_metrics_dict(self) -> Dict:
        metrics_dict = {}
        metrics_dict['s_val'] = self.deviation_network()
        if self.config.progressive_training:
            metrics_dict['progressive_level'] = self.progressive_level

        # debug
        # if self.step % 500 == 0:
        #     out_path = Path(f'debug/neus_field/snapshot_{self.step:08d}.jpg')
        #     if self.step == 0:
        #         shutil.rmtree(out_path.parent)
        #     self.save_field_snapshot(out_path, verbose=True)
        return metrics_dict
