from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

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
        return torch.exp(self.variance*10).clip(1e-6, 1e6)

    def get_variance(self) -> torch.Tensor:
        """return current variance value"""
        return self()


@dataclass
class V3dNeuralangeloFieldConfig(V3dBaseFieldConfig):
    _target: Type=field(default_factory=lambda: V3dNeuralangeloField)
    is_geometry_field: bool = False
    geometry_mlp_config: V3dMLPConfig = field(default_factory=lambda: V3dMLPConfig(
        num_layers=3,
        layer_width=64,
        activation=torch.nn.Softplus(beta=100),
        geometric_init=None,
        weight_norm=True,
        implementation='torch'))

    variance_init: float = 0.1
    numerical_gradients_delta_init: float =0.0001
    use_sdf_numerical_gradient: bool = True
    level_init: int = 6
    steps_per_level: int = 5000
    progressive_training: bool = True
    progressive_training_warmup: int = 5000

class V3dNeuralangeloField(V3dBaseField):
    geometry_type: Literal['density', 'sdf'] = 'sdf'
    config: V3dNeuralangeloFieldConfig

    def populate_geometry_modules(self) -> None:
        super().populate_geometry_modules()
        self.deviation_network = SingleVarianceLayer(self.config.variance_init)
        self.cos_anneal_ratio = 1.0
        self.hash_encoding_mask = torch.nn.Parameter(torch.ones(
            self.config.position_encoder_config.n_levels * self.config.position_encoder_config.n_features_per_level,
            dtype=torch.float32,
        ), requires_grad=False)
        self.numerical_gradients_delta = self.config.numerical_gradients_delta_init

    def set_step(self, step):
        super().set_step(step)
        self.update_encoding_mask(step)
        self.update_numerical_gradients_delta(step)

    def update_encoding_mask(self, step: int):
        level = max(0, (step - self.config.progressive_training_warmup)) // self.config.steps_per_level + self.config.level_init
        self.progressive_level = min(self.config.position_encoder_config.n_levels, level)
        self.hash_encoding_mask.data.fill_(1.0)
        self.hash_encoding_mask.data[level * self.config.position_encoder_config.n_features_per_level:] = 0.0


    def update_numerical_gradients_delta(self, step: int) -> None:

        encoding_config = self.config.position_encoder_config

        base_res = encoding_config.base_resolution
        max_res = encoding_config.desired_resolution
        growth_factor = encoding_config.per_level_scale
        delta = 1. / (base_res * growth_factor ** (self.progressive_level - 1))
        delta = max(1. / max_res, delta)

        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta


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
        points = self.normalize_points(points)
        feature = self.position_encoder(points.view(-1, 3)).view(*shape, -1)
        if self.config.progressive_training:
            feature = self.hash_encoding_mask * feature
        feature = feature.to(points)
        feature = torch.cat([points*2-1, feature], dim=-1)
        feat = self.geometry_mlp(feature)
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


    def get_sdf_gradient_feature(self, x):
        sdf, feature = self.get_geometry(x)

        scale = (self.aabb[1] - self.aabb[0]).max().item()
        delta = self.numerical_gradients_delta * scale
        if self.config.use_spatial_distortion:
            delta *= 4
        xyz = x
        N = xyz.shape[-1]
        offsets = delta * torch.stack([torch.eye(N, dtype=xyz.dtype, device=xyz.device), -torch.eye(N, dtype=xyz.dtype, device=xyz.device)], dim=1) # [3, 2, 3]
        
        neighbor_points = xyz.view(xyz.shape[:-1]+(1, 1, N)) + offsets
        nsdf, _ = self.get_geometry(neighbor_points)
        # [B, N, 3, 2]
        nsdf = nsdf.squeeze(-1)
        # [B, N, 3]
        gradients = (nsdf[..., 0] - nsdf[..., 1]) / (2*delta)

        # laplacian
        curv = (nsdf.sum(dim=-1) - 2 * sdf) / delta**2
        if self.config.use_spatial_distortion:
            # spatial distortion changes the geometry gradients.
            gradients, curv = self.spatial_distortion.rescale_geometry(x, [gradients, curv])

        return sdf, gradients, feature, curv

    def get_outputs(
        self, ray_samples: RaySamples
    ) -> Dict[FieldHeadNames, torch.Tensor]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """
        points = ray_samples.frustums.get_positions()

        if self.config.use_sdf_numerical_gradient:
            sdf, gradients, space_feat, curv = self.get_sdf_gradient_feature(points)
        else:
            with torch.enable_grad():
                points_with_grad = torch.nn.Parameter(points)
                sdf, space_feat = self.get_geometry(points_with_grad)
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf, inputs=points_with_grad, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
                )[0]
            curv = None

        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        rgb = self.get_color(ray_samples.camera_indices, points, ray_samples.frustums.directions, space_feat, gradients)
        outputs = {}
        outputs[FieldHeadNames.SDF] = sdf
        outputs[FieldHeadNames.RGB] = rgb
        outputs[FieldHeadNames.GRADIENT] = gradients
        outputs[FieldHeadNames.NORMALS] = normals
        if curv is not None:
            outputs[FieldHeadNames.CURVATURE] = curv

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
            weight, trans = self.get_weights_and_transmittance_from_alphas(alpha)
            field_outputs[FieldHeadNames.ALPHA] = alpha
            field_outputs[FieldHeadNames.WEIGHT] = weight
            field_outputs[FieldHeadNames.TRANSMITTANCE] = trans
        return field_outputs

    def get_alpha(self, ray_samples: RaySamples, sdf: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
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


    def get_metrics_dict(self) -> Dict:
        metrics_dict = {}
        metrics_dict['s_val'] = self.deviation_network()
        metrics_dict['numerical_gradients_delta'] = self.numerical_gradients_delta
        metrics_dict['progressive_level'] = self.progressive_level
        return metrics_dict
