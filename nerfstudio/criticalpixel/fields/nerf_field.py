from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Type

import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.criticalpixel.fields.base_field import FieldHeadNames, shift_directions_for_tcnn
from nerfstudio.criticalpixel.fields.base_field import BaseField, BaseFieldConfig
from nerfstudio.criticalpixel.field_components.mlp import MLP, MLPConfig
from nerfstudio.criticalpixel.field_components.encodings import TcnnGridEncoderConfig, TcnnSphereHarmonicsEncoderConfig


@dataclass
class NerfFieldConfig(BaseFieldConfig):
    _target: Type = field(default_factory=lambda: NerfField)
    is_geometry_field: bool = False
    """The geometry field does not have a color network."""
    use_spatial_distortion: bool = True
    """If there are objects outside the bbox in the image, this scene is usually considered an unbounded scene. Spatial distortion is needed to normalize the scene in an unbounded scene."""
    spatial_distortion_bound: float = 2
    # encoder
    position_encoder_config: TcnnGridEncoderConfig = field(
        default_factory=lambda: TcnnGridEncoderConfig(
            type="hashgrid",
            n_input_dims=3,
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=21,
            base_resolution=16,
            desired_resolution=7000,
            interpolation="Linear",
        )
    )

    use_direction: bool = True
    use_appearance_embedding: bool = True
    n_appearance_embedding_dim: int = 32
    use_average_appearance_embedding: bool = True

    n_space_feature_dim: int = 15
    direction_encoder_config: TcnnSphereHarmonicsEncoderConfig = field(
        default_factory=lambda: TcnnSphereHarmonicsEncoderConfig(degree=4)
    )
    geometry_mlp_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(num_layers=2, layer_width=64, implementation="tcnn")
    )
    color_mlp_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(num_layers=3, layer_width=64, out_dim=3, implementation="tcnn")
    )
    with_semantic: bool = False
    semantic_mlp_config: MLPConfig = field(
        default_factory=lambda: MLPConfig(
            num_layers=2,
            layer_width=64,
            out_dim=-1,
            activation=torch.nn.ReLU6(),
            implementation="torch",
        )
    )


class NerfField(BaseField):
    geometry_type: Literal["density", "sdf"] = "density"
    config: NerfFieldConfig

    def get_geometry(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = points.shape[:-1]
        points = self.normalize_points(points.view(-1, 3))
        selector = ((points > 0.0) & (points < 1.0)).all(dim=-1)
        points = points * selector[..., None]
        feature = self.position_encoder(points)
        feature = torch.cat([points, feature], dim=-1)
        feat = self.geometry_mlp(feature).view(*shape, -1)
        density = trunc_exp(feat[..., 0:1])
        density = density * selector[..., None].view(*shape, -1)
        space_feat = feat[..., 1:]
        return density, space_feat

    def get_color(
        self, camera_indices: torch.Tensor, view_direction: torch.Tensor, space_feat: torch.Tensor
    ) -> torch.Tensor:
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

        features = torch.cat(features, dim=-1)
        features = features.view(-1, features.shape[-1])
        color = self.color_mlp(features).view(*shape, -1)
        color = torch.sigmoid(color)
        return color

    def get_outputs(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, torch.Tensor]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """
        points = ray_samples.frustums.get_positions()
        density, space_feat = self.get_geometry(points)
        rgb = self.get_color(ray_samples.camera_indices, ray_samples.frustums.directions, space_feat)
        outputs = {}
        outputs[FieldHeadNames.DENSITY] = density
        outputs[FieldHeadNames.RGB] = rgb
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
            alpha = self.get_alpha(ray_samples, field_outputs[FieldHeadNames.DENSITY])
            weight, trans = self.get_weights_and_transmittance_from_alphas(alpha)
            weight = weight.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            field_outputs[FieldHeadNames.ALPHA] = alpha
            field_outputs[FieldHeadNames.WEIGHT] = weight
            field_outputs[FieldHeadNames.TRANSMITTANCE] = trans
        return field_outputs

    def get_alpha(self, ray_samples: RaySamples, densities: Optional[torch.Tensor] = None) -> torch.Tensor:
        if densities is None:
            densities = self.get_geometry(ray_samples.frustums.get_positions())[0]
        delta_density = ray_samples.deltas * densities
        alphas = 1 - torch.exp(-delta_density)
        return alphas

    def get_density(self, positions: torch.Tensor) -> torch.Tensor:
        return self.get_geometry(positions)[0]
