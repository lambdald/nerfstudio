from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.criticalpixel.field_components.spatial_distortions import NonuniformSceneContraction
from nerfstudio.fields.base_field import Field, FieldConfig, FieldHeadNames
from nerfstudio.plugins.registry_dataparser import CONSOLE
from nerfstudio.criticalpixel.field_components.encodings import TcnnGridEncoderConfig, TcnnSphereHarmonicsEncoderConfig
from nerfstudio.criticalpixel.field_components.mlp import MLPConfig


def shift_directions_for_tcnn(directions: torch.Tensor) -> torch.Tensor:
    """Shift directions from [-1, 1] to [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_points(
    points: torch.Tensor, aabb: torch.Tensor, spatial_distortion: Optional[NonuniformSceneContraction] = None
):
    # normalize the position to [0, 1]
    points = SceneBox.get_normalized_positions(points, aabb)  # normalize the point in bbox to [0, 1]
    if spatial_distortion is not None:
        # unbounded scene
        points = spatial_distortion(points * 2 - 1)
        points = (points + spatial_distortion.bound) / (spatial_distortion.bound * 2)
    return points


@dataclass
class BaseFieldConfig(FieldConfig):
    _target: Type = field(default_factory=lambda: V3dBaseField)
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


class BaseField(Field):
    geometry_type: Literal["density", "sdf"]

    config: BaseFieldConfig
    aabb: torch.Tensor

    def __init__(self, config: BaseFieldConfig, box: SceneBox, num_images: int) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("aabb", box.aabb)
        self.num_images = num_images
        self.populate_modules()

    def populate_geometry_modules(self) -> None:
        if self.config.use_spatial_distortion:
            self.spatial_distortion = NonuniformSceneContraction(
                order=float("inf"), bound=self.config.spatial_distortion_bound
            )
        else:
            self.spatial_distortion = None
        self.position_encoder = self.config.position_encoder_config.get_encoder()
        # encoding + position
        self.config.geometry_mlp_config.in_dim = self.position_encoder.n_output_dims + 3

        self.config.geometry_mlp_config.out_dim = 1
        if not self.config.is_geometry_field:
            self.config.geometry_mlp_config.out_dim += self.config.n_space_feature_dim
        self.geometry_mlp = self.config.geometry_mlp_config.get_mlp()

    def populate_color_modules(self):
        self.config.color_mlp_config.in_dim = self.config.n_space_feature_dim
        if self.config.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.config.n_appearance_embedding_dim)
            self.config.color_mlp_config.in_dim += self.embedding_appearance.get_out_dim()
        if self.config.use_direction:
            self.direciton_encoder = self.config.direction_encoder_config.get_encoder()
            self.config.color_mlp_config.in_dim += self.direciton_encoder.n_output_dims
        self.config.color_mlp_config.out_dim = 3
        self.color_mlp = self.config.color_mlp_config.get_mlp()

    def populate_semantic_modules(self):
        self.config.semantic_mlp_config.in_dim = self.config.n_space_feature_dim
        self.semantic_mlp = self.config.semantic_mlp_config.get_mlp()

    def populate_modules(self):
        self.populate_geometry_modules()
        if not self.config.is_geometry_field:
            self.populate_color_modules()
            if self.config.with_semantic:
                self.populate_semantic_modules()
        print(self)

    def set_step(self, step: int):
        self.step = step

    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        return normalize_points(points, self.aabb, self.spatial_distortion)

    def get_geometry(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_geometry_and_gradients(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            points_with_grad = torch.nn.Parameter(points)
            geometry, space_feat = self.get_geometry(points_with_grad)
            d_output = torch.ones_like(geometry, requires_grad=False, device=geometry.device)
            gradients = torch.autograd.grad(
                outputs=geometry,
                inputs=points_with_grad,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        return geometry, gradients, space_feat

    def get_color(self) -> torch.Tensor:
        raise NotImplementedError

    def get_semantic(self, space_feat: torch.Tensor):
        assert self.config.with_semantic
        shape = space_feat.shape[:-1]
        n_dim = space_feat.shape[-1]
        if not self.training:
            space_feat = space_feat.float()

        semantic = self.semantic_mlp(space_feat.view(-1, n_dim)).view(*shape, -1)
        return semantic

    def get_outputs(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, torch.Tensor]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """
        raise NotImplementedError

    def forward(self, ray_samples: RaySamples, return_alphas: bool = False) -> Dict[FieldHeadNames, torch.Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_samples)
        return field_outputs

    def get_alpha(self, ray_samples: RaySamples, geomeray: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def get_alpha_for_sampler(self, **kwargs) -> torch.Tensor:
        return self.get_alpha(**kwargs)

    def get_weights_and_transmittance_from_alphas(
        self, alphas: torch.Tensor, weights_only: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return weights based on predicted alphas
        Args:
            alphas: Predicted alphas (maybe from sdf) for samples along ray
            weights_only: If function should return only weights
        Returns:
            Tuple of weights and transmittance for each sample
        """
        transmittance = torch.cumprod(
            torch.cat([torch.ones((*alphas.shape[:1], 1, 1), device=alphas.device), 1.0 - alphas + 1e-7], 1), 1
        )
        weights = alphas * transmittance[:, :-1, :]
        weights = torch.nan_to_num(weights)

        if weights_only:
            return weights
        return weights, transmittance

    @torch.no_grad()
    def extract_geometry_fields(
        self, bbox: Optional[torch.Tensor] = None, precision: float = 0.01, dim_batch_size: int = 512
    ):
        CONSOLE.print(f"Extract {self.geometry_type} geometry field")
        if bbox is None:
            bbox = self.aabb

        CONSOLE.print(f"Bbox: {bbox.tolist()}(Model bbox={self.aabb}")
        scene_range = torch.abs(bbox[1] - bbox[0])
        resolution = scene_range / precision
        CONSOLE.print(f"scene range={scene_range}, scene resolution={resolution}")
        resolution = torch.round(resolution).cpu().to(torch.int).tolist()
        X = torch.linspace(bbox[0, 0].item(), bbox[1, 0].item(), resolution[0]).split(dim_batch_size)
        Y = torch.linspace(bbox[0, 1].item(), bbox[1, 1].item(), resolution[1]).split(dim_batch_size)
        Z = torch.linspace(bbox[0, 2].item(), bbox[1, 2].item(), resolution[2]).split(dim_batch_size)

        field = torch.zeros(resolution, dtype=torch.float32, device="cpu")

        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    # [b, b, b]
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.stack([xx, yy, zz], dim=-1).cuda()  # [b, b, b, 3]
                    geometry = self.get_geometry(pts)[0].squeeze(-1)  # [b, b, b, 3] --> [b, b, b]
                    geometry = geometry.detach().cpu()
                    x_range = [xi * dim_batch_size, xi * dim_batch_size + len(xs)]
                    y_range = [yi * dim_batch_size, yi * dim_batch_size + len(ys)]
                    z_range = [zi * dim_batch_size, zi * dim_batch_size + len(zs)]
                    field[x_range[0] : x_range[1], y_range[0] : y_range[1], z_range[0] : z_range[1]] = geometry
        return field

    @torch.no_grad()
    def save_field_snapshot(self, out_path: Path, resolution: int = 512, verbose: bool = False):
        precision = (self.aabb[1] - self.aabb[0]).max().item() / resolution
        field = self.extract_geometry_fields(self.aabb, precision, dim_batch_size=128).cpu().numpy()
        x, y, z = field.shape
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
        # plot just the positive data and save the
        # color "mappable" object returned by ax1.imshow
        if verbose:
            field_pd = pd.DataFrame(field.reshape(-1))
            CONSOLE.print("Geometry snapshot:\n", field_pd.describe())

        yz = ax1.imshow(field[x // 2])
        ax1.set_title("YZ")
        fig.colorbar(yz, ax=ax1)

        neg = ax2.imshow(field[:, y // 2])
        fig.colorbar(neg, ax=ax2)
        ax2.set_title("XZ")
        pos_neg_clipped = ax3.imshow(field[..., z // 2])
        cbar = fig.colorbar(pos_neg_clipped)
        ax3.set_title("XY")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path))
        if verbose:
            CONSOLE.print("save geometry snapshot to ", out_path)
