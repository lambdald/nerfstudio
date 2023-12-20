

import dataclasses
from dataclasses import dataclass
from tracemalloc import start
from typing import Callable, List, Literal, Optional, Tuple, Type, Union
from typing_extensions import Self

import torch
from nerfacc import OccGridEstimator

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.model_components.ray_samplers import (PDFSampler, Sampler,
                                                      UniformSampler)
from nerfstudio.v3d.core.field_components.encodings import TcnnEncoderConfig
from nerfstudio.v3d.core.field_components.v3d_mlp import V3dMLPConfig
from nerfstudio.v3d.core.fields.v3d_base_field import V3dBaseField
from nerfstudio.v3d.core.fields.v3d_nerf_field import V3dNerfFieldConfig
from nerfstudio.v3d.core.fields.v3d_neus_field import V3dNeusField


@dataclass
class RaySamplerConfig(InstantiateConfig):
    pass


@dataclass
class ProposalSamplerConfig(RaySamplerConfig):
    proposal_nets: List[V3dNerfFieldConfig] = dataclasses.field(default_factory=lambda: [
        V3dNerfFieldConfig(
            is_geometry_field=True,
            position_encoder_config=TcnnEncoderConfig(
                log2_hashmap_size=17, n_levels=5, desired_resolution=128),
            geometry_mlp_config=V3dMLPConfig(num_layers=2, layer_width=16, implementation='tcnn')),
        V3dNerfFieldConfig(
            is_geometry_field=True,
            position_encoder_config=TcnnEncoderConfig(
                log2_hashmap_size=17, n_levels=5, desired_resolution=256),
            geometry_mlp_config=V3dMLPConfig(num_layers=2, layer_width=16, implementation='tcnn')),])

    proposal_initial_sampler: Literal["piecewise", "uniform"] = "uniform"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""

    use_single_jitter: bool = True
    num_proposal_samples_per_ray: Tuple[int,...] = dataclasses.field(default_factory=lambda: (256, 512))
    num_nerf_samples_per_ray: int = 48

    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000

    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    def __post_init__(self):
        self.num_proposal_nets = len(self.proposal_nets)


@dataclass
class NeuSHierarchicalSamplerConfig(RaySamplerConfig):
    _target: Type = dataclasses.field(default_factory=lambda: NeuSHierarchicalSampler)
    num_coarse_samples: int = 128
    num_fine_samples: int = 64
    num_prior_samples: int = 32
    num_fine_steps: int = 4
    num_fine_samples_per_step: int = -1
    base_variance:int =64
    def __post_init__(self):
        self.num_fine_samples_per_step = self.num_fine_samples // self.num_fine_steps
    

class NeuSHierarchicalSampler(Sampler):
    config: NeuSHierarchicalSamplerConfig
    def __init__(self, config: NeuSHierarchicalSamplerConfig) -> None:
        super().__init__()
        self.config = config
        self.init_sampler = UniformSampler(self.config.num_coarse_samples)
        self.pdf_sampler = PDFSampler(self.config.num_fine_samples_per_step, include_original=False)

    @torch.no_grad()
    def generate_ray_samples(
        self,
        field: V3dNeusField,
        ray_bundle: RayBundle
    ) -> RaySamples:

        last_samples = self.init_sampler.generate_ray_samples(ray_bundle)
        last_sdf: Optional[torch.Tensor] = field.get_geometry(last_samples.frustums.get_positions())[0]

        for idx_sample in range(self.config.num_fine_steps):
            alphas = field.get_alpha_for_sampler(last_samples, last_sdf, inv_s=self.config.base_variance* 2**idx_sample)
            weights = field.get_weights_and_transmittance_from_alphas(alphas, weights_only=True)
            weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)

            new_samples = self.pdf_sampler(ray_bundle, last_samples, weights, self.config.num_fine_samples_per_step)
            new_sdf = field.get_geometry(new_samples.frustums.get_positions())[0]
            last_samples, sorted_index = self.merge_ray_samples(ray_bundle, last_samples, new_samples)
            print('alpha=', alphas.shape)
            print('weight=',weights.shape)
            print('new_sdf=',new_sdf.shape)


            if idx_sample != self.config.num_fine_steps -1:
                sdf_merge = torch.cat([last_sdf.squeeze(-1), new_sdf.squeeze(-1)], -1)
                print(sdf_merge.shape, sorted_index.shape, sorted_index.min(), sorted_index.max())
                last_sdf = torch.gather(sdf_merge, dim=1, index=sorted_index.long()).unsqueeze(-1)
        return last_samples


    @staticmethod
    def merge_ray_samples(ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values
        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        assert ray_samples_1.spacing_starts is not None and ray_samples_2.spacing_starts is not None
        assert ray_samples_1.spacing_ends is not None and ray_samples_2.spacing_ends is not None
        assert ray_samples_1.spacing_to_euclidean_fn is not None
        starts_1 = ray_samples_1.spacing_starts[..., 0]
        starts_2 = ray_samples_2.spacing_starts[..., 0]

        ends = torch.maximum(ray_samples_1.spacing_ends[..., -1:, 0], ray_samples_2.spacing_ends[..., -1:, 0])

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index



@dataclass
class NeRFHierarchicalSamplerConfig(RaySamplerConfig):
    _target: Type = dataclasses.field(default_factory=lambda: NeRFHierarchicalSampler)
    num_coarse_samples: int = 256
    num_fine_samples: int = 256
    num_prior_samples: int = 32
    num_fine_steps: int = 4

    num_fine_samples_per_step: int = -1

    def __post_init__(self):
        self.num_fine_samples_per_step = self.num_fine_samples // self.num_fine_steps
    

class NeRFHierarchicalSampler(Sampler):
    config: NeRFHierarchicalSamplerConfig
    def __init__(self, config: NeRFHierarchicalSamplerConfig) -> None:
        super().__init__()
        self.config = config
        self.init_sampler = UniformSampler(self.config.num_coarse_samples)
        self.pdf_sampler = PDFSampler(self.config.num_fine_samples_per_step, include_original=False)

    @torch.no_grad()
    def generate_ray_samples(
        self,
        field: V3dBaseField,
        ray_bundle: RayBundle
    ) -> RaySamples:

        init_ray_samples = self.init_sampler.generate_ray_samples(ray_bundle)
        sample_list = [init_ray_samples]
        weight_list = []

        for idx_sample in range(self.config.num_fine_steps):
            last_sample = sample_list[-1]
            alphas = field.get_alpha(last_sample)
            weight = field.get_weights_and_transmittance_from_alphas(alphas, weights_only=True)
            new_samples = self.pdf_sampler.generate_ray_samples(ray_bundle, last_sample, weight, self.config.num_fine_samples_per_step)
            sample_list.append(new_samples)
            weight_list.append(weight)

        #TODO sample based depth proir.

        all_samples = sample_list[0]
        for sample_to_merge in sample_list[1:]:
            all_samples, _ = self.merge_ray_samples(ray_bundle, all_samples, sample_to_merge)
        return all_samples


    @staticmethod
    def merge_ray_samples(ray_bundle: RayBundle, ray_samples_1: RaySamples, ray_samples_2: RaySamples):
        """Merge two set of ray samples and return sorted index which can be used to merge sdf values
        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        assert ray_samples_1.spacing_starts is not None and ray_samples_2.spacing_starts is not None
        assert ray_samples_1.spacing_ends is not None and ray_samples_2.spacing_ends is not None
        assert ray_samples_1.spacing_to_euclidean_fn is not None
        starts_1 = ray_samples_1.spacing_starts[..., 0]
        starts_2 = ray_samples_2.spacing_starts[..., 0]

        ends = torch.maximum(ray_samples_1.spacing_ends[..., -1:, 0], ray_samples_2.spacing_ends[..., -1:, 0])

        bins, sorted_index = torch.sort(torch.cat([starts_1, starts_2], -1), -1)

        bins = torch.cat([bins, ends], dim=-1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = ray_samples_1.spacing_to_euclidean_fn(bins)

        ray_samples = ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples_1.spacing_to_euclidean_fn,
        )

        return ray_samples, sorted_index


@dataclass
class VolumetricSamplerConfig(RaySamplerConfig):
    _target: Type = dataclasses.field(default_factory=lambda: VolumetricSampler)

    occ_type: Literal['density', 'alpha'] = 'density'
    occ_resolution: Union[int, List[int], torch.Tensor] = 128
    occ_level: int = 1

    alpha_thre: float = 0.0
    cone_angle: float = 0.0

    min_ray_marching_step: float = 0.1
    max_ray_marching_step: float = 2

    near_plane: float = 0.2
    far_plane: float = 250

    warmup_steps: int = 2000
    warmup_update_every_n_step: int=1
    update_every_n_step:int =16

    early_stop_eps: float = 0.1


class VolumetricSampler(Sampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the occ_fn is provided.

    Args:
    occ_fn: Function that evaluates occ at a given point. (density or alpha function)
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    config: VolumetricSamplerConfig
    def __init__(
        self,
        config: VolumetricSamplerConfig,
        aabb: torch.Tensor,
        occ_fn: Callable,
    ):
        super().__init__()
        self.occupancy_grid = OccGridEstimator(aabb.view(-1).tolist(), config.occ_resolution, config.occ_level)
        self.occ_fn = occ_fn
        self.render_step_size = config.max_ray_marching_step
        self.config = config

    def get_occ_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:
            origins: Origins of rays
            directions: Directions of rays
            times: Times at which rays are sampled
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.occ_fn is None or not self.training:
            return None

        occ_fn = self.occ_fn

        def get_point_occ(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            ray_steps = t_ends - t_starts
            assert t_starts.shape[0]
            ray_samples = RaySamples(
                frustums=Frustums(
                    origins=t_origins,
                    directions=t_dirs,
                    starts=t_starts[..., None],
                    ends=t_ends[..., None],
                    pixel_area=torch.full(t_origins.shape[:-1]+(1,), 0.0, device=t_origins.device)
,
                ),
                deltas=ray_steps[..., None],
            )
            return occ_fn(ray_samples).squeeze(-1)
        return get_point_occ


    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )


    def set_step(self, step: int) -> None:
        self.step = step



    def set_render_step_size(self, render_step_size: float):
        self.render_step_size = min(max(render_step_size, self.config.min_ray_marching_step), self.config.max_ray_marching_step)
        print('render step size:', self.render_step_size)


    def update_occupancy_grid(self, step):
        if step < self.config.warmup_steps:
            every_n_step = self.config.warmup_update_every_n_step
        else:
            every_n_step = self.config.update_every_n_step

        if self.training:
            # print('update...[train mode]', 'step=', step, ' num voxel=',torch.prod(self.occupancy_grid.resolution))
            # n0 = torch.sum(self.occupancy_grid.binaries)

            def point_occ(positions):

                shape = positions.shape[:-1] + (1,)
                device = positions.device
                ray_samples = RaySamples(
                    frustums=Frustums(
                        origins=positions,
                        directions=torch.full(shape, 1.0, device=device),
                        starts=torch.full(shape, 0.0, device=device),
                        ends=torch.full(shape, 0.0, device=device),
                        pixel_area=torch.full(shape, 1.0, device=device),
                    ),
                    deltas=torch.full(shape, self.render_step_size, device=device),
                )
                return self.occ_fn(ray_samples)


            self.occupancy_grid.update_every_n_steps(step=step,
                                                    occ_eval_fn=point_occ,
                                                    n=every_n_step,
                                                    warmup_steps=self.config.warmup_steps,
                                                    occ_thre=1e-3)
            # n1 = torch.sum(self.occupancy_grid.binaries)
            # print('occupancy gird num=',f'{n0}->{n1}', f'diff={n1-n0}')



    def forward(
        self,
        ray_bundle: RayBundle,
    ) -> Tuple[RaySamples, torch.Tensor]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None


        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None

        if self.config.occ_type == 'density':
            sigma_fn = self.get_occ_fn(rays_o, rays_d)
            alpha_fn = None
        elif self.config.occ_type == 'alpha':
            sigma_fn = None
            alpha_fn = self.get_occ_fn(rays_o, rays_d)
        else:
            raise NotImplementedError(f'Unknown occupancy grid type: {self.config.occ_type}')


        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=sigma_fn,
            alpha_fn=alpha_fn,
            render_step_size=self.render_step_size,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
            stratified=self.training,
            cone_angle=self.config.cone_angle,
            alpha_thre=0.0,
            early_stop_eps=self.config.early_stop_eps
        )

        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
            deltas=(ends-starts)[..., None]
        )

        #TODO: add importance sampling, refer to neuralsim.
        # if ray_bundle.times is not None:
        #     ray_samples.times = ray_bundle.times[ray_indices]
        return ray_samples, ray_indices
