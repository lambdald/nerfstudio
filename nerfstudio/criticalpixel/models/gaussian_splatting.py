# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsplat.sh import num_sh_bases
from pytorch_msssim import SSIM
from tensordict import tensorclass
from torch.nn import Parameter

# metrics
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.criticalpixel.appearance.sphere_harmonics import RGB2SH, SH2RGB
from nerfstudio.criticalpixel.camera.posed_camera import PosedCamera
from nerfstudio.criticalpixel.geometry.bbox import AxisAlignedBoundingBox
from nerfstudio.criticalpixel.geometry.gaussian import Gaussian3D
from nerfstudio.criticalpixel.geometry.point_cloud import PointCloud
from nerfstudio.criticalpixel.geometry.transform import CoordinateType
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.rich_utils import CONSOLE


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 3000
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    percent_dense: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000

    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""

    random_init_num_points: int = 500000

    ssim_lambda: float = 0.02
    """weight of ssim loss"""
    stop_split_at: int = 20000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=CameraOptimizerConfig)
    """camera optimizer config"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

    near: float = 1
    far: float = 1000


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


@tensorclass
class GaussianPointCloud(Gaussian3D):
    #! Geometry
    opacity: torch.nn.Parameter

    #! Appearance
    feature_dc: torch.nn.Parameter
    feature_rest: torch.nn.Parameter


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    # q = torch.nn.functional.normalize(r, p=2, dim=-1)

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    scene_box: AxisAlignedBoundingBox
    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "scene_metadata" in kwargs:
            self.seed_points3d: Optional[PointCloud] = kwargs["scene_metadata"].points3d
        else:
            self.seed_points3d = None
        super().__init__(*args, **kwargs)

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def num_points(self):
        return self.mean.shape[0]

    def populate_modules(self):
        self.setup_functions()

        #! init gaussian point cloud.
        if self.seed_points3d is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points3d.points)
        else:
            rand_norm_pts = torch.rand((self.config.random_init_num_points, 3)) * 2 - 1
            rand_pts = self.scene_box.get_denormalized_points(rand_norm_pts).float()
            means = torch.nn.Parameter(rand_pts)

        num_points = means.shape[0]

        dim_sh = num_sh_bases(self.config.sh_degree)
        if self.seed_points3d is not None and not self.config.random_init and self.seed_points3d.colors is not None:
            seed_color = self.seed_points3d.colors.float() / 255
            shs = torch.zeros((num_points, dim_sh, 3), dtype=torch.float32)
            if self.config.sh_degree > 0:
                shs[:, 0] = RGB2SH(seed_color.float())
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0] = torch.logit(seed_color, eps=1e-10)

            features_dc = torch.nn.Parameter(shs[:, 0:1])
            features_rest = torch.nn.Parameter(shs[:, 1:])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 1, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        # quats = torch.nn.Parameter(random_quat_tensor(num_points))
        quats = torch.zeros((num_points, 4))
        quats[:, 0] = 1.0
        quats = torch.nn.Parameter(quats)
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        #! gaussian model
        self.mean = means
        self.scale = scales
        self.quaternion = quats
        self.opacity = opacities
        self.feature_dc = features_dc
        self.feature_rest = features_rest

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.seed_points3d = None
        self.active_sh_degree = 0

        # for densify
        self.xyz_gradient_accum: Optional[torch.Tensor] = None
        self.max_radii2D: Optional[torch.Tensor] = None
        self.denom: Optional[torch.Tensor] = None

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.feature_dc)
        else:
            return torch.sigmoid(self.feature_dc)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scale)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.quaternion)

    @property
    def get_xyz(self):
        return self.mean

    @property
    def get_features(self):
        features_dc = self.feature_dc
        features_rest = self.feature_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def cat_tensors_to_optimizer(self, tensors_dict: Dict) -> Dict:
        optimizable_tensors = {}
        for param_name, new_param in tensors_dict.items():
            optimizer = self.optimizer.optimizers[param_name]
            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1

            old_param = group["params"][0]
            stored_state = optimizer.state[old_param]

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(new_param)), dim=0
                ).contiguous()
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(new_param)), dim=0
                ).contiguous()

                del optimizer.state[old_param]
                group["params"][0] = torch.nn.Parameter(
                    torch.cat((group["params"][0], new_param), dim=0).contiguous().requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[param_name] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(
                    torch.cat((group["params"][0], new_param), dim=0).contiguous().requires_grad_(True)
                )
                optimizable_tensors[param_name] = group["params"][0]

        return optimizable_tensors

    def save_training_stats(self, optimizers: Optimizers):
        self.optimizer = optimizers
        if self.xyz_gradient_accum is None:
            self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
            self.denom = torch.zeros((self.num_points, 1), device=self.device)
            self.max_radii2D = torch.zeros((self.num_points), device=self.device)

        assert "visibility_filter" in self.render_result
        visibility_filter = self.render_result["visibility_filter"]
        radii = self.render_result["radii"]
        viewspace_points = self.render_result["viewspace_points"]
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])

        self.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True
        )

        print("\n\n\nMax of xyz_gradient_accum=", self.xyz_gradient_accum.max())

        self.denom[visibility_filter] += 1
        print("Max of denom=", self.denom.max())

    @torch.no_grad()
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
    ):
        d = {
            "xyz": new_xyz,
            "feature_dc": new_features_dc,
            "feature_rest": new_features_rest,
            "opacity": new_opacities,
            "scale": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.mean = optimizable_tensors["xyz"]
        self.feature_dc = optimizable_tensors["feature_dc"]
        self.feature_rest = optimizable_tensors["feature_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scale = optimizable_tensors["scale"]
        self.quaternion = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.device)
        self.denom = torch.zeros((self.num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((self.num_points), device=self.device)

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for param_name, optimizer in self.optimizer.optimizers.items():
            group = optimizer.param_groups[0]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask].contiguous()
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask].contiguous()

                del optimizer.state[group["params"][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][mask].contiguous().requires_grad_(True)))
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[param_name] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][mask].contiguous().requires_grad_(True))
                optimizable_tensors[param_name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self.mean = optimizable_tensors["xyz"]
        self.feature_dc = optimizable_tensors["feature_dc"]
        self.feature_rest = optimizable_tensors["feature_rest"]
        self.opacity = optimizable_tensors["opacity"]
        self.scale = optimizable_tensors["scale"]
        self.quaternion = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @torch.no_grad()
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.config.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.quaternion[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self.quaternion[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.feature_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.feature_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: torch.Tensor, scene_extent: float):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.config.percent_dense * scene_extent
        )

        new_xyz = self.mean[selected_pts_mask]
        new_features_dc = self.feature_dc[selected_pts_mask]
        new_features_rest = self.feature_rest[selected_pts_mask]
        new_opacities = self.opacity[selected_pts_mask]
        new_scaling = self.scale[selected_pts_mask]
        new_rotation = self.quaternion[selected_pts_mask]
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
        )

    @torch.no_grad()
    def reset_opacity(self):
        self.opacity.data.fill_(min(self.opacity.min().item(), 0.01))

    def set_step(self, optimizers: Optimizers, step: int):
        self.step = step

        if (
            self.step > 0
            and self.step % self.config.sh_degree_interval == 0
            and self.active_sh_degree < self.config.sh_degree
        ):
            self.active_sh_degree += 1

        if step < self.config.stop_split_at:
            self.save_training_stats(optimizers)

            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            scene_scale = self.scene_box.get_lenghts().max().item()

            if self.step > self.config.warmup_length and self.step % self.config.refine_every == 0:
                size_threshold = 20 if step > self.config.reset_alpha_every else None
                self.densify_and_prune(self.config.densify_grad_thresh, 0.005, scene_scale, size_threshold)

            if self.step % self.config.reset_alpha_every == 0:
                self.reset_opacity()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.set_step,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.mean],
            "feature_dc": [self.feature_dc],
            "feature_rest": [self.feature_rest],
            "opacity": [self.opacity],
            "scale": [self.scale],
            "rotation": [self.quaternion],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def render_image(self, posed_camera: PosedCamera, background_color: torch.Tensor):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.mean, requires_grad=True, device=self.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # camera intrinsic
        height, width = posed_camera.cameras.hws.view(-1).tolist()
        fx, fy = posed_camera.cameras.params.view(-1)[:2].tolist()
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        # camera pose
        pose_c2w = posed_camera.pose_c2w.squeeze().clone()
        if posed_camera.coord_type == CoordinateType.OpenGL:
            pose_c2w[:3, 1:3] *= -1
        elif posed_camera.coord_type == CoordinateType.OpenCV:
            pass
        else:
            raise NotImplementedError(f"Unknown coordiante type: {posed_camera.coord_type}")

        pose_w2c = torch.inverse(pose_c2w)

        R = pose_w2c[:3, :3].T.cpu().numpy()
        T = pose_w2c[:3, 3].cpu().numpy()

        world_view_transform = torch.tensor(getWorld2View2(R, T, np.zeros((3,)), 1.0)).transpose(0, 1).cuda()
        projection_matrix = (
            getProjectionMatrix(znear=self.config.near, zfar=self.config.far, fovX=FovX, fovY=FovY)
            .transpose(0, 1)
            .cuda()
        )
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        gsplat_config = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background_color.to(self.device),
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=gsplat_config)

        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        cov3D_precomp = None
        scales = self.get_scaling
        rotations = self.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        shs = self.get_features
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": torch.clip(rendered_image, 0.0, 1.0),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_outputs(self, posed_camera: PosedCamera) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(posed_camera, PosedCamera):
            print("Called get_outputs with not a camera")
            return {}
        assert posed_camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(posed_camera)
        if self.training:
            background = torch.rand(3, device=self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE
            else:
                background = self.back_color.to(self.device)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.mean).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(posed_camera.cameras.hws.view(-1) + [1])}
        else:
            crop_ids = None

        render_pkg = self.render_image(posed_camera, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        if self.training:
            self.render_result = render_pkg
        else:
            self.render_result = {}

        return {"rgb": image}  # type: ignore

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]

        #     # torchvision can be slow to import, so we do it lazily.
        #     import torchvision.transforms.functional as TF

        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     gt_img = batch["image"]

        gt_img = batch["image"].squeeze(0).permute(2, 0, 1)

        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb.unsqueeze(0), gt_rgb.unsqueeze(0))

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points
        metrics_dict["sh_degree"] = self.active_sh_degree
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]

        #     # torchvision can be slow to import, so we do it lazily.
        #     import torchvision.transforms.functional as TF

        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     gt_img = batch["image"]

        gt_img = batch["image"].squeeze(0).permute(2, 0, 1).to(self.device)

        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.unsqueeze(0), outputs["rgb"].unsqueeze(0))

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        return {
            "rgb": (1 - self.config.ssim_lambda) * Ll1,
            "ssim": self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     # torchvision can be slow to import, so we do it lazily.
        #     import torchvision.transforms.functional as TF

        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        #     predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     gt_img = batch["image"]
        #     predicted_rgb = outputs["rgb"]

        # [H, W, C]
        gt_rgb = batch["image"].squeeze(0).to(self.device)
        predicted_rgb = outputs["rgb"].permute(1, 2, 0)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.feature_dc.shape[1] * self.feature_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self.feature_rest.shape[1] * self.feature_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.quaternion.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        from plyfile import PlyData, PlyElement

        xyz = self.mean.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.feature_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.feature_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.quaternion.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(str(path))
