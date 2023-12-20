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

"""Space distortions."""

import abc
from typing import List, Optional, Union

import torch
from functorch import jacrev, vmap
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.utils.math import Gaussians
from nerfstudio.field_components.spatial_distortions import SpatialDistortion


class NonuniformSceneContraction(SpatialDistortion):
    """The original SceneContraction transforms the space into a cube with a length of 4, where the foreground is in the [-1,1]^3 space, which only accounts for 1/8 of the total volume. The "bound" parameter is added to adjust the proportion of the background area, improving the network's ability to fit the foreground.

    Args:
        order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.
    """

    def __init__(self, order: Optional[Union[float, int]] = None, bound: float = 2.0) -> None:
        super().__init__()
        self.order = order
        assert bound > 1.0, "Scene bound must be larger than 1."
        self.bound = bound

    def rescale_geometry(self, x, data: List[torch.Tensor]) -> List[torch.Tensor]:
        mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
        scale = torch.where(mag < 1.0, 1.0, ((1.0 - (1.0 / mag)) * (self.bound - 1.0) + 1.0) / mag)
        rescaled_data = []
        for item in data:
            rescaled_data.append(item * scale)
        return rescaled_data

    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1.0, x, ((1.0 - (1.0 / mag)) * (self.bound - 1.0) + 1.0) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x):
                return (
                    (1.0 - 1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)) * (self.bound - 1.0) + 1.0
                ) * (x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True))

            jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)
