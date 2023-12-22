"""
Code for sampling pixels.
"""

import random

import torch
from torch import Tensor

from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.criticalpixel.data.sampler.utils import erode_mask
from typing import (
    Dict,
    Optional,
    Type,
    Union,
)
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameItemType, Frame1DAttrTypes, Frame2DAttrTypes


@dataclass
class PixelSamplerConfig(InstantiateConfig):
    """Configuration for pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: PixelSampler)
    """Target class to instantiate."""
    num_rays_per_batch: int = 4096
    """Number of rays to sample per batch."""
    keep_full_image: bool = False
    """Whether or not to include a reference to the full image in returned batch."""
    is_equirectangular: bool = False
    """List of whether or not camera i is equirectangular."""


class PixelSampler:
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: PixelSamplerConfig

    def __init__(self, config: PixelSamplerConfig, **kwargs) -> None:
        self.kwargs = kwargs
        self.config = config
        # Possibly override some values if they are present in the kwargs dictionary
        self.config.num_rays_per_batch = self.kwargs.get("num_rays_per_batch", self.config.num_rays_per_batch)
        self.config.keep_full_image = self.kwargs.get("keep_full_image", self.config.keep_full_image)
        self.config.is_equirectangular = self.kwargs.get("is_equirectangular", self.config.is_equirectangular)
        self.set_num_rays_per_batch(self.config.num_rays_per_batch)

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def collate_image_dataset_batch(self, batch: Dict, ivus: torch.Tensor, keep_full_image: bool = False):
        """
        Operates on a batch of images and samples pixels to use for generating rays.
        Returns a collated batch which is input to the Graph.
        It will sample only within the valid 'mask' if it's specified.

        Args:
            batch: batch of images to sample from
            num_rays_per_batch: number of rays to sample per batch
            keep_full_image: whether or not to include a reference to the full image in returned batch
        """
        device = batch[FrameItemType.Pose].device

        c, y, x = (i.flatten() for i in torch.split(ivus, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {}
        for key, value in batch.items():
            if key in Frame1DAttrTypes:
                collated_batch[key] = value[c]
            elif key in Frame2DAttrTypes:
                collated_batch[key] = value[c, y, x]
            else:
                collated_batch[key] = value

        assert collated_batch[FrameItemType.Image].shape[0] == ivus.shape[0]
        collated_batch["indices"] = ivus  # with the abs camera indices
        if keep_full_image:
            collated_batch["full_image"] = batch[FrameItemType.Image]

        return collated_batch

    def sample_pixels(self, batch_size: int, hws: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            nonzero_indices = torch.nonzero(mask[..., 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=self.num_rays_per_batch)
            indices = nonzero_indices[chosen_indices]
        else:
            fids = (
                torch.round(torch.rand((batch_size,), device=hws.device) * (hws.shape[0] - 1))
                .long()
                .clip(max=hws.shape[0] - 1)
            )
            vus = torch.round(torch.rand((batch_size, 2), device=hws.device) * (hws[fids] - 1)).long()
            indices = torch.cat((fids.unsqueeze(-1), vus), dim=1)  # [n, 3]
        return indices

    def sample(self, batch: Dict, mask: Optional[torch.Tensor] = None):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        assert FrameItemType.Size in batch
        hws = batch[FrameItemType.Size]
        ivus = self.sample_pixels(self.num_rays_per_batch, hws, mask)
        pixel_batch = self.collate_image_dataset_batch(batch, ivus, keep_full_image=self.config.keep_full_image)
        return pixel_batch


@dataclass
class PatchPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PatchPixelSampler."""

    _target: Type = field(default_factory=lambda: PatchPixelSampler)
    """Target class to instantiate."""
    patch_size: int = 32
    """Side length of patch. This must be consistent in the method
    config in order for samples to be reshaped into patches correctly."""


class PatchPixelSampler(PixelSampler):
    """Samples 'pixel_batch's from 'image_batch's. Samples square patches
    from the images randomly. Useful for patch-based losses.

    Args:
        config: the PatchPixelSamplerConfig used to instantiate class
    """

    config: PatchPixelSamplerConfig

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch. Overridden to deal with patch-based sampling.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = (num_rays_per_batch // (self.config.patch_size**2)) * (self.config.patch_size**2)

    # overrides base method
    def sample_pixels(self, batch_size: int, hws: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()
        if isinstance(mask, Tensor):
            sub_bs = batch_size // (self.config.patch_size**2)
            half_patch_size = int(self.config.patch_size / 2)
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=half_patch_size)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=sub_bs)
            indices = nonzero_indices[chosen_indices]

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys - half_patch_size
            indices[:, ..., 2] += xxs - half_patch_size

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)
        else:
            sub_bs = batch_size // (self.config.patch_size**2)
            indices = torch.rand((sub_bs, 3), device=device) * torch.tensor(
                [num_images, image_height - self.config.patch_size, image_width - self.config.patch_size],
                device=device,
            )

            indices = (
                indices.view(sub_bs, 1, 1, 3)
                .broadcast_to(sub_bs, self.config.patch_size, self.config.patch_size, 3)
                .clone()
            )

            yys, xxs = torch.meshgrid(
                torch.arange(self.config.patch_size, device=device), torch.arange(self.config.patch_size, device=device)
            )
            indices[:, ..., 1] += yys
            indices[:, ..., 2] += xxs

            indices = torch.floor(indices).long()
            indices = indices.flatten(0, 2)

        return indices


@dataclass
class PairPixelSamplerConfig(PixelSamplerConfig):
    """Config dataclass for PairPixelSampler."""

    _target: Type = field(default_factory=lambda: PairPixelSampler)
    """Target class to instantiate."""
    radius: int = 2
    """max distance between pairs of pixels."""


class PairPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """

    def __init__(self, config: PairPixelSamplerConfig, **kwargs) -> None:
        self.config = config
        self.radius = self.config.radius
        super().__init__(self.config, **kwargs)
        self.rays_to_sample = self.config.num_rays_per_batch // 2

    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor:
        rays_to_sample = self.rays_to_sample
        if isinstance(mask, Tensor):
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)
            nonzero_indices = torch.nonzero(m[:, 0], as_tuple=False).to(device)
            chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_to_sample)
            indices = nonzero_indices[chosen_indices]
        else:
            rays_to_sample = self.rays_to_sample
            if batch_size is not None:
                assert (
                    int(batch_size) % 2 == 0
                ), f"PairPixelSampler can only return batch sizes in multiples of two (got {batch_size})"
                rays_to_sample = batch_size // 2

            s = (rays_to_sample, 1)
            ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices
