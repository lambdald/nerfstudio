"""
Code for sampling images from a dataset of images.
"""

# for multithreading
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from nerfstudio.criticalpixel.data.dataset.frame_dataset import FrameDataset
from nerfstudio.criticalpixel.utils.tensor import move_tensor_to_device, collate_list_of_dict


class CacheDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        num_images_to_load: int = -1,
        num_iters_to_reload_images: int = -1,
        device: Union[torch.device, str] = "cpu",
        exclude_batch_keys_from_device: Optional[List] = None,
        collate_fn: Callable = collate_list_of_dict,
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_iters_to_reload_images = num_iters_to_reload_images
        self.cache_all_images = (num_images_to_load == -1) or (num_images_to_load >= len(self.dataset))
        self.num_images_to_load = len(self.dataset) if self.cache_all_images else num_images_to_load
        self.device = device
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device
        self.collate_fn = collate_fn
        self.num_repeated = self.num_iters_to_reload_images  # starting value
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_iters_to_reload_images == -1:
            print(f"Caching {self.num_images_to_load} out of {len(self.dataset)} images, without resampling.")
        else:
            print(
                f"Caching {self.num_images_to_load} out of {len(self.dataset)} images, "
                f"resampling every {self.num_iters_to_reload_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_load)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = move_tensor_to_device(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_iters_to_reload_images != -1 and self.num_repeated >= self.num_iters_to_reload_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_iters_to_reload_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


class FrameDataloader(DataLoader):
    """Collated image dataset that implements caching of default-pytorch-collatable data.
    Creates batches of the InputDataset return type.

    Args:
        dataset: Dataset to sample from.
        num_samples_to_collate: How many images to sample rays for each batch. -1 for all images.
        num_times_to_repeat_images: How often to collate new images. -1 to never pick new images.
        device: Device to perform computation.
        collate_fn: The function we will use to collate our training data
    """

    def __init__(
        self,
        dataset: Dataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        if exclude_batch_keys_from_device is None:
            exclude_batch_keys_from_device = ["image"]
        self.dataset = dataset
        assert isinstance(self.dataset, Sized)

        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset
        self.num_images_to_load = len(self.dataset) if self.cache_all_images else num_images_to_load
        self.device = device
        self.num_workers = kwargs.get("num_workers", 0)
        self.exclude_batch_keys_from_device = exclude_batch_keys_from_device
        self.first_time = True

        self.cached_collated_batch = None
        if self.cache_all_images:
            print(f"Caching all {len(self.dataset)} images.")
            if len(self.dataset) > 500:
                print(
                    "[bold yellow]Warning: If you run out of memory, try reducing the number of images to sample from."
                )
            self.cached_collated_batch = self._get_collated_batch()
        elif self.num_iters_to_reload_images == -1:
            print(f"Caching {self.num_images_to_load} out of {len(self.dataset)} images, without resampling.")
        else:
            print(
                f"Caching {self.num_images_to_load} out of {len(self.dataset)} images, "
                f"resampling every {self.num_iters_to_reload_images} iters."
            )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)
        indices = random.sample(range(len(self.dataset)), k=self.num_images_to_load)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())
        return batch_list

    def _get_collated_batch(self):
        """Returns a collated batch."""
        batch_list = self._get_batch_list()
        collated_batch = self.collate_fn(batch_list)
        collated_batch = move_tensor_to_device(
            collated_batch, device=self.device, exclude=self.exclude_batch_keys_from_device
        )
        return collated_batch

    def __iter__(self):
        while True:
            if self.cache_all_images:
                collated_batch = self.cached_collated_batch
            elif self.first_time or (
                self.num_iters_to_reload_images != -1 and self.num_repeated >= self.num_iters_to_reload_images
            ):
                # trigger a reset
                self.num_repeated = 0
                collated_batch = self._get_collated_batch()
                # possibly save a cached item
                self.cached_collated_batch = collated_batch if self.num_iters_to_reload_images != 0 else None
                self.first_time = False
            else:
                collated_batch = self.cached_collated_batch
                self.num_repeated += 1
            yield collated_batch


# class EvalDataloader(DataLoader):
#     """Evaluation dataloader base class

#     Args:
#         input_dataset: InputDataset to load data from
#         device: Device to load data to
#     """

#     def __init__(
#         self,
#         input_dataset: Dataset,
#         device: Union[torch.device, str] = "cpu",
#         **kwargs,
#     ):
#         self.input_dataset = input_dataset
#         self.device = device
#         self.kwargs = kwargs
#         super().__init__(dataset=input_dataset)

#     @abstractmethod
#     def __iter__(self):
#         """Iterates over the dataset"""
#         return self

#     @abstractmethod
#     def __next__(self) -> Tuple[RayBundle, Dict]:
#         """Returns the next batch of data"""

#     def get_camera(self, image_idx: int = 0) -> Cameras:
#         """Get camera for the given image index

#         Args:
#             image_idx: Camera image index
#         """
#         return self.cameras[image_idx]

#     def get_data_from_image_idx(self, image_idx: int) -> Tuple[RayBundle, Dict]:
#         """Returns the data for a specific image index.

#         Args:
#             image_idx: Camera image index
#         """
#         ray_bundle = self.cameras.generate_rays(camera_indices=image_idx, keep_shape=True)
#         batch = self.input_dataset[image_idx]
#         batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
#         assert isinstance(batch, dict)
#         return ray_bundle, batch


# class RandIndicesEvalDataloader(EvalDataloader):
#     """Dataloader that returns random images.
#     Args:
#         input_dataset: InputDataset to load data from
#         device: Device to load data to
#     """

#     def __iter__(self):
#         return self

#     def __next__(self):
#         # choose a random image index
#         image_idx = random.randint(0, len(self.cameras) - 1)
#         ray_bundle, batch = self.get_data_from_image_idx(image_idx)
#         return ray_bundle, batch
