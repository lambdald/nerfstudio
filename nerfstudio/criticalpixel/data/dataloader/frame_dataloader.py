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

    dataset: FrameDataset

    def __init__(
        self,
        dataset: Dataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        assert isinstance(self.dataset, Sized)
        super().__init__(dataset=dataset, **kwargs)  # This will set self.dataset

        

    def __iter__(self):
        return self

    def __next__(self):
        # choose a random image index
        image_idx = random.randint(0, len(self.dataset) - 1)
        ray_bundle, batch = self.dataset.get_data(image_idx)
        return ray_bundle, batch