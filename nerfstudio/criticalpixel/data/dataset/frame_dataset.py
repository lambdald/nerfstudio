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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameItemType
from nerfstudio.criticalpixel.data.dataset.scene_metadata import SensorMetadata


class FrameDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = [FrameItemType.Image, FrameItemType.Depth, FrameItemType.Semantic]

    def __init__(self, metadata: SensorMetadata, scale_factor: float = 1.0):
        super().__init__()
        self._metadata = metadata
        self._scale_factor = scale_factor

    def __len__(self):
        return len(self._metadata.frame_metadata)

    def get_data(self, frame_idx: int) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        return self._metadata.frame_metadata.load_data(torch.full((1,), frame_idx, dtype=torch.long))

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data
