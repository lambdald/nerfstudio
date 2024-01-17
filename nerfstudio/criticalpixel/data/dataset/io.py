import cv2
from pathlib import Path
from typing import Union
import torch
import numpy as np


def read_image(path: Union[Path, str]) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def process_image(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img)


def read_mask(path: Union[Path, str]) -> np.ndarray:
    """
    load binary mask.
    Returns:
        np.ndarray: binary mask[h, w] (0: background, 255: foreground)
    """
    mask_path = Path(path)
    assert mask_path.exists(), "mask path {} does not exist".format(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) > 128
    return mask


# def process_mask(self, mask):
#     """消除一部分可能存在于边界的无效点"""
#     kErodeKernelSize = 3
#     kIterations = 5
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kErodeKernelSize, kErodeKernelSize))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=kIterations)
#     return mask

# def load_mask(self):
#     """
#     load binary mask.
#     Returns:
#         np.ndarray: binary mask[h, w] (0: background, 255: foreground)
#     """

#     mask_path = self.get_abspath(self.data_paths["mask_path"])
#     assert mask_path.exists(), "mask path {} does not exist".format(mask_path)

#     mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

#     mask_h, mask_w = mask.shape[:2]
#     if mask_h != self.h or mask_w != self.w:
#         mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
#     return mask


# def load_semantic(self):
#     semantic_path = self.root_dir / self.data_paths["semantic_path"]
#     assert os.path.exists(semantic_path), "semantic path {} does not exist".format(semantic_path)

#     semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
#     h, w = semantic.shape[:2]
#     if h != self.h or w != self.w:
#         semantic = cv2.resize(semantic, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
#     return semantic

# def load_depth(self):
#     depth_path = self.root_dir / self.data_paths["depth_path"]

#     assert os.path.exists(depth_path), "depth_sigma path {} does not exist".format(depth_path)

#     depth = load_data_by_zip(str(depth_path), "np")
#     h, w = depth.shape[:2]

#     if h != self.h or w != self.w:
#         depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
#     return depth
