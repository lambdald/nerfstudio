"""
Description: 
"""
from typing import Dict, Union, Callable, List, Optional
from pathlib import Path

from enum import Enum
from dataclasses import dataclass
import torch
from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing import cpu_count


class FrameItemType(Enum):
    SensorId = "camera_indices"
    UniqueId = "image_indices"
    Pose = "pose"  # camera to world
    Size = "size"
    Image = "image"
    Mask = "mask"
    Depth = "depth"
    Semantic = "semantic"
    PointCloud = "pointcloud"


OptionalAttrTypes = {
    FrameItemType.Image,
    FrameItemType.Mask,
    FrameItemType.Depth,
    FrameItemType.Semantic,
    FrameItemType.PointCloud,
}

Frame1DAttrTypes = {FrameItemType.SensorId, FrameItemType.UniqueId, FrameItemType.Pose, FrameItemType.Size}

Frame2DAttrTypes = {FrameItemType.Image, FrameItemType.Depth, FrameItemType.Semantic}


@dataclass
class FrameItems:
    item: FrameItemType
    item_loader: Optional[Callable]
    item_processor: Optional[Callable]
    relpaths: List[Path]


class FrameMetadata:
    def __init__(
        self,
        root_dir: Path,
        hw: torch.Tensor,
        camera_id: torch.Tensor,
        unique_id: torch.Tensor,
        pose_c2w: torch.Tensor,
        items: Dict[FrameItemType, FrameItems],
    ):
        self.root_dir = root_dir
        if hw.shape[:-1] != pose_c2w.shape[:-1]:
            hw = hw.expand(pose_c2w.shape[:-2] + (2,))
        self.hw = hw
        self.camera_id = camera_id
        self.unique_id = unique_id
        self.pose_c2w = pose_c2w
        assert isinstance(items, dict)
        for k in items:
            assert isinstance(k, FrameItemType)
        self.items = items

    def __str__(self) -> str:
        s = "FrameMetadata:\n"
        s += f"{len(self)} frames with " + (", ".join(str(s) for s in self.items.keys()))
        return s

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return self.pose_c2w.shape[0]

    def get_abspath(self, relpath: Union[str, Path]) -> Path:
        return self.root_dir / relpath

    def __getitem__(self, idx: int):
        return self.load_data(torch.tensor([idx]).long())

    def has_item(self, item_type: FrameItemType) -> bool:
        return item_type in self.items

    def load_item(self, item_type: FrameItemType, index: int) -> torch.Tensor:
        assert self.has_item(item_type)
        item = self.items[item_type]
        data = item.item_loader(self.root_dir / item.relpaths[index])
        if item.item_processor is not None:
            data = item.item_processor(data)
        return data

    def load_items(self, item_type: FrameItemType, indices: torch.Tensor) -> torch.Tensor:
        items = []
        num_workers = cpu_count() // 2
        with ThreadPoolExecutor(num_workers) as executor:
            futures = []
            for idx in indices.view(-1).tolist():
                f: Future = executor.submit(self.load_item, item_type, idx)
                futures.append(f)

            for f in futures:
                items.append(f.result())
        return torch.stack(items, dim=0)

    def load_data(self, indices: torch.Tensor) -> Dict[FrameItemType, torch.Tensor]:
        """get all data of this frame
        Returns:
            Dict
        """
        data = {}
        for key in self.items:
            data[key] = self.load_items(key, indices)
        data[FrameItemType.SensorId] = self.camera_id[indices]
        data[FrameItemType.Pose] = self.pose_c2w[indices]
        data[FrameItemType.Size] = self.hw[indices]
        data[FrameItemType.UniqueId] = self.unique_id[indices]
        return data

    def dict(self) -> Dict:
        """get all params of this image:

        Returns:
            Dict
        """
        metadata = {
            "root_dir": str(self.root_dir),
            FrameItemType.SensorId.value: self.camera_id.tolist(),
            FrameItemType.UniqueId.value: self.unique_id.tolist(),
            FrameItemType.Size.value: self.hw.tolist(),
            FrameItemType.Pose.value: self.pose_c2w.tolist(),
        }
        metadata.update({key.value: value for key, value in self.items.items()})
        return metadata

    @staticmethod
    def from_dict(metadata_dict: Dict[str, Union[torch.Tensor, str]], root_dir: Optional[Union[str, Path]] = None):
        if root_dir is None:
            root_dir = Path(metadata_dict["root_dir"])
        else:
            root_dir = Path(root_dir)

        camera_id = torch.tensor(metadata_dict[FrameItemType.SensorId.value])
        unique_id = torch.tensor(metadata_dict[FrameItemType.UniqueId.value])
        hw = torch.tensor(metadata_dict[FrameItemType.Size.value])
        pose_c2w = torch.tensor(metadata_dict[FrameItemType.Pose.value])

        items = {op_key: metadata_dict[op_key.value] for op_key in OptionalAttrTypes if op_key.value in metadata_dict}

        return FrameMetadata(Path(root_dir), hw, camera_id, unique_id, pose_c2w, items=items)
