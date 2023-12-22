from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Type, List
from nerfstudio.criticalpixel.data.dataset.scene_metadata import SceneMetadata, SensorMetadata
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameMetadata, FrameItems
import torch


@dataclass
class DataPartationConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: DataPartation)
    num_eval_data: int = 10


class DataPartation:
    config: DataPartationConfig

    def __init__(self, config) -> None:
        self.config = config

    def run(self, scene_metadata: SceneMetadata) -> List[SceneMetadata]:
        train_sensor_metadatas = {}
        eval_sensor_metadatas = {}
        for sensor_name in scene_metadata.sensor_metadatas:
            sensor_data = scene_metadata.sensor_metadatas[sensor_name]
            train_frame_metadata, eval_frame_metadata = self.split_frame_metadata(sensor_data.frame_metadata)

            train_sensor_data = SensorMetadata(
                sensor_data.root_dir, sensor_data.sensor_name, sensor_data.camera, train_frame_metadata
            )
            eval_sensor_data = SensorMetadata(
                sensor_data.root_dir, sensor_data.sensor_name, sensor_data.camera, eval_frame_metadata
            )

            train_sensor_metadatas[sensor_name] = train_sensor_data
            eval_sensor_metadatas[sensor_name] = eval_sensor_data

        train_scene_metadata = SceneMetadata(
            scene_metadata.root_dir,
            train_sensor_metadatas,
            points3d=scene_metadata.points3d,
            bbox=scene_metadata.bbox,
            transform_w2n=scene_metadata.transform_w2n,
        )

        eval_scene_metadata = SceneMetadata(
            scene_metadata.root_dir,
            eval_sensor_metadatas,
            points3d=scene_metadata.points3d,
            bbox=scene_metadata.bbox,
            transform_w2n=scene_metadata.transform_w2n,
        )

        return train_scene_metadata, eval_scene_metadata

    def split_frame_metadata(self, frame_metadata: FrameMetadata):
        raise NotImplementedError()


@dataclass
class FullDataPartationConfig(DataPartationConfig):
    _target: Type = field(default_factory=lambda: FullDataPartation)


class FullDataPartation(DataPartation):
    """
    Split full data to train and eval data.
    """

    config: FullDataPartationConfig

    def __init__(self, config) -> None:
        self.config = config

    def split_frame_metadata(self, frame_metadata: FrameMetadata):
        print("num_data", len(frame_metadata))
        N = len(frame_metadata)

        perm = torch.randperm(N)
        eval_idx = perm[: self.config.num_eval_data]
        train_idx = perm[self.config.num_eval_data :]

        full_data = frame_metadata.dict()
        test_data = {}
        train_data = {}

        for key, data in full_data.items():
            if isinstance(data, torch.Tensor):
                test_data[key] = data[eval_idx]
            elif isinstance(data, FrameItems):
                test_data[key] = FrameItems(
                    data.item,
                    data.item_loader,
                    data.item_processor,
                    relpaths=[data.relpaths[i] for i in eval_idx.tolist()],
                )
            else:
                test_data[key] = data

            if isinstance(data, torch.Tensor):
                train_data[key] = data[train_idx]
            elif isinstance(data, FrameItems):
                train_data[key] = FrameItems(
                    data.item,
                    data.item_loader,
                    data.item_processor,
                    relpaths=[data.relpaths[i] for i in train_idx.tolist()],
                )
            else:
                train_data[key] = data
        return FrameMetadata.from_dict(train_data), FrameMetadata.from_dict(test_data)


@dataclass
class TrainFullDataPartationConfig(DataPartationConfig):
    _target: Type = field(default_factory=lambda: TrainFullDataPartation)


class TrainFullDataPartation(DataPartation):
    """
    Train using full data and select eval data from full data.
    """

    config: TrainFullDataPartationConfig

    def __init__(self, config) -> None:
        self.config = config

    def split_frame_metadata(self, frame_metadata: FrameMetadata):
        print("num_data", len(frame_metadata))
        N = len(frame_metadata)

        perm = torch.randperm(N)
        idx = perm[: self.config.num_eval_data]

        full_data = frame_metadata.dict()
        test_data = {}

        for key, data in full_data.items():
            if isinstance(data, torch.Tensor):
                test_data[key] = data[idx]
            elif isinstance(data, FrameItems):
                test_data[key] = FrameItems(
                    data.item, data.item_loader, data.item_processor, relpaths=[data.relpaths[i] for i in idx.tolist()]
                )
            elif isinstance(data, list):
                assert len(data) == N
                test_data[key] = [data[i] for i in idx.tolist()]
            else:
                test_data[key] = data
        return frame_metadata, FrameMetadata.from_dict(test_data)
