"""
Description: scene metadata
"""

from typing import Dict, Optional, List
import json
import copy
import torch
import itertools
import numpy as np
from pathlib import Path
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameMetadata, FrameItemType
from nerfstudio.criticalpixel.camera.camera import Camera, create_camera_from_dict
from nerfstudio.criticalpixel.geometry.bbox import AxisAlignedBoundingBox
from nerfstudio.criticalpixel.geometry.point_cloud import PointCloud
from nerfstudio.criticalpixel.geometry.transform import Transform3d
import bisect
import yaml


class SensorMetadata:
    def __init__(self, root_dir: str, sensor_name: str, camera: Camera, frame_metadata: FrameMetadata) -> None:
        self.root_dir = root_dir
        self.sensor_name = sensor_name
        self.camera = camera
        self.frame_metadata = frame_metadata

    def __len__(self):
        return len(self.frame_metadata)

    def __str__(self):
        return str(self.camera) + "\n" + str(self.frame_metadata)

    def dict(self) -> Dict:
        data = {"sensor_name": self.sensor_name, "camera": self.camera.to_dict(), "frame": self.frame_metadata.dict()}
        return data

    @staticmethod
    def from_dict(data, root_dir: Optional[Path] = None) -> "SensorMetadata":
        camera = create_camera_from_dict(data["camera"])
        frame_metadata = FrameMetadata.from_dict(data["frame"], root_dir=root_dir)
        return SensorMetadata(
            frame_metadata.root_dir, sensor_name=data["sensor_name"], camera=camera, frame_metadata=frame_metadata
        )


class SceneMetadata:
    def __init__(
        self,
        root_dir: Path,
        sensor_metadatas: Dict[str, SensorMetadata],
        points3d: PointCloud,
        bbox: AxisAlignedBoundingBox,
        transform_w2n: Transform3d,
    ) -> None:
        self.root_dir = root_dir
        self.sensor_metadatas = sensor_metadatas
        self.points3d = points3d
        self.bbox = bbox
        self.scene_box = self.bbox
        self.transform_w2n = transform_w2n

    @property
    def cameras(self):
        sensors = list(m.camera for m in self.sensor_metadatas.values())
        return sensors

    def get_num_sensors(self) -> int:
        return len(self.sensor_metadatas)

    def save(self, output_path: Path):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        data = self.dict()
        with output_path.open("w") as outfile:
            yaml.dump(data, outfile, Dumper=yaml.Dumper)
        print("save scene metadata to ", output_path)

    @staticmethod
    def load(metadata_path: str, root_dir: str = None):
        with open(metadata_path, "r") as f:
            scene_metadata = yaml.load(f, Loader=yaml.Loader)
        scene_metadata = SceneMetadata.from_dict(scene_metadata, root_dir=root_dir)
        return scene_metadata

    def __len__(self):
        s = 0
        for metadata in self.sensor_metadatas.values():
            s += len(metadata)
        return s

    def __getitem__(self, idx: int):
        lens = [0]
        for metadata in self.sensor_metadatas.values():
            lens.append(len(metadata))
        acc_lens = list(itertools.accumulate(lens))
        sensor_idx = bisect.bisect_left(acc_lens, idx) - 1

        sensor_metadata = list(self.sensor_metadatas.values())[sensor_idx]

        data = sensor_metadata.frame_metadata[idx - acc_lens[sensor_idx]]
        return data

    def get_viewer_json(self, idx: int):
        lens = [0]
        for metadata in self.sensor_metadatas.values():
            lens.append(len(metadata))
        acc_lens = list(itertools.accumulate(lens))
        sensor_idx = bisect.bisect_right(acc_lens, idx) - 1
        sensor_metadata = list(self.sensor_metadatas.values())[sensor_idx]

        data = sensor_metadata.frame_metadata[idx - acc_lens[sensor_idx]]

        image = data[FrameItemType.Image].squeeze(0)
        camera = sensor_metadata.camera[data[FrameItemType.SensorId]]

        bgr = image[..., [2, 1, 0]]
        camera_json = camera.to_json(
            camera_idx=idx, pose_c2w=data[FrameItemType.Pose].squeeze(0), image=bgr, max_size=100
        )

        return camera_json

    @staticmethod
    def from_dict(data: Dict, root_dir: Optional[Path] = None) -> "SceneMetadata":
        if root_dir is None:
            root_dir = Path(data["root_dir"])

        bbox = AxisAlignedBoundingBox(np.array(data["bbox"]))
        transform = Transform3d(torch.tensor(data["transform"]))
        pointcloud = PointCloud(torch.tensor(data["pointcloud"]))

        sensor_metadatas = {}
        for sensor_name in data["sensors"]:
            sensor_metadatas[sensor_name] = SensorMetadata.from_dict(data["sensors"][sensor_name], root_dir=root_dir)
        scene_metadata = SceneMetadata(
            root_dir=root_dir,
            sensor_metadatas=sensor_metadatas,
            points3d=pointcloud,
            bbox=bbox,
            transform_w2n=transform,
        )
        return scene_metadata

    def dict(self):
        data = {"sensors": {}}
        for sensor in self.sensor_metadatas:
            data["sensors"][sensor] = self.sensor_metadatas[sensor].dict()
        data["root_dir"] = str(self.root_dir)
        data["bbox"] = self.bbox.aabb.tolist()
        data["transform"] = self.transform_w2n.trans.tolist()
        data["pointcloud"] = self.points3d.points.tolist()
        return data

    def get_num_images(self):
        raise NotImplementedError

    def get_num_cameras(self):
        raise NotImplementedError

    def __str__(self) -> str:
        s = "SceneMetadata:"
        for sensor_name in self.sensor_metadatas:
            s += "\n" + str(self.sensor_metadatas[sensor_name])
        return s


# def filter_scene_metadata_by_camera_ids(scene_metadata: SceneMetadata, camera_ids: list):
#     image_ids_to_delete = []
#     camera_ids = set(camera_ids)
#     for image_id in scene_metadata.images:
#         if scene_metadata.images[image_id].camera_id in camera_ids:
#             image_ids_to_delete.append(image_id)
#     new_scene_metadata = copy.deepcopy(scene_metadata)
#     for key in image_ids_to_delete:
#         del new_scene_metadata.images[key]
#     return new_scene_metadata


# def filter_scene_metadata_by_image_dirs(scene_metadata: SceneMetadata, image_dirs: list):
#     image_ids_to_delete = []
#     for image_id in scene_metadata.images:
#         image_relpath: str = scene_metadata.images[image_id].data_paths["image_path"]

#         for image_dir in image_dirs:
#             if image_relpath.startswith(image_dir):
#                 print("delete image in ", image_dir, ":", image_relpath)
#                 image_ids_to_delete.append(image_id)
#                 continue

#     new_scene_metadata = copy.deepcopy(scene_metadata)
#     for key in image_ids_to_delete:
#         del new_scene_metadata.images[key]
#     return new_scene_metadata
