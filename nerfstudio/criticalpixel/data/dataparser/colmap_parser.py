from nerfstudio.criticalpixel.data.dataparser.colmap.model import read_model
from nerfstudio.configs.base_config import InstantiateConfig
from dataclasses import dataclass, field
from typing import Type, Dict
from nerfstudio.criticalpixel.data.dataset.scene_metadata import FrameMetadata, SensorMetadata, SceneMetadata
from nerfstudio.criticalpixel.data.dataset.frame_metadata import FrameItems, FrameItemType
from pathlib import Path
from rich import print
from nerfstudio.criticalpixel.camera.camera import CameraModel, Camera, create_camera, create_camera_from_dict
import torch
import numpy as np
from nerfstudio.criticalpixel.data.dataset.io import read_image, process_image
from nerfstudio.criticalpixel.geometry.point_cloud import PointCloud
from nerfstudio.criticalpixel.geometry.transform import Transform3d
from nerfstudio.criticalpixel.data.dataparser.dataparser import DataParserConfig, CoordinateType
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig, DataParser


@dataclass
class ColmapDataparserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: ColmapDataparser)
    sparse_reldir: Path = Path("sparse")
    image_reldir: Path = Path("images")
    scene_metadata_path: Path = Path("nerf/scene_metadata.yaml")
    coordinate_type: CoordinateType = CoordinateType.OpenGL


class ColmapDataparser(DataParser):
    config: ColmapDataparserConfig

    def __init__(self, config) -> None:
        self.config = config

    def get_abspath(self, path):
        print(self.config)
        return self.config.data / path

    def get_pointcloud(self, points: Dict) -> PointCloud:
        point_list = []
        rgb_list = []
        for point_id, point in points.items():
            point_list.append(point.xyz)
            rgb_list.append(point.rgb)

        points = np.array(point_list)
        rgbs = np.array(rgb_list)

        pc = PointCloud(torch.from_numpy(points), torch.from_numpy(rgbs))
        return pc

    def parse_data(self) -> SceneMetadata:
        if self.get_abspath(self.config.sparse_reldir / "0").exists():
            sparse_dir = self.get_abspath(self.config.sparse_reldir / "0")
        else:
            sparse_dir = self.get_abspath(self.config.sparse_reldir)

        cameras, images, points3D = read_model(str(sparse_dir))
        camera_dict = {}
        camera_ids = {}
        for cam_id, cam in cameras.items():
            if cam.model == "OPENCV":
                cam_type = CameraModel.OpenCV
            elif cam.model == "PINHOLE":
                cam_type = CameraModel.Pinhole
            elif cam.model == "FISHEYE":
                cam_type = CameraModel.Fisheye
            elif cam.model == "SIMPLE_RADIAL":
                cam_type = CameraModel.SimpleRadial
            else:
                raise NotImplementedError()

            if cam_type not in camera_dict:
                camera_dict[cam_type] = []
                camera_ids[cam_type] = []
            # put cameras with same camera to list
            camera_dict[cam_type].append(cam)
            camera_ids[cam_type].append(cam.id)

        print(camera_ids)

        frames_with_same_camtype = {cam_type: [] for cam_type in camera_dict}
        for img_id, img in images.items():
            for cam_type, cam_ids in camera_ids.items():
                if img.camera_id in cam_ids:
                    frames_with_same_camtype[cam_type].append(img_id)

        sensor_metadatas = {}

        for cam_type in camera_dict:
            print(cam_type)

            cams = camera_dict[cam_type]
            cam_ids = camera_ids[cam_type]

            hws = np.array([(cam.height, cam.width) for cam in cams])
            hws = torch.tensor(hws, dtype=torch.int32)

            params = np.array([cam.params for cam in cams])
            params = torch.tensor(params, dtype=torch.float32)

            if cam_type == CameraModel.OpenCV:
                params = torch.cat([params, torch.zeros_like(params[..., :1])], dim=1)

            current_cameras = create_camera(cam_type, hws, params)
            current_cam_ids = torch.tensor(cam_ids)

            print(current_cameras)
            cam_ids = camera_ids[cam_type]
            frame_ids = frames_with_same_camtype[cam_type]

            current_image_ids = np.array([images[img_id].id for img_id in frame_ids])
            current_image_camera_ids = np.array([images[img_id].camera_id for img_id in frame_ids])

            current_image_local_camera_ids = []
            for cam_id in current_image_camera_ids:
                # print(cam_id)
                local_cam_id = (current_cam_ids == cam_id).nonzero()
                assert local_cam_id.numel() == 1
                current_image_local_camera_ids.append(local_cam_id)
            current_image_local_camera_ids = torch.tensor(current_image_local_camera_ids)

            current_image_rotmat = np.array([images[img_id].qvec2rotmat() for img_id in frame_ids])
            current_image_tvec = np.array([images[img_id].tvec for img_id in frame_ids])

            pose_w2c = np.concatenate([current_image_rotmat, current_image_tvec[..., np.newaxis]], axis=-1)
            pose_btm = np.stack([np.array([[0.0, 0.0, 0.0, 1.0]]) for _ in range(len(pose_w2c))])
            pose_w2c = np.concatenate([pose_w2c, pose_btm], axis=1)
            pose_w2c = torch.from_numpy(pose_w2c)
            pose_c2w = torch.inverse(pose_w2c)

            if self.config.coordinate_type == CoordinateType.OpenGL:
                pose_c2w[..., 1:3] *= -1

            print("pose.shape=", pose_w2c.shape)

            current_image_names = [self.config.image_reldir / images[img_id].name for img_id in frame_ids]
            # print(current_image_names)

            current_frames = FrameItems(
                FrameItemType.Image,
                item_loader=read_image,
                item_processor=process_image,
                relpaths=current_image_names,
            )
            items = {FrameItemType.Image: current_frames}
            frame_metadata = FrameMetadata(
                root_dir=self.config.data,
                hw=hws,
                camera_id=current_image_local_camera_ids,
                unique_id=torch.from_numpy(current_image_ids),
                pose_c2w=pose_c2w,
                items=items,
            )
            sensor_metadata = SensorMetadata(
                self.config.data, sensor_name=cam_type.name, camera=current_cameras, frame_metadata=frame_metadata
            )

            sensor_metadatas[cam_type.name] = sensor_metadata

        pointcloud = self.get_pointcloud(points3D)
        bbox = pointcloud.get_bbox(ignore_percentile=0.01)
        scene_metadata = SceneMetadata(
            self.config.data,
            sensor_metadatas=sensor_metadatas,
            points3d=pointcloud,
            bbox=bbox,
            transform_w2n=Transform3d(torch.eye(4)),
        )
        scene_metadata.save(self.get_abspath(self.config.scene_metadata_path))
        return scene_metadata
