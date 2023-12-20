'''
Description: parse and process drone dataset
'''

import os
from typing import Dict, Tuple, List, Union
from pathlib import Path

import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent
from rich.progress import track

from nerfstudio.v3d.core.camera import BaseCamera
from nerfstudio.v3d.core.geometry.point_cloud import get_bbox_from_points
from nerfstudio.v3d.core.dataset.image_metadata import ImageMetadata
from nerfstudio.v3d.core.dataset.scene_metadata import SceneMetadata
from nerfstudio.v3d.core.dataset.colmap.colmap_io import read_model_concurrent, Camera, Image, Point3D
from nerfstudio.v3d.core.geometry.scene_box import SceneBox

from nerfstudio.v3d.core.geometry.depth_map import DepthMap
from nerfstudio.v3d.core.utils.file import save_data_by_zip

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


class ColmapDatasetBuilder:
    """转换标准Colmap源数据到nerf数据集格式
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def get_abspath(self, relpath) -> Path:
        return self.dataset_config.data_dir / relpath

    def get_relpath(self, abspath: Union[str, Path]) -> Path:
        return Path(abspath).relative_to(self.dataset_config.data_dir)


    def build_dataset(self) -> SceneMetadata:
        cameras, images, points3d = self._load_colmap_model()
        points3d = self._process_points3d(points3d)
        cameras = self._process_cameras(cameras)
        images_metadata = self._process_images(images, cameras)
        # bbox = get_bbox_from_points(points3d[..., :3], ignore_percentile=0.0)
        bbox = self.get_bbox_from_poses(points3d, images_metadata)
        root_dir = self.dataset_config.source_data.root_dir
        scene_metadata = SceneMetadata(root_dir, cameras, images_metadata, points3d, bbox)
        return scene_metadata

    def get_bbox_from_poses(self, points3d, images):
        centers = np.array([meta.pose[:3, 3] for meta in images.values()])
        cam_bbox = get_bbox_from_points(centers, ignore_percentile=0.0)
        cam_bbox = SceneBox(cam_bbox).enlarge_box(1).get_box().cpu().numpy()
        heights = cam_bbox[1]

        size = cam_bbox[1] - cam_bbox[0]
        center = (cam_bbox[0] + cam_bbox[1])*0.5
        ratio = np.array([1.0, 1.0, 1.0])
        cam_bbox[0] = center - size * 0.5 * ratio
        cam_bbox[1] = center + size * 0.5 * ratio
        # pts_bbox = get_bbox_from_points(points3d[:, :3], ignore_percentile=0.1)
        # cam_bbox[:, 1] = pts_bbox[:, 1]
        return cam_bbox


    def _load_colmap_model(self) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
        """读图稀疏建图结果: 相机参数、图片位姿、稀疏3D点
           无人机使用的colmap不是标准colmap, 重写了colmap的读取方法
        Returns:
            Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]: _description_
        """
        sparse_absdir = self.get_abspath(self.dataset_config.source_data.sparse_model_dir)
        cameras, images, points3D = read_model_concurrent(sparse_absdir, ext=self.dataset_config.source_data.ext)
        return cameras, images, points3D


    def _process_points3d(self, points3d: Dict[int, Point3D]) -> np.ndarray:
        pts3d_world = []
        for i, pt_idx in enumerate(points3d):
            pts3d_world.append(points3d[pt_idx].xyz.tolist() + points3d[pt_idx].rgb.tolist())
        return np.array(pts3d_world)

    def _process_cameras(self, cameras: Dict[int, Camera]) -> Dict[int, Dict]:
        """根据相机参数计算像素的射线

        Args:
            cameras (_type_): _description_
        """
        camera_models = {}
        for camera_id, camera in cameras.items():
            camera_models[camera_id] = camera.model
        return camera_models

    def _process_image(self, image_id: int, image: Image, camera: BaseCamera) -> ImageMetadata:

        image_name = image.name
        image_path = os.path.join(self.dataset_config.source_data.image_dir, image_name)
        unique_id = Path(image_path).stem

        camera_id = image.camera_id
        cam_h = camera.height
        cam_w = camera.width

        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = R
        pose_w2c[:3, 3] = t.squeeze()
        pose_c2w = np.linalg.inv(pose_w2c)

        data_paths = {
            'image_path': image_path,
        }

        depth_path = self.get_abspath(self.dataset_config.source_data.depthmap_dir) / f'{image_name}.geometric.bin'

        if depth_path.exists():                
            distance_relpath = os.path.join(self.dataset_config.distance_dir, image_name+'.zip')
            distance_abspath = self.get_abspath(distance_relpath)

            depth = read_array(depth_path)
            invalid_mask = np.bitwise_or(depth < self.dataset_config.depth_near, depth > self.dataset_config.depth_far)
            depth[invalid_mask] = np.finfo(np.float32).min
            directions = camera.get_pixelwise_image_rays().numpy()
            # depth->distance，与全景图保持一致
            depth_map = DepthMap(depth)
            distance = depth_map.depth_to_distance(depth, directions)
            
            distance_abspath.parent.mkdir(parents=True, exist_ok=True)
            save_data_by_zip(distance_abspath, distance, 'np')

            data_paths['distance_path'] = distance_relpath
            data_paths['depth_path'] = str(self.get_relpath(depth_path))

        metadata = ImageMetadata(height=cam_h,
                                    width=cam_w,
                                    camera_id=camera_id,
                                    pose_c2w=pose_c2w,
                                    unique_id=unique_id,
                                    root_dir=str(self.dataset_config.data_dir),
                                    data_paths=data_paths)
        return metadata

    def _process_images(self, images: Dict[int, Image], cameras: Dict[int, BaseCamera]) -> Dict[int, ImageMetadata]:
        image_metadata = {}

        num_workers = multiprocessing.cpu_count() - 2
        image_metadata = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for image_id, im in track(images.items(), description='process images...'):
                futures[image_id] = executor.submit(self._process_image, image_id, im, cameras[im.camera_id])

                if len(futures) > self.dataset_config.multiprocess_buffer:
                    for image_id in list(futures.keys()):
                        metadata = futures[image_id].result()
                        if metadata is not None:
                            image_metadata[image_id] = metadata
                        del futures[image_id]
            for image_id in list(futures.keys()):
                metadata = futures[image_id].result()
                if metadata is not None:
                    image_metadata[image_id] = metadata
                del futures[image_id]
        return image_metadata
