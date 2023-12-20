import torch
# import open3d as o3d
import numpy as np
from .bbox import AxisAlignedBoundingBox


def get_bbox_from_points(points: np.ndarray, ignore_percentile=0):
    """_summary_

    Args:
        points (_type_): [N, d]
        ignore_percentile: 忽略首尾比较稀疏的点

    Returns:
        _type_: _description_
    """
    d = points.shape[1]
    points = np.array(points).T

    bbox = np.zeros((2, d), np.float64)
    if points.size == 0:
        return bbox
    for i in range(d):
        bbox[:, i] = [np.percentile(points[i], ignore_percentile), np.percentile(points[i], 100 - ignore_percentile)]
    center = np.mean(bbox, axis=0)
    scene_range = bbox[1] - bbox[0]
    bbox[0] = center - scene_range / 2
    bbox[1] = center + scene_range / 2
    return bbox


class PointCloud:
    def __init__(self, points: torch.Tensor, colors: torch.Tensor = None) -> None:
        self.points = points
        self.colors = colors       # uint8

    def transform(self, transform: torch.Tensor):
        self.vertices = self.vertices @ transform[:3, :3].T + transform[:3, 3:4].T

    @staticmethod
    def load(file_path: str):
        pcd = o3d.io.read_point_cloud(file_path)
        points = PointCloud(torch.from_numpy(np.asarray(pcd.points)))
        return points

    def save(self, file_path: str):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.points.cpu().numpy().astype(np.double))
        pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy().astype(np.double))
        o3d.io.write_point_cloud(file_path, pcd_o3d)

    def get_bbox(self, ignore_percentile=0) -> AxisAlignedBoundingBox:
        bbox = get_bbox_from_points(self.points.cpu().numpy(), ignore_percentile)
        return AxisAlignedBoundingBox(bbox)
