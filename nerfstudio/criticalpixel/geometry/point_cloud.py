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


# class PointCloud:
#     def __init__(self, points: torch.Tensor, colors: torch.Tensor = None) -> None:
#         self.points = points
#         self.colors = colors       # uint8

#     def transform(self, transform: torch.Tensor):
#         self.vertices = self.vertices @ transform[:3, :3].T + transform[:3, 3:4].T

#     @staticmethod
#     def load(file_path: str):
#         pcd = o3d.io.read_point_cloud(file_path)
#         points = PointCloud(torch.from_numpy(np.asarray(pcd.points)))
#         return points

#     def save(self, file_path: str):
#         pcd_o3d = o3d.geometry.PointCloud()
#         pcd_o3d.points = o3d.utility.Vector3dVector(self.points.cpu().numpy().astype(np.double))
#         pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy().astype(np.double))
#         o3d.io.write_point_cloud(file_path, pcd_o3d)

#     def get_bbox(self, ignore_percentile=0) -> AxisAlignedBoundingBox:
#         bbox = get_bbox_from_points(self.points.cpu().numpy(), ignore_percentile)
#         return AxisAlignedBoundingBox(bbox)

from tensordict import tensorclass
import torch
from typing import Optional
import open3d as o3d
from pathlib import Path
import numpy as np
from rich.progress import track


@tensorclass
class PointCloud:
    points: torch.Tensor  # float32[N, 3]
    colors: Optional[torch.Tensor] = None  # uint8[N, 3]
    normal: Optional[torch.Tensor] = None

    def save(self, path: Path) -> None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.cpu().to(torch.float64).numpy())
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors.cpu().to(torch.float64).numpy())
        if self.normal is not None:
            pcd.normal = o3d.utility.Vector3dVector(self.normal.cpu().to(torch.float64).numpy())

        path.parent.mkdir(exist_ok=True, parents=True)
        o3d.io.write_point_cloud(path, pcd)
        print("save to ", path)

    def load(self, path: Path) -> "PointCloud":
        assert path.exists()
        pcd = o3d.io.read_point_cloud(path)
        points = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
        if pcd.colors:
            colors = torch.from_numpy(np.asarray(pcd.colors).astype(np.float32))

        return PointCloud(points=points, colors=colors)

    def transform(self, transform: torch.Tensor, inplace=False) -> Optional["PointCloud"]:
        assert transform.shape[0] in [3, 4] and transform.shape[1] == 4, f"error shape of transform {transform.shape}"
        rotmat = transform[:3, :3]
        new_points = self.points @ rotmat.T + transform[:3, 3:4].T
        if inplace:
            self.points = new_points
        else:
            return PointCloud(point=new_points, colors=self.colors, normal=self.normal)

    def to_o3d(self, tensor: bool = False) -> o3d.geometry.PointCloud:
        # open3d support cuda tensor

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(self.point.cpu().numpy().astype(np.float64)))

        if self.color is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray((self.color.to(torch.float32) / 255).cpu().numpy())
            )

        if tensor:
            print(self.device)
            device = o3d.core.Device(str(self.device).upper())
            tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd, device=device)
            print(tensor_pcd)
            return tensor_pcd
        return pcd

    def knn(self, num_knn=1) -> torch.Tensor:
        indices = []
        sq_dists = []

        pcd = self.to_o3d(False)

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for p in track(pcd.points, transient=True):
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
        return np.array(sq_dists), np.array(indices)

    def get_bbox(self, ignore_percentile=0) -> AxisAlignedBoundingBox:
        bbox = get_bbox_from_points(self.points.cpu().numpy(), ignore_percentile)
        return AxisAlignedBoundingBox(bbox)
