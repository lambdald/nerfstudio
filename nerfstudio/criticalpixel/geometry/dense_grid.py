from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import open3d as o3d
import nerfacc

from .bbox import SceneBox
from .point_cloud import get_bbox_from_points
from .triangle_mesh import TriangleMesh
from .ray import Ray


class DenseGrid(nn.Module):
    grid: torch.Tensor
    resolution: torch.Tensor

    def __init__(self, resolution: List, bbox: Union[np.ndarray, torch.Tensor, SceneBox]) -> None:
        super().__init__()
        ndim = len(resolution)
        # 3D feature
        self.register_buffer("grid", torch.zeros(resolution, dtype=torch.float32))
        self.register_buffer("resolution", torch.tensor(resolution).float())
        self.bbox = SceneBox(bbox)
        assert ndim == self.bbox.ndim

    def forward(self, x: torch.Tensor, is_normalized=False):
        """query occ
        Args:
            x (torch.Tensor): [N, D] in bbox
        """
        if not is_normalized:
            x = self.bbox.get_normalized_points(x)
        x = x.clip(0.0, 1.0 - torch.finfo(torch.float32).eps)
        x = torch.floor(x * self.resolution).to(torch.long)
        indices = x.unbind(-1)
        return self.grid[indices]

    def update(self, x: torch.Tensor, val: Union[torch.Tensor, float], is_normalized=False):
        """_summary_

        Args:
            x (torch.Tensor): [N, D] in [0, 1]^D
        """
        if not is_normalized:
            x = self.bbox.get_normalized_points(x, use_max_scale=False)

        x = x.clip(0.0, 1.0 - torch.finfo(torch.float32).eps)
        x = torch.floor(x * (self.resolution)).to(torch.long)
        indices = x.unbind(-1)
        if isinstance(val, float) or isinstance(val, int):
            val = torch.full(x.shape[:-1], val, device=self.grid.device)
        else:
            assert val.dtype == torch.float32
        self.grid.index_put_(indices, val, accumulate=True)

    def get_grid(self):
        return self.grid

    def get_all_cells(self) -> torch.Tensor:
        device = self.resolution.device
        n_dims = len(self.resolution)
        # Xs, Ys, Zs
        dims = []
        for res in self.resolution.tolist():
            steps = torch.linspace(0, res - 1, int(res), device=device)
            dims.append(steps)

        normalized_points = torch.stack(torch.meshgrid(dims, indexing="ij"), dim=-1).view(-1, n_dims) / self.resolution
        return self.bbox.get_denormalized_points(normalized_points, use_max_scale=False)

    def get_occupied_cell_anchors(self, threshold):
        device = self.resolution.device
        n_dims = len(self.resolution)

        points = self.get_all_cells()
        values = self(points, is_normalized=False)
        occupied_points = points[values >= threshold]

        if len(occupied_points) == 0:
            return None
        xyz = torch.meshgrid(
            [
                torch.tensor([torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps], device=device)
                for _ in range(n_dims)
            ],
            indexing="ij",
        )
        offsets = torch.stack(xyz, dim=-1).view(1, -1, n_dims) / self.resolution
        occupied_points_norm = (
            self.bbox.get_normalized_points(occupied_points, use_max_scale=False).view(-1, 1, n_dims) + offsets
        )
        occupied_points_norm = occupied_points_norm.view(-1, n_dims)
        return self.bbox.get_denormalized_points(occupied_points_norm, use_max_scale=False)

    def get_mesh(self, threshold) -> TriangleMesh:
        points = self.get_occupied_cell_anchors(threshold)
        if points is None:
            return None
        faces_of_cell = torch.tensor(
            [
                # 上下
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                # 前后
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
                # 左右
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
            ],
            device=points.device,
        )

        faces = torch.arange(len(points) // 8, dtype=torch.long, device=points.device).view(-1, 1, 1) * 8
        faces = faces + faces_of_cell
        mesh = TriangleMesh(points.float(), triangles=faces.view(-1, 3))
        return mesh

    def get_new_grid(self, resolution):
        new_grid = DenseGrid(resolution, self.bbox).to(self.resolution.device)
        cell_points = self.get_occupied_cell_anchors(1)
        if cell_points is None:
            return RuntimeError()
        new_grid.update(cell_points, self(cell_points) + 1.0)
        assert torch.all(new_grid(cell_points) > 0)
        return new_grid

    def dilate(self, kernel_size=3):
        shape = self.grid.shape
        with torch.no_grad():
            self.grid = F.max_pool3d(self.grid.view(1, 1, *shape), kernel_size=kernel_size, stride=1, padding=1).view(
                shape
            )

    def __repr__(self):
        return super().__repr__() + "\b" + f"  Grid Resolution: {self.resolution.tolist()}\n)"

    @property
    def ndim(self):
        return len(self.resolution)

    def get_tight_occupancy_grid(self):
        cell_anchors = self.get_occupied_cell_anchors(1.0)
        tight_bbox = SceneBox(get_bbox_from_points(cell_anchors.cpu().numpy()))
        cell_size = self.bbox.get_lenghts() / self.resolution
        new_resolution = tight_bbox.get_resolution(cell_size)

        tight_occupancy_grid = DenseGrid(new_resolution.tolist(), tight_bbox)
        tight_occupancy_grid.update(cell_anchors, 100.0)
        assert torch.all(tight_occupancy_grid(cell_anchors) > 0)
        return tight_occupancy_grid

    # @staticmethod
    # def from_nerfacc_occ_grid(grid: nerfacc.OccGridEstimator) -> "DenseGrid":
    #     aabb: torch.Tensor = grid._roi_aabb
    #     resolution: torch.Tensor = grid.resolution
    #     occs: torch.Tensor = grid.occs

    #     occ_grid = DenseGrid(resolution.tolist(), aabb.clone().reshape(2, 3)).to(occs.device)
    #     occ_grid.grid[:] = occs.view(resolution.tolist())
    #     return occ_grid


def get_o3d_mesh_from_occupancy_grid(grid3d: DenseGrid) -> TriangleMesh:
    assert grid3d.ndim == 3
    mesh = grid3d.get_mesh(0.5)
    o3d_mesh = mesh.o3d_mesh()

    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
    mesh = TriangleMesh.from_o3d(o3d_mesh)
    return mesh


def get_bbox_from_occupancy_grid(occupancy_grid: DenseGrid):
    bbox_anchors = occupancy_grid.get_occupied_cell_anchors(2).cpu().numpy()
    new_bbox = SceneBox(get_bbox_from_points(bbox_anchors))
    return new_bbox


def test():
    precision = 1
    resolution = torch.tensor([50, 50, 20]).cuda()
    scene_range = precision * resolution

    bbox = SceneBox(torch.tensor([[0, 0, 0], scene_range.tolist()]))

    grid3d = DenseGrid(resolution.tolist(), bbox).cuda()
    x = grid3d.get_all_cells()[: 50 * 20]
    grid3d.update(x, 100.0)

    x = grid3d.get_all_cells()[::20]
    grid3d.update(x, 100.0)
    mesh = get_o3d_mesh_from_occupancy_grid(grid3d)
    mesh.save("debug/occupancy_grid_test_ori.ply")
    print(f"{grid3d}")
    grid3d = grid3d.get_new_grid(torch.ceil(resolution / 10).to(torch.long).tolist())
    mesh = get_o3d_mesh_from_occupancy_grid(grid3d)
    o3d.io.write_triangle_mesh("debug/occupancy_grid_test_lowres.ply", mesh, compressed=True, write_ascii=False)


class VisibilityGrid(DenseGrid):
    def __init__(self, resolution: List, bbox: Union[np.ndarray, torch.Tensor, SceneBox]) -> None:
        super().__init__(resolution, bbox)

    def update(self, ray: Ray) -> None:
        pass


if __name__ == "__main__":
    test()
