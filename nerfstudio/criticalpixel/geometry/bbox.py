from typing import Union, List
import torch
import numpy as np
from itertools import product

import itertools
from nerfstudio.criticalpixel.geometry.range import UniformRangeWithOverlap


class AxisAlignedBoundingBox(torch.nn.Module):
    """ND Axis-Aligned Bounding Box(AABB) in [2, N] torch.FloatTensor"""

    DEFAULT_DTYPE = torch.float64

    aabb: torch.Tensor

    def __init__(self, input: Union[np.ndarray, torch.Tensor, "AxisAlignedBoundingBox"]) -> None:
        super().__init__()
        box = self._get_box_from_input(input)
        assert box.shape[0] == 2 and box.ndim == 2
        assert torch.all(box[1] > box[0])
        self.register_buffer("aabb", box)

    def _get_box_from_input(self, bbox) -> torch.Tensor:
        if isinstance(bbox, torch.Tensor):
            return torch.clone(bbox).to(torch.float64)
        elif isinstance(bbox, AxisAlignedBoundingBox):
            return bbox.get_box()
        elif isinstance(bbox, np.ndarray):
            return torch.from_numpy(bbox).to(torch.float64)
        else:
            raise NotImplementedError

    def get_normalized_points(
        self, points: torch.Tensor, dtype=DEFAULT_DTYPE, use_max_scale: bool = True
    ) -> torch.Tensor:
        """return normalized point in [0, 1]^3

        Args:
            points (torch.Tensor): _description_
            dtype (_type_, optional): _description_. Defaults to DEFAULT_DTYPE.

        Returns:
            torch.Tensor: _description_
        """
        if use_max_scale:
            scale = 1.0 / self.get_lenghts().max()
        else:
            scale = 1.0 / self.get_lenghts()

        normalized_points = (points - self.aabb[0]) * scale
        return normalized_points.to(dtype)

    def get_denormalized_points(
        self, normalized_points: torch.Tensor, dtype=DEFAULT_DTYPE, use_max_scale: bool = True
    ) -> torch.Tensor:
        if use_max_scale:
            scale = self.get_lenghts().max()
        else:
            scale = self.get_lenghts()

        points = normalized_points * scale + self.aabb[0]
        return points.to(dtype)

    def get_lenghts(self) -> torch.Tensor:
        return self.aabb[1] - self.aabb[0]

    def get_center(self) -> torch.Tensor:
        return (self.aabb[0] + self.aabb[1]) * 0.5

    def get_box(self) -> torch.Tensor:
        return torch.clone(self.aabb)

    def get_vertices(self) -> torch.Tensor:
        bbox = self.aabb.T.tolist()
        vertices = list(itertools.product(*bbox))
        return torch.tensor(vertices)

    @property
    def ndim(self) -> int:
        return self.aabb.shape[1]

    def __repr__(self) -> str:
        s = f"{self.aabb.shape[1]}D {self.__class__.__name__}: {self.aabb.tolist()}, Length: {self.get_lenghts().tolist()}"
        return s

    def enlarge_box(self, bbox_enlarge_ratio: float) -> "AxisAlignedBoundingBox":
        """Keep the center point unchanged and expand the bounding box

        Args:
            bbox_enlarge_ratio (float): new_lenght = old_length * bbox_enlarge_ratio

        Returns:
            SceneBox: new scene box
        """
        center = self.get_center()
        length = self.get_lenghts()
        half_length = length * 0.5

        min_anchor = center - half_length * bbox_enlarge_ratio
        max_anchor = center + half_length * bbox_enlarge_ratio

        new_bbox = torch.stack([min_anchor, max_anchor], dim=0)
        return AxisAlignedBoundingBox(new_bbox)

    def translate(self, translation: torch.Tensor) -> "AxisAlignedBoundingBox":
        bbox = self.aabb + translation
        return AxisAlignedBoundingBox(bbox)

    def to_list(self):
        return self.aabb.tolist()

    def get_resolution(self, cell_size: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(cell_size, torch.Tensor):
            assert cell_size.numel() == self.ndim

        resolution = self.get_lenghts() / cell_size
        resolution = torch.ceil(resolution).to(torch.long)
        return resolution

    def get_point_indices_within_box(self, points: Union[torch.Tensor, np.ndarray]):
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(self.aabb.device)

        is_vertices_in_bbox = torch.all(torch.bitwise_and(points > self.aabb[0], points < self.aabb[1]), dim=-1)

        return is_vertices_in_bbox

    def split_bbox_xy(self, cell_size_x, cell_size_y, overlap_ratio) -> List["AxisAlignedBoundingBox"]:
        print("scene bbox:", self)

        min_pos = self.aabb[0].cpu().numpy()
        max_pos = self.aabb[1].cpu().numpy()
        center = self.get_center().cpu().numpy()
        range = self.get_lenghts().cpu().numpy()

        # 先粗略计算分块的大致数量,只划分xy平面
        num_x = round(range[0] / cell_size_x)
        num_y = round(range[1] / cell_size_y)

        # 计算实际大小
        uniform_sampler_x = UniformRangeWithOverlap(min_pos[0], max_pos[0], overlap_ratio)
        centers_x, cell_size_x = uniform_sampler_x.sample(num_x)

        uniform_sampler_y = UniformRangeWithOverlap(min_pos[1], max_pos[1], overlap_ratio)
        centers_y, cell_size_y = uniform_sampler_y.sample(num_y)

        cell_size_z = range[2]
        print("range:", range.tolist())
        print("x-axis:", centers_x, cell_size_x)
        print("y-axis:", centers_y, cell_size_y)

        centers = list(product(centers_x, centers_y, [center[2]]))
        cell_size = np.array((cell_size_x, cell_size_y, cell_size_z))
        bboxes = []
        for center in centers:
            center = np.array(center)
            p1 = center - cell_size / 2
            p2 = center + cell_size / 2
            sub_bbox = np.array([p1, p2])
            bboxes.append(AxisAlignedBoundingBox(sub_bbox))
        return bboxes


AABB = AxisAlignedBoundingBox
BBox = AxisAlignedBoundingBox
SceneBox = AxisAlignedBoundingBox


def test():
    aabb = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    box = AxisAlignedBoundingBox(aabb)
    print(box.get_lenghts(), box.get_center())

    aabb = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int) * 100
    box = AxisAlignedBoundingBox(aabb)
    print(box.get_lenghts(), box.get_center())

    aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.int) * 500
    box = AxisAlignedBoundingBox(aabb)
    print(box.get_lenghts(), box.get_center())

    aabb = AxisAlignedBoundingBox(box)
    print(aabb)
    points = torch.rand([100, 3]) * 10
    normalized_pts = aabb.get_normalized_points(points)
    denormalized_points = aabb.get_denormalized_points(normalized_pts)
    # assert torch.equal(points, denormalized_points)
    diff = points - denormalized_points
    print(torch.max(torch.abs(diff)))

    aabb2d = np.array([[2, 3], [5, 7]])
    bbox2d = AxisAlignedBoundingBox(aabb2d)
    points2d = torch.rand([100, 2]) * 10

    normalized_pts2d = bbox2d.get_normalized_points(points2d)
    denormalized_points2d = bbox2d.get_denormalized_points(normalized_pts2d)
    # assert torch.equal(points, denormalized_points)
    diff = points2d - denormalized_points2d
    print(torch.max(torch.abs(diff)))

    print(aabb)
    new_bbox = aabb.enlarge_box(1.1)
    print(new_bbox)


if __name__ == "__main__":
    test()
