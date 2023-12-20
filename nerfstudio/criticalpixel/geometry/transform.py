import torch
from enum import Enum


class CoordinateType(Enum):
    OpenCV = "Coord_OpenCV"
    OpenGL = "Coord_OpenGL"


class Transform3d:
    def __init__(self, transform: torch.Tensor) -> None:
        self.trans = transform

    def transform(self, points: torch.Tensor):
        pass
