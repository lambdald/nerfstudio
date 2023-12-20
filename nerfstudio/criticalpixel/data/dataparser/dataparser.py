from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from enum import Enum
from typing import Type
from nerfstudio.criticalpixel.geometry.transform import CoordinateType


@dataclass
class DataParserConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: DataParser)
    coordinate_type: CoordinateType = CoordinateType.OpenCV


class DataParser:
    config: DataParserConfig

    def __init__(self, config) -> None:
        self.config = config

    def parse_data(self):
        raise NotImplementedError()
