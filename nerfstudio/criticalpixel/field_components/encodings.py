from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import tinycudann as tcnn
import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.encodings import Encoding
from nerfstudio.field_components.mlp import MLP


@dataclass
class EncoderConfig:
    n_input_dims: int = 3


@dataclass
class TcnnGridEncoderConfig(EncoderConfig):
    type: Literal["hashgrid", "densegrid"] = "hashgrid"
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 21
    base_resolution: int = 16
    desired_resolution: int = 7000
    interpolation: Literal["Linear", "Smoothstep"] = "Linear"
    per_level_scale: float = -1

    def get_encoder(self):
        """reference: https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md"""

        self.per_level_scale = np.exp2(np.log2(self.desired_resolution / self.base_resolution) / (self.n_levels - 1))
        tcnn_encoding_config = {
            "otype": "Grid",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
            "interpolation": self.interpolation,
        }

        if self.type == "hashgrid":
            tcnn_encoding_config.update(
                {
                    "type": "Hash",
                    "log2_hashmap_size": self.log2_hashmap_size,
                }
            )
        elif self.type == "densegrid":
            tcnn_encoding_config = {
                "type": "Dense",
            }
        else:
            raise NotImplementedError()

        tcnn_encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=tcnn_encoding_config)
        return tcnn_encoding


@dataclass
class TcnnFrequencyEncoderConfig(EncoderConfig):
    n_frequencies: int = 8

    def get_encoder(self):
        tcnn_encoding_config = {
            "otype": "Frequency",
            "n_frequencies": self.n_frequencies,
        }
        tcnn_encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=tcnn_encoding_config)
        return tcnn_encoding


@dataclass
class TcnnSphereHarmonicsEncoderConfig(EncoderConfig):
    degree: int = 4

    def get_encoder(self):
        tcnn_encoding_config = {
            "otype": "SphericalHarmonics",
            "degree": self.degree,
        }
        tcnn_encoding = tcnn.Encoding(n_input_dims=self.n_input_dims, encoding_config=tcnn_encoding_config)
        return tcnn_encoding
