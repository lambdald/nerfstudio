"""
Multi Layer Perceptron
"""
from dataclasses import dataclass
import dataclasses
from typing import Literal, Optional, Set, Tuple, Union
import math

import torch
from torch import Tensor, nn
from jaxtyping import Float, Int

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.printing import print_tcnn_speed_warning

from nerfstudio.utils.rich_utils import CONSOLE

try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ModuleNotFoundError:
    TCNN_EXISTS = False


def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


@dataclass
class MLPGeometricInitConfig:
    init_type: Literal["off", "sphere", "plane"] = "off"
    init_bias: float = 0.5
    # sphere
    is_camera_inside: bool = False
    """Whether the cameras is near the center"""

    # plane
    plane_axis: int = 2  # 0-X, 1-Y, 2-Z
    is_camera_bottom: bool = False
    """Whether the cameras is near the bottom"""


@dataclass
class MLPConfig:
    in_dim: int = -1
    num_layers: int = 2
    layer_width: int = 64
    out_dim: int = -1
    skip_connections: Optional[Tuple[int]] = None
    activation: Optional[torch.nn.Module] = torch.nn.ReLU()
    out_activation: Optional[torch.nn.Module] = None
    implementation: Literal["tcnn", "torch"] = "tcnn"
    weight_norm: bool = False

    geometric_init: Optional[MLPGeometricInitConfig] = dataclasses.field(
        default_factory=lambda: MLPGeometricInitConfig()
    )
    """Whether the cameras is near the center"""

    def get_mlp(self):
        return MLP(
            in_dim=self.in_dim,
            num_layers=self.num_layers,
            layer_width=self.layer_width,
            out_dim=self.out_dim,
            skip_connections=self.skip_connections,
            activation=self.activation,
            out_activation=self.out_activation,
            implementation=self.implementation,
            weight_norm=self.weight_norm,
            geometric_init=self.geometric_init,
        )


class MLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        weight_norm: weight norm using torch.nn.utils.weight_norm
        geometric_init: Whether to use geometric initialization.
        is_inside: Is the camera inside the scene? For objects data, it's false, and it's ture for outdoor and indoor.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
        weight_norm: bool = False,
        geometric_init: Optional[MLPGeometricInitConfig] = MLPGeometricInitConfig(),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.weight_norm = weight_norm
        self.geometric_init = geometric_init

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("MLP")
        elif implementation == "tcnn":
            activation_str = activation_to_tcnn_string(activation)
            output_activation_str = activation_to_tcnn_string(out_activation)
            if layer_width in [16, 32, 64, 128]:
                network_config = {
                    "otype": "FullyFusedMLP",
                    "activation": activation_str,
                    "output_activation": output_activation_str,
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 1,
                }
            else:
                CONSOLE.line()
                CONSOLE.print("[bold yellow]WARNING: Using slower TCNN CutlassMLP instead of TCNN FullyFusedMLP")
                CONSOLE.print(
                    "[bold yellow]Use layer width of 16, 32, 64, or 128 to use the faster TCNN FullyFusedMLP."
                )
                CONSOLE.line()
                network_config = {
                    "otype": "CutlassMLP",
                    "activation": activation_str,
                    "output_activation": output_activation_str,
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 1,
                }

            self.tcnn_encoding = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=out_dim,
                network_config=network_config,
            )

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []

        curr_in_dim = self.in_dim
        out_dim = self.layer_width
        # mlp layers
        layers = []
        for i in range(self.num_layers):
            if i in self._skip_connections:
                curr_in_dim += self.in_dim
            elif i + 1 in self._skip_connections:
                out_dim = self.layer_width - self.in_dim
            else:
                out_dim = self.out_dim if i == self.num_layers - 1 else self.layer_width
            layer = nn.Linear(curr_in_dim, out_dim)

            if self.geometric_init != None and self.geometric_init.init_type != "off":
                # deep mlp need geometric init for better convergence
                # reference: https://github.com/Totoro97/NeuS/blob/main/models/fields.py

                if i != self.num_layers - 1:
                    self._geometric_init(
                        layer,
                        curr_in_dim,
                        out_dim,
                        i == 0,
                        skip_dim=(self.in_dim if i in self._skip_connections else 0),
                    )
                else:
                    if self.geometric_init.init_type == "sphere":
                        invert = self.geometric_init.is_camera_inside
                    elif self.geometric_init.init_type == "plane":
                        invert = self.geometric_init.is_camera_bottom
                    else:
                        raise NotImplementedError

                    self._geometric_init_sdf(layer, curr_in_dim, out_bias=self.geometric_init.init_bias, invert=invert)
            else:
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0)

            if self.weight_norm:
                layer = nn.utils.weight_norm(layer)
            layers.append(layer)
            curr_in_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def _geometric_init(self, linear, k_in, k_out, first=False, skip_dim=0):
        torch.nn.init.constant_(linear.bias, 0.0)
        torch.nn.init.normal_(linear.weight, 0.0, math.sqrt(2 / k_out))
        if first:
            CONSOLE.print("sphere geometric init...")
            torch.nn.init.constant_(linear.weight[:, 3:], 0.0)  # positional encodings
            if self.geometric_init != None and self.geometric_init.init_type == "plane":
                CONSOLE.print("plane geometric init...")
                torch.nn.init.constant_(linear.weight, 0.0)  # z plane
                torch.nn.init.normal_(linear.weight[:, self.geometric_init.plane_axis], 0.0, math.sqrt(2 / k_out))
        if skip_dim:
            torch.nn.init.constant_(linear.weight[:, -skip_dim:], 0.0)  # skip connections

    def _geometric_init_sdf(self, linear, k_in, out_bias=0.5, invert=False):
        torch.nn.init.normal_(linear.weight, mean=math.sqrt(math.pi / k_in), std=0.0001)
        torch.nn.init.constant_(linear.bias, -out_bias)
        if invert:
            linear.weight.data *= -1
            linear.bias.data *= -1

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
