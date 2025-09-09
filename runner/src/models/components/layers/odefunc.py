import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffeq_layers
from .squeeze import squeeze, unsqueeze

__all__ = ["ODEnet", "AutoencoderDiffEqNet"]


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class ODEnet(nn.Module):
    """Helper class to make neural nets for use in continuous normalizing flows."""

    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        layer_type="concat",
        nonlinearity="softplus",
        num_squeeze=0,
    ):
        super().__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "concat_v2": diffeq_layers.ConcatConv2d_v2,
                "concatsquash": diffeq_layers.ConcatSquashConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError(f"Unsupported stride: {stride}")

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx


class AutoencoderDiffEqNet(nn.Module):
    """Helper class to make neural nets for use in continuous normalizing flows."""

    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        layer_type="concat",
        nonlinearity="softplus",
    ):
        super().__init__()
        assert layer_type in ("ignore", "hyper", "concat", "concatcoord", "blend")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")

        self.nonlinearity = {
            "tanh": F.tanh,
            "relu": F.relu,
            "softplus": F.softplus,
            "elu": F.elu,
        }[nonlinearity]
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        encoder_layers = []
        decoder_layers = []
        hidden_shape = input_shape
        for i, (dim_out, stride) in enumerate(zip(hidden_dims + (input_shape[0],), strides)):
            if i <= len(hidden_dims) // 2:
                layers = encoder_layers
            else:
                layers = decoder_layers

            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError(f"Unsupported stride: {stride}")

            layers.append(base_layer(hidden_shape[0], dim_out, **layer_kwargs))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, t, y):
        h = y
        for layer in self.encoder_layers:
            h = self.nonlinearity(layer(t, h))

        dx = h
        for i, layer in enumerate(self.decoder_layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.decoder_layers) - 1:
                dx = self.nonlinearity(dx)
        return h, dx
