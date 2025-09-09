from typing import List, Optional

import torch
from torch import nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str,
        batch_norm: bool = True,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        dims = [input_size, *hidden_dims, target_size]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DivergenceFreeNet(SimpleDenseNet):
    """Implements a divergence free network as the gradient of a scalar potential function."""

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=1, *args, **kwargs)

    def energy(self, x):
        return self.model(x)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        x = torch.cat([t, x], dim=-1)
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.model(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class TimeInvariantVelocityNet(SimpleDenseNet):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim, target_size=dim, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        del t
        return self.model(x)


class VelocityNet(SimpleDenseNet):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=dim, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        x = torch.cat([t, x], dim=-1)
        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
    _ = TimeInvariantVelocityNet()
