import torch
import torch.nn as nn

from .wrappers import diffeq_wrapper


class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers.

    Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList([diffeq_wrapper(layer) for layer in layers])

    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


class MixtureODELayer(nn.Module):
    """Produces a mixture of experts where output = sigma(t) * f(t, x).

    Time-dependent weights sigma(t) help learn to blend the experts without resorting to a highly
    stiff f. Supports both regular and diffeq experts.
    """

    def __init__(self, experts):
        super().__init__()
        assert len(experts) > 1
        wrapped_experts = [diffeq_wrapper(ex) for ex in experts]
        self.experts = nn.ModuleList(wrapped_experts)
        self.mixture_weights = nn.Linear(1, len(self.experts))

    def forward(self, t, y):
        dys = []
        for f in self.experts:
            dys.append(f(t, y))
        dys = torch.stack(dys, 0)
        weights = self.mixture_weights(t).view(-1, *([1] * (dys.ndimension() - 1)))

        dy = torch.sum(dys * weights, dim=0, keepdim=False)
        return dy
