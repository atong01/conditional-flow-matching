from inspect import signature

import torch.nn as nn

__all__ = ["diffeq_wrapper", "reshape_wrapper"]


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        if len(signature(self.module.forward).parameters) == 1:
            self.diffeq = lambda t, y: self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            self.diffeq = self.module
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def forward(self, t, y):
        return self.diffeq(t, y)

    def __repr__(self):
        return self.diffeq.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class ReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super().__init__()
        assert (
            len(signature(net.forward).parameters) == 2
        ), "use diffeq_wrapper before reshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def reshape_wrapper(input_shape, layer):
    return ReshapeDiffEq(input_shape, layer)
