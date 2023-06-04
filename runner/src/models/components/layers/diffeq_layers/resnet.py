import torch.nn as nn

from . import basic, container

NGROUPS = 16


class ResNet(container.SequentialDiffEq):
    def __init__(self, dim, intermediate_dim, n_resblocks, conv_block=None):
        super().__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks

        layers = []
        layers.append(conv_block(dim, intermediate_dim, ksize=3, stride=1, padding=1, bias=False))
        for _ in range(n_resblocks):
            layers.append(BasicBlock(intermediate_dim, conv_block))
        layers.append(nn.GroupNorm(NGROUPS, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_block(intermediate_dim, dim, ksize=1, bias=False))

        super().__init__(*layers)

    def __repr__(self):
        return (
            "{name}({dim}, intermediate_dim={intermediate_dim}, n_resblocks={n_resblocks})".format(
                name=self.__class__.__name__, **self.__dict__
            )
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim, conv_block=None):
        super().__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.norm1 = nn.GroupNorm(NGROUPS, dim, eps=1e-4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(NGROUPS, dim, eps=1e-4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_block(dim, dim, ksize=3, stride=1, padding=1, bias=False)

    def forward(self, t, x):
        residual = x

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)

        out += residual

        return out
