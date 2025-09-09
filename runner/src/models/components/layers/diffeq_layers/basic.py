import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 or classname.find("Conv") != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[: self.dim_out].view(self.dim_out)
        w = params[self.dim_out :].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))) + self._hyper_bias(
            t.view(1, 1)
        )


class HyperConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        assert (
            dim_in % groups == 0 and dim_out % groups == 0
        ), "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(
                self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize
            )
        else:
            weight = params[:weight_size].view(
                self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize
            )
        bias = params[: self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class IgnoreConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        return self._layer(x)


class SquashConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)


class ConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._hyper_gate = nn.Linear(1, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(1, 1))).view(
            1, -1, 1, 1
        ) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)


class ConcatCoordConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.layer_f = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )
        self.layer_g = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
    ):
        super().__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super().__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class BlendConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
        **unused_kwargs,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._layer1 = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t
