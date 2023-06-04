from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LocallyConnected


class ResNetBlock(nn.Module):
    def __init__(self, in_size=16, h_size=16, out_size=16):
        super().__init__()

        layer = nn.ModuleList()
        layer.append(nn.Linear(in_features=in_size, out_features=h_size))
        layer.append(nn.ReLU())
        layer.append(nn.Linear(in_features=h_size, out_features=out_size))

        self.f = nn.Sequential(*layer)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return F.relu(self.f(x) + self.shortcut(x))


class HyperResNet(nn.Module):
    def __init__(self, in_size=16, h_size=16, out_size=16, num_block=2):
        super().__init__()

        blocks = nn.ModuleList()
        for _ in range(num_block):
            blocks.append(ResNetBlock(in_size, h_size, out_size))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class HyperLocallyConnected(nn.Module):
    """Hyper Local linear layer, i.e. Conv1dLocal() with filter size 1 which parameters are learned
    from another netwokr:

            y = LocallyConnected_{params}(x),
            where params = h(G)

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    VALID_HYPER = [
        "mlp",
        "gnn",
        "invariant",
        "per_graph",
        "deep_set",
    ]

    def __init__(
        self,
        num_linear,
        input_features,
        output_features,
        hyper,
        n_ens=1,
        bias=True,
        hyper_hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.n_ens = n_ens
        self.hyper = hyper

        assert (
            hyper in self.VALID_HYPER
        ), f"hyper hparam not a valid option - choices: {self.VALID_HYPER}"

        if hyper == "invariant":
            hyper_type = HyperInvariant
        elif hyper == "mlp":
            hyper_type = HyperMLP
        elif hyper == "per_graph":
            hyper_type = HyperInvariantPerGraph
        elif hyper == "deep_set":
            hyper_type = DeepSet

        self.hyper_layer = hyper_type(
            n_ens=n_ens,
            num_linear=num_linear,
            input_features=input_features,
            output_features=output_features,
            bias=bias,
            hidden_dims=hyper_hidden_dims,
        )

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # [n_ens, n, d, 1, m2] = [n_ens, n, d, 1, m1] @ [n_ens, 1, d, m1, m2]
        weights, biases = self.hyper_layer(G.to(x))
        x = torch.matmul(x, weights.unsqueeze(1))
        if biases is not None:
            # [n, d, m2] += [d, m2]
            x += biases.unsqueeze(-2).unsqueeze(1)
        return x


class AnalyiticLinearLocallyConnected(nn.Module):
    """Analytic linear Local linear layer, i.e. Conv1dLocal() with filter size 1 which parameters
    are learned from another netwokr:

            y = LocallyConnected_{params}(x),
            where params = h(G)

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(
        self,
        num_linear,
        input_features,
        hyper,
        n_ens=1,
        bias=True,
        hyper_hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.n_ens = n_ens
        self.hyper = hyper

        self.hyper_layer = HyperAnalyticLinear(
            n_ens=n_ens,
            num_linear=num_linear,
            input_features=input_features,
        )

        self.weights = torch.randn((n_ens, num_linear, num_linear))

    def forward(self, x: torch.Tensor, dx: torch.Tensor, G: torch.Tensor):
        # [n_ens, n, d, 1, m2] = [n_ens, n, d, 1, m1] @ [n_ens, 1, d, m1, m2]
        self.weights = self.hyper_layer(x, dx, G.to(x))
        x = torch.matmul(
            self.weights.unsqueeze(1).transpose(-2, -1),
            x.squeeze(-2).squeeze(0),
        )
        return x


class NodeHyperLocallyConnected(nn.Module):
    """Hyper Local linear layer, i.e. Conv1dLocal() with filter size 1 which parameters are learned
    from another netwokr:

            y = LocallyConnected_{params}(x),
            where params = h(G)

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    VALID_HYPER = [
        "mlp",
        "gnn",
        "invariant",
        "per_graph",
        "deep_set",
    ]

    def __init__(
        self,
        num_linear,
        input_features,
        output_features,
        hyper,
        n_ens=1,
        bias=True,
        hyper_hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.n_ens = n_ens
        self.hyper = hyper
        self.G = None

        assert (
            hyper in self.VALID_HYPER
        ), f"hyper hparam not a valid option - choices: {self.VALID_HYPER}"

        if hyper == "invariant":
            hyper_type = HyperInvariant
        elif hyper == "mlp":
            hyper_type = HyperMLP
        elif hyper == "per_graph":
            hyper_type = HyperInvariantPerGraph
        elif hyper == "deep_set":
            hyper_type = DeepSet

        self.hyper_layer = hyper_type(
            n_ens=n_ens,
            num_linear=num_linear,
            input_features=input_features,
            output_features=output_features,
            bias=bias,
            hidden_dims=hyper_hidden_dims,
        )

    def forward(self, x: torch.Tensor):
        # [n_ens, n, d, 1, m2] = [n_ens, n, d, 1, m1] @ [n_ens, 1, d, m1, m2]
        G = self.G
        weights, biases = self.hyper_layer(G.to(x))
        x = torch.matmul(x, weights.unsqueeze(1))
        if biases is not None:
            # [n, d, m2] += [d, m2]
            x += biases.unsqueeze(-2).unsqueeze(1)
        return x


class MLP(nn.Module):
    def __init__(self, dims, bias=True):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(dims) - 1):
            if i > 0:
                self.net.append(nn.ELU())
            self.net.append(nn.Linear(dims[i], dims[i + 1], bias=bias))

    def forward(self, x):
        return self.net(x)


class DeepSet(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_features,
        output_features,
        bias=True,
        embedding_size: Optional[int] = None,
        phi_dims: Optional[list] = None,
        f_dims: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        if embedding_size is None:
            embedding_size = 16
        if phi_dims is None:
            phi_dims = [64, 64]
        if f_dims is None:
            f_dims = [64, 64]
        self.embedding_size = embedding_size
        self.phi_dims = phi_dims
        self.f_dims = f_dims
        self.node_embedding = nn.Parameter(torch.Tensor(embedding_size, num_nodes))
        self.phi_net = MLP([embedding_size + input_features, *phi_dims, embedding_size], bias=bias)
        self.f_net = MLP([embedding_size, *f_dims, output_features], bias=bias)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x: [n_ens, batch, d, 1, d]
        # [n_ens, n, d, 1, m2] = [n_ens, n, d, 1, m1] @ [n_ens, 1, d, m1, m2]
        del G
        x = self.phi_net(torch.cat(self.node_embeddings, x))
        # [n_ens, batch, d, 1, emb]
        x = torch.sum(x, dim=-2)

        return x


class HyperMLP(nn.Module):
    """Hypernetwork that takes in a graph (represented as an adjacency matrix) and outputs weights
    and biases for a linear layer over each node."""

    def __init__(
        self,
        num_linear,
        input_features,
        output_features,
        bias=True,
        hidden_dims: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        if hidden_dims is None:
            # hidden_dims = [64, 64, 64]
            # hidden_dims = [64, 64]
            hidden_dims = [1024, 512, 128, 64]
            # hidden_dims = [1024, 1024, 1024, 64]
        self.dims = hidden_dims
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias

        self.w_features = self.num_linear * self.input_features * self.output_features
        self.b_features = self.num_linear * self.output_features
        self.total_features = self.w_features
        if self.bias:
            self.total_features += self.b_features
        full_dims = [num_linear**2, *self.dims, self.total_features]
        self.net = nn.Sequential()
        for i in range(len(full_dims) - 1):
            if i > 0:
                self.net.append(nn.ELU())
            self.net.append(nn.Linear(full_dims[i], full_dims[i + 1]))

    def forward(self, x):
        # input = G ~ A [n_ens x d x d]
        # Want: output = |params|
        # params = h(G)
        n_ens = x.shape[0]
        x = x.reshape(n_ens, -1)
        x = self.net(x)
        x_w = x[:, : self.w_features].reshape(
            n_ens, self.num_linear, self.input_features, self.output_features
        )
        x_b = None
        if self.bias:
            x_b = x[:, self.w_features :].reshape(n_ens, self.num_linear, self.output_features)
        return x_w, x_b


class HyperAnalyticLinear(LocallyConnected):
    """Analytic linear hyper-net module.

    Locally connected but directly returns weights
    """

    def __init__(
        self,
        n_ens,
        num_linear,
        input_features,
    ):
        super(LocallyConnected, self).__init__()
        self.n_ens = n_ens
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = input_features
        self.beta = 0.01  # per-node-GFN = 0.01

        # self.weight = nn.Parameter(
        #    torch.FloatTensor(n_ens, num_linear, num_linear)
        # )
        self.register_parameter("bias", None)

        # self.reset_parameters()

    def analytic_linear(self, x, dx, G):
        Gt = torch.transpose(G.to(x), -2, -1).unsqueeze(1)
        x_masked = Gt * x
        A_est = []
        for p in range(self.num_linear):
            w_est = torch.linalg.solve(
                (torch.transpose(x_masked[:, :, p, :], -2, -1) @ x_masked[:, :, p, :])
                + self.beta * torch.eye(self.num_linear).unsqueeze(0).type_as(x_masked),
                torch.transpose(x_masked[:, :, p, :], -2, -1) @ dx[:, :, p],
            )
            A_est.append(w_est)
        A_est = torch.cat(A_est, dim=2)
        return A_est

    def forward(self, x, dx, G):
        self.weights = self.analytic_linear(x, dx, G).to(x)
        return self.weights


class HyperInvariantPerGraph(LocallyConnected):
    """Invariant hyper-net module per graph.

    Locally connected but directly returns weights
    """

    def __init__(self, n_ens, num_linear, input_features, output_features, bias=True, **kwargs):
        super(LocallyConnected, self).__init__()
        self.n_ens = n_ens
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(n_ens, num_linear, input_features, output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_ens, num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input):
        return self.weight, self.bias


class HyperInvariant(LocallyConnected):
    """Invariant hyper-net module.

    Locally connected but directly returns weights
    """

    def __init__(self, num_linear, input_features, output_features, bias=True, **kwargs):
        super().__init__(num_linear, input_features, output_features, bias)

    def forward(self, input):
        return self.weight.unsqueeze(0), self.bias.unsqueeze(0)
