import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    BayesLinear,
    DeepEnsDibsLayer,
    DibsLayer,
    Intervenable,
    LocallyConnected,
)
from .hyper_nets import HyperLocallyConnected


class DeepEnsEmbedMLPODEF(Intervenable):
    pass


class MLPODEF(Intervenable):
    """Define an MLP ODE function according to Neural Graphical Models definition."""

    def __init__(self, dims, GL_reg=0.01, bias=True, time_invariant=True):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter

        self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        """Old way of implementing time_invariant.

        if time_invariant:
            self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        else:
            self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)
        """

        # fc2: local linear layers
        layers = []
        for i in range(len(dims) - 2):
            layers.append(
                LocallyConnected(
                    dims[0],
                    dims[i + 1] + (0 if self.time_invariant else 1),
                    dims[i + 2],
                    bias=bias,
                )
            )
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        if not self.time_invariant:
            x = torch.cat((x, t.repeat(*x.shape[:-1], 1)), dim=-1)
        for fc in self.fc2:
            x = fc(self.elu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x = x.unsqueeze(dim=1)  # [n, 1, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def grn_reg(self, grn):
        """
        Args:
            grn: torch.tensor (d x d) 1 if likely edge 0 if not
        """
        fc1_weight = self.fc1.weight  # d * m1, d
        d = fc1_weight.shape[-1]
        fc1_weight = fc1_weight.reshape(d, -1, d)
        fc1_weight = fc1_weight.transpose(0, 1)  # m1, d, d
        return torch.sum(torch.abs(fc1_weight * (1 - grn)))

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def get_structure(self):
        """Score each edge based on the the weight sum."""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


class BayesMLPODEF(Intervenable):
    """Define an Bayes-MLP ODE (via SVI) function according to Neural Graphical Models
    definition."""

    def __init__(
        self,
        dims,
        GL_reg=0.01,
        init_log_var=-5,
        dibs=False,
        k_hidden=1,
        alpha=0.1,
        beta=0.5,
        bias=True,
        time_invariant=True,
        sparse=False,
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter
        self.dibs = dibs

        if time_invariant:
            if self.dibs:
                assert k_hidden <= dims[0], "U,V dimension k larger than amount of nodes!"
                self.fc1 = DibsLayer(
                    dims[0],
                    dims[0] * dims[1],
                    k_hidden=k_hidden,
                    init_log_var=init_log_var,
                    alpha=alpha,
                    beta=beta,
                    bias=bias,
                )
            else:
                self.fc1 = BayesLinear(
                    dims[0],
                    dims[0] * dims[1],
                    init_log_var=init_log_var,
                    bias=bias,
                    sparse=sparse,
                )
        else:
            if self.dibs:
                assert k_hidden <= dims[0], "U,V dimension k larger than amount of nodes!"
                self.fc1 = DibsLayer(
                    dims[0] + 1,
                    dims[0] * dims[1],
                    k_hidden=k_hidden,
                    init_log_var=init_log_var,
                    alpha=alpha,
                    beta=beta,
                    bias=bias,
                )
            else:
                self.fc1 = BayesLinear(
                    dims[0] + 1,
                    dims[0] * dims[1],
                    init_log_var=init_log_var,
                    bias=bias,
                    sparse=sparse,
                )

        # fc2: local linear layers
        layers = []
        for i in range(len(dims) - 2):
            layers.append(LocallyConnected(dims[0], dims[i + 1], dims[i + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.elu = nn.ELU(inplace=True)

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = fc(self.elu(x))  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x = x.unsqueeze(dim=1)  # [n, 1, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.get_structure(
            self.alpha_t
        )  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def compute_kl_term(self, net, prior_log_sigma, t=1):
        kl = 0.0
        for module in net.children():
            if self.dibs:
                if isinstance(module, DibsLayer):
                    kl += module.kl_with_prior(prior_log_sigma, t)
            else:
                if isinstance(module, BayesLinear):
                    kl += module.kl_with_prior(prior_log_sigma, t)
        return kl

    def kl_reg(self, net, prior_log_sigma, t=1):
        return self.compute_kl_term(net, prior_log_sigma, t)

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def get_structure(self, t=1, test_mode: bool = False):
        """Score each edge based on the the weight sum."""
        if self.dibs:
            d = self.dims[0]
            W_list = []
            for _ in range(1000):
                W_tmp = self.fc1.get_graph(d, t, get_structure_flag=True)
                W_tmp = W_tmp.cpu().detach().numpy()  # [i, j]
                W_list.append(W_tmp)
            W = np.mean(np.array(W_list), axis=0)
            if test_mode:
                W_std = np.std(np.array(W_list), axis=0)

        else:
            d = self.dims[0]
            W_list = []
            for _ in range(1000):
                fc1_weight, _ = self.fc1.sample()  # [j * m1, i]
                fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
                W_tmp = fc1_weight.cpu().detach().numpy()  # [i, j]
                W_list.append(W_tmp)
            W = np.mean(np.array(W_list), axis=0)
            W = np.sum(W**2, axis=1) ** (0.5)  # [i, j]
            if test_mode:
                W_std = np.std(np.array(W_list), axis=0)
                W_std = np.sum(W_std**2, axis=1) ** (0.5)  # [i, j]
            # fc1_weight = self.fc1.weight  # [j * m1, i]
            # fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
            # W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
            # W = W.cpu().detach().numpy()  # [i, j]
        if test_mode:
            return W, W_std
        else:
            return W

    def get_structures(self, n_structures: int):
        d = self.dims[0]
        w_samples = self.fc1.sample_weights(n_structures)
        w_samples = w_samples.view(n_structures, d, -1, d)  # [n, j, m1, i]
        W = torch.sum(w_samples**2, dim=2).pow(0.5)  # [n, i, j]
        W = W.cpu().detach().numpy()  # [n, i, j]
        return W

    def reset_parameters(self):
        self.fc1.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


class DeepEnsMLPODEF(Intervenable):
    """Define an DeepEns-MLP ODE function according to Neural Graphical Models definition."""

    def __init__(
        self,
        dims,
        n_ens=25,
        GL_reg=0.01,
        dibs=False,
        k_hidden=1,
        alpha=0.1,
        dropout_flag=False,
        bias=True,
        time_invariant=True,
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.n_ens = n_ens
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter
        self.dibs = dibs
        self.alpha = alpha

        self.fc1_modules = []
        if self.dibs:
            for m in range(self.n_ens):
                if time_invariant:
                    self.fc1_modules.append(
                        DeepEnsDibsLayer(
                            dims[0],
                            dims[0] * dims[1],
                            k_hidden=k_hidden,
                            dropout_flag=True,
                            bias=bias,
                        )
                    )
                else:
                    self.fc1_modules.append(
                        DeepEnsDibsLayer(
                            dims[0] + 1,
                            dims[0] * dims[1],
                            k_hidden=k_hidden,
                            dropout_flag=True,
                            bias=bias,
                        )
                    )
            self.fc1 = nn.ModuleList(self.fc1_modules)
        else:
            for m in range(self.n_ens):
                if time_invariant:
                    self.fc1_modules.append(nn.Linear(dims[0], dims[0] * dims[1], bias=bias))
                else:
                    self.fc1_modules.append(nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias))
            self.fc1 = nn.ModuleList(self.fc1_modules)

        # if time_invariant:
        #    self.fc1 = nn.Linear(dims[0], dims[0] * dims[1], bias=bias)
        # else:
        #    self.fc1 = nn.Linear(dims[0] + 1, dims[0] * dims[1], bias=bias)

        # fc2: local linear layers
        self.fc2_modules, self.elu_modules = nn.ModuleList(), nn.ModuleList()
        for m in range(self.n_ens):
            layers = []
            for i in range(len(dims) - 2):
                layers.append(LocallyConnected(dims[0], dims[i + 1], dims[i + 2], bias=bias))
            # self.fc2 = nn.ModuleList(layers)
            self.fc2_modules.append(nn.ModuleList(layers))
            self.elu = nn.ELU(inplace=True)

    def update_p(self):
        for fc in self.fc1:
            fc.update_p()

    def set_sample_flag(self):
        for fc in self.fc1:
            fc.sample_once_flag = True

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)

        x_tmp = []
        for fc in self.fc1:
            x_tmp.append(fc(x)[0])
        x = torch.stack(x_tmp)
        # x = x.mean(dim=0)
        # x = self.fc1(x)

        x = x.view(self.n_ens, -1, self.dims[0], self.dims[1])  # [n_ens, n, d, m1]
        # TODO broken for > 1 layer
        x_out = []
        m = 0
        for fc2 in self.fc2_modules:
            for fc in fc2:
                x_out.append(fc(F.elu(x[m])))  # [n, d, m2]
            m += 1
        x = torch.stack(x_out)
        x = x.squeeze(dim=3)  # [n_ens, n, d]
        x = x.unsqueeze(dim=2)  # [n_ens, n, 1, d]
        return x

    def l2_reg(self):
        """L2 regularization on all parameters."""
        reg = 0.0
        fc1_weight = self.fc1.weight  # [j * m1, i], m1 = number of hidden nodes
        reg += torch.sum(fc1_weight**2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight**2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.weight))

    def DeepEns_prior(self, net, prior_var):
        reg = 0.0
        with torch.no_grad():
            for fc in net.fc1:
                for p in fc.parameters():
                    reg += torch.norm(p) ** 2 / (2 * prior_var)
        return reg

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        fc1_weight = self.fc1.weight.view(self.dims[0], -1, self.dims[0])  # [j, m1, i]
        weights = torch.sum(fc1_weight**2, dim=1).pow(gamma).data  # [i, j]
        return weights

    def get_structure(self, t=1, test_mode: bool = False):
        """Score each edge based on the the weight sum."""
        if self.dibs:
            d = self.dims[0]
            G_list = []
            for fc in self.fc1:
                fc1_weight = torch.matmul(fc.w.t(), fc.v).t()
                fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
                # Z = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
                Z = torch.sum(torch.abs(fc1_weight), dim=1)  # [i, j]
                Z = Z - torch.mean(Z)  # [i, j]
                self.alpha_t = self.alpha * t
                G_list.append(torch.sigmoid(self.alpha_t * Z))
            G = torch.stack(G_list)
            # G_mean = P_G.mean(dim=0)
            # G_std = P_G.std(dim=0).detach().numpy()
            # G_mean = G_mean.cpu().detach().numpy()
            # G = np.heaviside(G_mean - 0.5, 0.5)
            if test_mode:
                G = G.cpu().detach().numpy()
                return G  # , G_mean, G_std
            else:
                return G
        else:
            d = self.dims[0]
            fc1_weight_list = []
            for fc in self.fc1:
                fc1_weight_list.append(fc.weight)  # [j * m1, i]
            fc1_weight = torch.stack(fc1_weight_list)
            fc1_weight = fc1_weight.mean(dim=0)
            # fc1_std = fc1_weight.std(dim=0)
            fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
            W = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
            W = W.cpu().detach().numpy()  # [i, j]
            return W

    def reset_parameters(self):
        for fc in self.fc1:
            fc.reset_parameters()
        for fc in self.fc2:
            fc.reset_parameters()


class EnsembleLayer(nn.Module):
    def __init__(self, n_ens, layer, *args, **kwargs):
        super().__init__()
        self.n_ens = n_ens
        self.ensemble = nn.ModuleList([layer(*args, **kwargs) for _ in range(n_ens)])


class DibsEnsembleLayer(EnsembleLayer):
    def __init__(self, n_ens, alpha, layer, *args, **kwargs):
        super().__init__(n_ens, layer, *args, **kwargs)
        self.alpha = alpha

    def forward(self, x):
        x_tmp = []
        gs = []
        print(x.shape)
        for fc in self.ensemble:
            x_i, g_i = fc(x, alpha=self.alpha)
            x_tmp.append(x_i)
            gs.append(g_i)
        x = torch.stack(x_tmp)
        print(x.shape)
        G = torch.stack(gs)
        return x, G

    def update_p(self):
        for fc in self.ensemble:
            fc.update_p()

    def set_sample_flag(self):
        for fc in self.ensemble:
            fc.sample_once_flag = True

    def get_structures(self):
        return torch.stack([fc.get_structure(self.alpha) for fc in self.ensemble])


class DeepEnsHyperMLPODEF(Intervenable):
    """Define an DeepEns-MLP ODE function to acquire graph structure G and use G in linear pipeline
    via hyper-net architecture.

    - goals: P(params | G)P(G)
    """

    def __init__(
        self,
        dims,
        n_ens=25,
        GL_reg=0.01,
        dibs=False,
        k_hidden=1,
        alpha=0.1,
        dropout_flag=False,
        hyper=None,
        bias=True,
        time_invariant=True,
    ):
        # dims: [number of variables, dimension hidden layers, output dim=1]
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1

        self.dims = dims
        self.n_ens = n_ens
        self.time_invariant = time_invariant
        self.GL_reg = GL_reg  # adaptive lasso parameter
        self.dibs = dibs
        self.alpha = alpha
        self.current_epoch = 0
        fc1_input_dim = dims[0] if time_invariant else dims[0] + 1

        if self.dibs:
            self.fc1 = DibsEnsembleLayer(
                n_ens,
                self.alpha,
                DeepEnsDibsLayer,
                fc1_input_dim,
                dims[0],
                k_hidden=k_hidden,
                dropout_flag=dropout_flag,
                bias=bias,
            )
        else:
            self.fc1 = DibsEnsembleLayer(
                n_ens,
                self.alpha,
                nn.Linear,
                fc1_input_dim,
                dims[0],
                bias=bias,
            )

        # fc2: let params be function of G ~ A
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                HyperLocallyConnected(
                    dims[0],  # num_linear
                    dims[i],  # input_features
                    dims[i + 1],  # output_features
                    n_ens=n_ens,
                    hyper=hyper,
                    bias=bias,
                )
            )
        self.fc2 = nn.ModuleList(layers)

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def set_sample_flag(self):
        self.fc1.set_sample_flag()

    def update_p(self):
        self.fc1.update_p()

    def forward(self, t, x):  # [n, 1, d] -> [n, 1, d]
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        self.fc1.alpha = self.alpha * (self.current_epoch + 1)
        x, G = self.fc1(x)
        x = x.unsqueeze(dim=2)  # [n_ens, n, 1, d]
        for fc in self.fc2:
            x = fc(F.elu(x), G)  # [n_ens, n, d, mi]
        # x.shape [n_ens, n, d, 1]
        return x.transpose(-1, -2)  # x.shape [n_ens, n, 1, d]

    def l2_reg(self):
        """L2 regularization on all parameters."""
        return torch.sum(self.fc1.get_structures() ** 2)

    def l1_reg(self):
        """L1 regularization on input layer parameters."""
        return torch.sum(torch.abs(self.fc1.get_structures()))

    def DeepEns_prior(self, net, prior_var):
        reg = 0.0
        with torch.no_grad():
            for p in net.fc1.parameters():
                reg += torch.norm(p) ** 2 / (2 * prior_var)
        return reg

    def group_weights(self, gamma=0.5):
        """Group lasso weights."""
        Gs = self.fc1.get_structures()
        weights = torch.sum(Gs**2, dim=0).pow(gamma).data  # [i, j]
        return weights

    def get_structure(self, t=1, test_mode: bool = False):
        """Score each edge based on the the weight sum."""
        G = self.fc1.get_structures()
        if test_mode:
            return G.cpu().detach().numpy()
        return G
