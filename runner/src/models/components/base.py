"""Implements dynamics models that support interventions on a known and prespecified set of
targets."""

import functools
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

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

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear,
            self.input_features,
            self.output_features,
            self.bias is not None,
        )


class Intervenable(nn.Module):
    """Models implementing intervenable are useful for learning in the experimental setting.

    This should represent interventions on a preexisting set of possible targets.
    """

    def __init__(self, targets=None):
        super().__init__()
        self.targets = targets
        self.current_target = None

    # def do(self, target, value=0.0):
    #    raise NotImplementedError

    def get_linear_structure(self):
        """Gets the linear approximation of the structure coefficients.

        May not be applicable for all models
        """
        raise NotImplementedError

    def get_structure(self) -> np.ndarray:
        """Extracts a single summary structure from the model."""
        raise NotImplementedError

    def get_structures(self, n_structures: int) -> np.ndarray:
        """Some models can provide empirical distributions over structures, this function samples a
        number of structures from the model."""
        raise NotImplementedError

    def set_target(self, target):
        if self.targets is not None and not np.isin(target, self.targets):
            raise ValueError("Bad Target selected {target}")
        self.current_target = target

    def l1_reg(self):
        raise NotImplementedError

    def l2_reg(self):
        raise NotImplementedError


def _parse_activation(activation):
    if activation == "softplus":
        return nn.Softplus
    if activation == "sigmoid":
        return nn.Sigmoid
    if activation == "tanh":
        return nn.Tanh
    if activation == "relu":
        return nn.ReLU
    if activation == "lrelu":
        return nn.LeakyReLU
    if activation == "elu":
        return nn.ELU
    if issubclass(activation, nn.Module):
        return activation
    raise ValueError(f"Urecognized activation function {activation}")


class MLP(Intervenable):
    """Basic MLP drift is that supports perfect interventions key piece is n_inputs is always equal
    to n_outputs."""

    def __init__(
        self,
        n_inputs,
        n_layers=3,
        n_hidden=64,
        activation="softplus",
        time_invariant=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = _parse_activation(activation)
        self.time_invariant = time_invariant
        self.model = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            self.activation(),
            *([nn.Linear(n_hidden, n_hidden), self.activation()] * (self.n_layers - 2)),
            nn.Linear(n_hidden, n_inputs),
        )

    def _get_linear_weights(self, absolute_weights=False):
        """Pretend the network is linear and find that weight matrix.

        This is used in Aliee et al.
        """
        m = self.model
        weights = [m[2 * i].weight for i in range(self.n_layers)[::-1]]
        if absolute_weights:
            weights = [torch.abs(w) for w in weights]
        return functools.reduce(lambda x, y: x @ y, weights)

    def get_linear_structure(self):
        return self._get_linear_weights().cpu().detach().numpy()

    def get_structure(self):
        """Score based on the absolute value of the coefficient."""
        # pretend there are no non-linearities?
        weight_matrix = self._get_linear_weights()
        return np.abs(weight_matrix.cpu().detach().numpy())

    def l1_reg(self, absolute_weights=False):
        weights = self._get_linear_weights(absolute_weights)
        return torch.mean(torch.abs(weights))

    def l2_reg(self, absolute_weights=False):
        weights = self._get_linear_weights(absolute_weights)
        return torch.mean(torch.abs(weights))

    def grn_reg(self, grn, absolute_weights=False):
        weights = self._get_linear_weights(absolute_weights)
        return torch.sum(torch.abs(weights * (1 - grn)))

    def forward(self, t, x, target=None):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        out = self.model(x)
        if target is not None:
            # out [Batch x [Dynamics, Conditions]]
            out[:, target] = 0.0
        if self.current_target is not None:
            out[:, self.current_target] = 0.0
        return out


class DeepEnsDibsLayer(nn.Module):
    """
    DEPRECATED:
        Current bugs linear is not correct for x, g
        Don't want bias
    Deep Ensemble for distribution over function space.
        - incorporating DiBS framework for distributions over
        graphs + functions (v.s. graphs + parameters)
    """

    def __init__(self, n_inputs, n_outputs, k_hidden, dropout_flag=False, bias=True):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.k_hidden = k_hidden
        self.sample_once_flag = True
        self.with_dropout = False

        # define network weights
        self.w = Parameter(torch.empty((self.k_hidden, self.n_inputs)))
        self.v = Parameter(torch.empty((self.k_hidden, self.n_outputs)))

        self.weight = torch.empty((self.n_outputs, self.n_inputs), requires_grad=False)

        # define network biases
        if bias:
            self.bias = Parameter(torch.empty(self.n_outputs))

            if dropout_flag:
                self.p_bias = Parameter(torch.empty(self.n_outputs))
                self.sampled_p_bias = Parameter(torch.empty(self.n_outputs))
        else:
            self.bias = None

        # define dropout and dropout parameters
        if dropout_flag:
            self.with_dropout = dropout_flag
            self.p_weight = Parameter(torch.empty((self.n_outputs, self.n_inputs)))
            self.sampled_p_weight = Parameter(torch.empty((self.n_outputs, self.n_inputs)))

        self.reset_parameters()

    def update_p(self):
        with torch.no_grad():
            self.p_weight.copy_(self.sampled_p_weight.detach().clone())
            self.p_bias.copy_(self.sampled_p_bias.detach().clone())

    def get_structure(self, alpha=1.0):
        Z = torch.matmul(self.w.t(), self.v).t()
        G = torch.sigmoid(alpha * Z)
        return G

    def forward(self, input, alpha=1.0):
        if self.with_dropout:
            if self.sample_once_flag:
                self.p = torch.sigmoid(F.linear(input, self.p_weight, self.p_bias))
                self.mask = torch.bernoulli(1 - self.p)
                self.sample_once_flag = False
                with torch.no_grad():
                    self.sampled_p_weight.copy_(self.p_weight.detach().clone())
                    self.sampled_p_bias.copy_(self.p_bias.detach().clone())
            else:
                raise NotImplementedError()
        G = self.get_structure(alpha)
        print(input.shape, G.shape, self.bias.shape)
        out = F.linear(input, G, self.bias)
        print(out.shape)
        if self.with_dropout:
            out = self.mask * out
        return out, G

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.with_dropout:
            torch.nn.init.constant_(self.p_weight, -0.5)
        # torch.nn.init.kaiming_uniform_(self.p_weight, a=math.sqrt(1))
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
            if self.with_dropout:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.p_weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.p_bias, -bound, bound)


class BayesLinear(nn.Module):
    """A Bayes linear layer for SVI-MLP (Bayes-by-backprop MLP).

    Based off code by Wilson and Izmailov 2020 'Bayesion deep learning and a probabilistic
    perspective'.
    """

    def __init__(self, n_inputs, n_outputs, init_log_var, bias=True, sparse=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.init_log_var = init_log_var
        self.sparse = sparse
        self.sample_once_flag = True
        self.pre_train = True
        self.count = 0

        self.weight = Parameter(torch.empty((self.n_outputs, self.n_inputs)))
        self.weight_isp_std = Parameter(torch.empty((self.n_outputs, self.n_inputs)))
        self.sampled_weights = Parameter(torch.empty((self.n_outputs, self.n_inputs)))
        if bias:
            self.bias_mean = Parameter(torch.empty(self.n_outputs))
            self.bias_isp_std = Parameter(torch.empty(self.n_outputs))
            self.sampled_biases = Parameter(torch.empty(self.n_outputs))
            self.with_bias = True
        else:
            self.with_bias = False

        self.reset_parameters()
        self.eps = 1e-8

    def sample_weights(self, n_samples: int = 0) -> Tensor:
        if n_samples > 0:
            w_sigma = F.softplus(self.weight_isp_std) + self.eps
            w = self.weight + torch.randn((n_samples, *self.weight.shape)) * w_sigma
        else:
            w_sigma = F.softplus(self.weight_isp_std) + self.eps
            w = self.weight + torch.randn_like(self.weight) * w_sigma
        return w

    def sample(self):
        w_sigma = F.softplus(self.weight_isp_std) + self.eps
        if self.sparse:
            e = 0.5 - torch.rand_like(self.weight)
            w = self.weight - torch.sign(e) * w_sigma * torch.log(1.0 - 2.0 * torch.abs(e))
        else:
            w = self.weight + torch.randn_like(self.weight) * w_sigma
        if self.with_bias:
            b_sigma = F.softplus(self.bias_isp_std) + self.eps
            if self.sparse:
                e = 0.5 - torch.rand_like(self.bias_mean)
                b = self.bias_mean - torch.sign(e) * b_sigma * torch.log(1.0 - 2.0 * torch.abs(e))
            else:
                b = self.bias_mean + torch.randn_like(self.bias_mean) * b_sigma
        else:
            b = None
        return w, b

    def update_means(self):
        with torch.no_grad():
            self.weight.copy_(self.sampled_weights.detach().clone())
            self.bias_mean.copy_(self.sampled_biases.detach().clone())

    def forward(self, input: Tensor):
        if self.pre_train:
            return F.linear(input, self.sampled_weights, self.sampled_biases)
        else:
            if self.sample_once_flag:
                w, b = self.sample()
                with torch.no_grad():
                    self.sampled_weights.copy_(w.detach().clone())
                    self.sampled_biases.copy_(b.detach().clone())
                if self.count >= 1:
                    self.sample_once_flag = False
                    return F.linear(input, w, b)
                else:
                    self.count += 1
                    return F.linear(input, w, b)
            else:
                return F.linear(input, self.sampled_weights, self.sampled_biases)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.sampled_weights, a=math.sqrt(5))
        self.weight_isp_std.data.fill_(self.init_log_var)
        # torch.nn.init.normal_(self.weight_isp_std, mean=self.init_log_var, std=0.1)
        if self.with_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_mean, -bound, bound)
            torch.nn.init.uniform_(self.sampled_biases, -bound, bound)
            self.bias_isp_std.data.fill_(self.init_log_var)
            # torch.nn.init.normal_(self.bias_isp_std, mean=self.init_log_var, std=0.1)
        print(
            "\ncheck init w_mean:",
            self.weight,
            "\ncheck init w_std:",
            F.softplus(self.weight_isp_std),
        )

    def _get_kl(self, param_mean, param_isp_std, prior_log_sigma):
        sigma = F.softplus(param_isp_std) + self.eps
        if self.sparse:
            kl = torch.sum(sigma * torch.exp(-torch.abs(param_mean) / sigma))
            kl = torch.sum((kl + torch.abs(param_mean)) / math.exp(prior_log_sigma))
            kl += prior_log_sigma - torch.sum(torch.log(sigma))
        else:
            kl = torch.sum(
                prior_log_sigma
                - torch.log(sigma)
                + 0.5 * (sigma**2) / (math.exp(prior_log_sigma * 2))
            )
            # kl += 0.5 * torch.sum(param_mean ** 2) / math.exp(prior_log_sigma * 2)
        return kl

    def kl_with_prior(self, prior_log_sigma, t=1):
        w_kl = self._get_kl(self.weight, self.weight_isp_std, prior_log_sigma)
        if self.with_bias:
            b_kl = self._get_kl(self.bias_mean, self.bias_isp_std, prior_log_sigma)
            return w_kl + b_kl
        else:
            return w_kl


class DibsLayer(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        k_hidden,
        init_log_var,
        alpha=0.1,
        beta=0.5,
        bias=True,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.k_hidden = k_hidden
        self.init_log_var = init_log_var
        self.alpha = alpha
        self.beta = beta
        self.sample_once_flag = True
        self.pre_train = True
        self.iter_num = 0
        self.count = 0

        self.w = Parameter(torch.empty((self.k_hidden, self.n_inputs)))
        self.w_isp_std = Parameter(torch.empty((self.k_hidden, self.n_inputs)))
        self.sampled_w = Parameter(torch.empty((self.k_hidden, self.n_inputs)))

        self.v = Parameter(torch.empty((self.k_hidden, self.n_outputs)))
        self.v_isp_std = Parameter(torch.empty((self.k_hidden, self.n_outputs)))
        self.sampled_v = Parameter(torch.empty((self.k_hidden, self.n_outputs)))

        if bias:
            self.b = Parameter(torch.empty(self.n_outputs))
            self.b_isp_std = Parameter(torch.empty(self.n_outputs))
            self.sampled_b = Parameter(torch.empty(self.n_outputs))

            self.with_bias = True
        else:
            self.with_bias = False

        # Latent graph Z
        self.weight = torch.empty((self.n_outputs, self.n_inputs), requires_grad=False)
        self.weight_sampled = torch.empty((self.n_outputs, self.n_inputs), requires_grad=False)

        self.reset_parameters()
        self.eps = 1e-8

    def get_graph(self, d, t=1, get_structure_flag=False):
        if t > self.iter_num:
            self.iter_num = t
        if get_structure_flag:
            W, V, _ = self.sample()
            self.weight = torch.matmul(W.t(), V).t()
        else:
            self.weight = torch.matmul(self.w.t(), self.v).t()
        fc1_weight = self.weight.view(d, -1, d)  # [j, m1, i]
        # Z = torch.sum(fc1_weight**2, dim=1).pow(0.5)  # [i, j]
        # Z = Z - torch.mean(Z)
        Z = torch.mean(fc1_weight, dim=1)  # [i, j]
        self.alpha_t = self.alpha * self.iter_num
        p_G_Z = torch.sigmoid(self.alpha_t * Z)
        # print(Z[1], p_G_Z[1])
        return p_G_Z

    def h_acyclic(self, t):
        d = self.n_inputs
        G = self.get_graph(d, t)
        arg = torch.matrix_power(torch.eye(d) + 1 / d * G, d)
        return torch.trace(arg) - d

    def sample(self):
        w_sigma, v_sigma = (
            F.softplus(self.w_isp_std) + self.eps,
            F.softplus(self.v_isp_std) + self.eps,
        )
        W = self.w + torch.randn_like(self.w) * w_sigma
        V = self.v + torch.randn_like(self.v) * v_sigma
        if self.with_bias:
            b_sigma = F.softplus(self.b_isp_std) + self.eps
            b = self.b + torch.randn_like(self.b) * b_sigma
        else:
            b = None
        return W, V, b

    def update_means(self):
        # self.w = self.sampled_w
        # self.v = self.sampled_v
        # self.b = self.sampled_b
        # self.weight = self.weight_sampled
        with torch.no_grad():
            self.w.copy_(self.sampled_w.detach().clone())
            self.v.copy_(self.sampled_v.detach().clone())
            if self.with_bias:
                self.b.copy_(self.sampled_b.detach().clone())
            self.weight.copy_(self.sampled_weights.detach().clone())

    def forward(self, input: Tensor):
        if self.pre_train:
            if self.with_bias:
                self.sampled_weights = torch.matmul(self.sampled_w.t(), self.sampled_v).t()
                return F.linear(input, self.sampled_weights, bias=self.sampled_b)
            else:
                self.sampled_weights = torch.matmul(self.sampled_w.t(), self.sampled_v).t()
                return F.linear(input, self.sampled_weights, bias=None)
        else:
            if self.sample_once_flag:
                W, V, b_samp = self.sample()

                with torch.no_grad():
                    self.sampled_w.copy_(W.detach().clone())
                    self.sampled_v.copy_(V.detach().clone())
                    if self.with_bias:
                        self.sampled_b.copy_(b_samp.detach().clone())

                self.sample_once_flag = False
                if self.with_bias:
                    self.weight = torch.matmul(W.t(), V).t()
                    return F.linear(input, self.weight, bias=b_samp)
                else:
                    self.weight = torch.matmul(W.t(), V).t()
                    return F.linear(input, self.weight, bias=None)

            else:
                if self.with_bias:
                    self.sampled_weights = torch.matmul(self.sampled_w.t(), self.sampled_v).t()
                    return F.linear(input, self.sampled_weights, bias=self.sampled_b)
                else:
                    self.sampled_weights = torch.matmul(self.sampled_w.t(), self.sampled_v).t()
                    return F.linear(input, self.sampled_weights, bias=None)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.sampled_w, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.sampled_v, a=math.sqrt(5))
        self.w_isp_std.data.fill_(self.init_log_var)
        self.v_isp_std.data.fill_(self.init_log_var)
        if self.with_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.v)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.b, -bound, bound)
            torch.nn.init.uniform_(self.sampled_b, -bound, bound)
            self.b_isp_std.data.fill_(self.init_log_var)
        print(
            "\ncheck init u_mean:",
            self.v,
            "\ncheck init u_std:",
            F.softplus(self.v_isp_std),
        )

    def _get_kl(self, param_mean, sigma, prior_log_sigma):
        kl = torch.sum(
            prior_log_sigma - torch.log(sigma) + 0.5 * (sigma**2) / (math.exp(prior_log_sigma * 2))
        )
        kl += 0.5 * torch.sum(param_mean**2) / math.exp(prior_log_sigma * 2)
        return kl

    def kl_with_prior(self, prior_log_sigma, t=1):
        self.weight = torch.matmul(self.w.t(), self.v).t()
        w_sigma, v_sigma = (
            F.softplus(self.w_isp_std) + self.eps,
            F.softplus(self.v_isp_std) + self.eps,
        )
        weight_std = torch.matmul(w_sigma.t(), v_sigma).t()
        z_kl = self._get_kl(self.weight, weight_std, prior_log_sigma) - self.beta * self.h_acyclic(
            t
        )
        if self.with_bias:
            b_sigma = F.softplus(self.b_isp_std) + self.eps
            b_kl = self._get_kl(self.b, b_sigma, prior_log_sigma)
        else:
            b_kl = 0.0
        return z_kl + b_kl


class BayesMLP(Intervenable):
    """Bayes MLP drift is that supports perfect interventions key piece is n_inputs is always equal
    to n_outputs."""

    def __init__(
        self,
        n_inputs,
        n_layers=3,
        n_hidden=64,
        activation="softplus",
        time_invariant=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activation = _parse_activation(activation)
        self.time_invariant = time_invariant
        self.model = nn.Sequential(
            BayesLinear(n_inputs, n_hidden),
            self.activation(),
            *([BayesLinear(n_hidden, n_hidden), self.activation()] * (self.n_layers - 2)),
            BayesLinear(n_hidden, n_inputs),
        )

    def _get_linear_weights(self, absolute_weights=False):
        """Pretend the network is linear and find that weight matrix.

        This is used in Aliee et al.
        """
        m = self.model
        weights = [m[2 * i].weight for i in range(self.n_layers)[::-1]]
        if absolute_weights:
            weights = [torch.abs(w) for w in weights]
        return functools.reduce(lambda x, y: x @ y, weights)

    def get_linear_structure(self):
        return self._get_linear_weights().cpu().detach().numpy()

    def get_structure(self):
        """Score based on the absolute value of the coefficient."""
        # pretend there are no non-linearities?
        weight_matrix = self._get_linear_weights()
        return np.abs(weight_matrix.cpu().detach().numpy())

    def l1_reg(self, absolute_weights=False):
        weights = self._get_linear_weights(absolute_weights)
        return torch.mean(torch.abs(weights))

    def l2_reg(self, absolute_weights=False):
        weights = self._get_linear_weights(absolute_weights)
        return torch.mean(torch.abs(weights))

    def forward(self, t, x, target=None):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        out = self.model(x)
        if target is not None:
            # out [Batch x [Dynamics, Conditions]]
            out[:, target] = 0.0
        if self.current_target is not None:
            out[:, self.current_target] = 0.0
        return out


class Linear(Intervenable):
    def __init__(self, n_inputs, targets=None, time_invariant=True):
        super().__init__(targets)
        self.time_invariant = time_invariant
        self.n_inputs = n_inputs
        self.model = nn.Sequential(nn.Linear(n_inputs, n_inputs, bias=False))

    def get_linear_structure(self):
        return self.model[0].weight.cpu().detach().numpy()

    def get_structure(self):
        """Score based on the absolute value of the coefficient."""
        return np.abs(self.model[0].weight.cpu().detach().numpy())

    def forward(self, t, x, target=None):
        if not self.time_invariant:
            x = torch.cat((x, t), dim=-1)
        out = self.model(x)
        if target is not None:
            out[:, target] = 0.0
        if self.current_target is not None:
            out[:, self.current_target] = 0.0
        return out
