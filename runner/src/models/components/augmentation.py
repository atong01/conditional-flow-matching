import torch
from torch import nn


class Regularizer(nn.Module):
    def __init__(self):
        pass


def _batch_root_mean_squared(tensor):
    tensor = tensor.view(tensor.shape[0], -1)
    return torch.norm(tensor, p=2, dim=1) / tensor.shape[1] ** 0.5


class RegularizationFunc(nn.Module):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        """Outputs a batch of scaler regularizations."""
        raise NotImplementedError


class L1Reg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        return torch.mean(torch.abs(dx), dim=1)


class L2Reg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        return _batch_root_mean_squared(dx)


class SquaredL2Reg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        to_return = dx.view(dx.shape[0], -1)
        return torch.pow(torch.norm(to_return, p=2, dim=1), 2)


def _get_minibatch_jacobian(y, x, create_graph=True):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    # assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j]),
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        jac.append(torch.unsqueeze(dy_j_dx, -1))
    jac = torch.cat(jac, -1)
    return jac


class JacobianFrobeniusReg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        jac = _get_minibatch_jacobian(dx, x)
        context.jac = jac
        return _batch_root_mean_squared(jac)


class JacobianDiagFrobeniusReg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        diagonal = jac.view(jac.shape[0], -1)[
            :, :: jac.shape[1]
        ]  # assumes jac is minibatch square, ie. (N, M, M).
        return _batch_root_mean_squared(diagonal)


class JacobianOffDiagFrobeniusReg(RegularizationFunc):
    def forward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        diagonal = jac.view(jac.shape[0], -1)[
            :, :: jac.shape[1]
        ]  # assumes jac is minibatch square, ie. (N, M, M).
        ss_offdiag = torch.sum(jac.view(jac.shape[0], -1) ** 2, dim=1) - torch.sum(
            diagonal**2, dim=1
        )
        ms_offdiag = ss_offdiag / (diagonal.shape[1] * (diagonal.shape[1] - 1))
        return ms_offdiag


def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd."""
    trJ = 0.0
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[
            0
        ][:, i]
    return trJ


def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd."""
    noise = torch.randn_like(x_in)
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum("bi,bi->b", jvp, noise)
    return trJ


class CNFReg(RegularizationFunc):
    def __init__(self, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.trace_estimator = autograd_trace
        if trace_estimator == "hutch":
            self.trace_estimator = hutch_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, dx, context):
        # TODO we could check if jac is in the context to speed up
        return -self.trace_estimator(dx, x) + 0 * x


class AugmentationModule(nn.Module):
    """Class orchestrating augmentations.

    Also establishes order.
    """

    def __init__(
        self,
        cnf_estimator: str = None,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        squared_l2_reg: float = 0.0,
        jacobian_frobenius_reg: float = 0.0,
        jacobian_diag_frobenius_reg: float = 0.0,
        jacobian_off_diag_frobenius_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnf_estimator = cnf_estimator
        names = []
        coeffs = []
        regs = []
        if cnf_estimator == "exact":
            names.append("log_prob")
            coeffs.append(1)
            regs.append(CNFReg(None, noise_dist=None))
        if l1_reg > 0.0:
            names.append("L1")
            coeffs.append(l1_reg)
            regs.append(L1Reg())
        if l2_reg > 0.0:
            names.append("L2")
            coeffs.append(l2_reg)
            regs.append(L2Reg())
        if squared_l2_reg > 0.0:
            names.append("squared_L2")
            coeffs.append(squared_l2_reg)
            regs.append(SquaredL2Reg())
        if jacobian_frobenius_reg > 0.0:
            names.append("jacobian_frobenius")
            coeffs.append(jacobian_frobenius_reg)
            regs.append(JacobianFrobeniusReg())
        if jacobian_diag_frobenius_reg > 0.0:
            names.append("jacobian_diag_frobenius")
            coeffs.append(jacobian_diag_frobenius_reg)
            regs.append(JacobianDiagFrobeniusReg())
        if jacobian_off_diag_frobenius_reg > 0.0:
            names.append("jacobian_off_diag_frobenius")
            coeffs.append(jacobian_off_diag_frobenius_reg)
            regs.append(JacobianOffDiagFrobeniusReg())
        self.names = names
        self.coeffs = torch.tensor(coeffs)
        self.regs = torch.nn.ModuleList(regs)
        assert len(self.coeffs) == len(self.regs)
        self.aug_dims = len(self.coeffs)
        self.augmenter = Augmenter(augment_idx=1, augment_dims=self.aug_dims)

    def forward(self, x):
        """Separates and adds together losses."""
        # if x.dim() > 2:
        # augmentation is broken, return regs = 0 for now
        #   reg = torch.zeros(1).type_as(x)
        #    return reg, x
        if self.cnf_estimator is None:
            if self.aug_dims == 0:
                reg = torch.zeros(1).type_as(x)
            else:
                aug, x = x[:, : self.aug_dims], x[:, self.aug_dims :]
                reg = aug * self.coeffs
            return reg, x
        delta_logprob, aug, x = x[:, :1], x[:, 1 : self.aug_dims], x[:, self.aug_dims :]
        reg = aug * self.coeffs[1:].to(aug)
        if self.aug_dims == 1:
            reg = torch.zeros(1).type_as(x)
        return delta_logprob, reg, x


class Augmenter(nn.Module):
    """Augmentation class.

    Can handle several types of augmentation strategies for Neural DEs.
    :param augment_dims: number of augmented dimensions to initialize
    :type augment_dims: int
    :param augment_idx: index of dimension to augment
    :type augment_idx: int
    :param augment_func: nn.Module applied to the input datasets of dimension `d` to determine the
        augmented initial condition of dimension `d + a`. `a` is defined implicitly in
        `augment_func` e.g. augment_func=nn.Linear(2, 5) augments a 2 dimensional input with 3
        additional dimensions.
    :type augment_func: nn.Module
    :param order: whether to augment before datasets [augmentation, x] or after [x, augmentation]
        along dimension `augment_idx`. Options: ('first', 'last')
    :type order: str
    """

    def __init__(
        self,
        augment_idx: int = 1,
        augment_dims: int = 5,
        augment_func=None,
        order="first",
    ):
        super().__init__()
        self.augment_dims, self.augment_idx, self.augment_func = (
            augment_dims,
            augment_idx,
            augment_func,
        )
        self.order = order

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        if not self.augment_func:
            x = x.reshape(x.shape[0], -1)
            new_dims = list(x.shape)
            new_dims[self.augment_idx] = self.augment_dims

            # if-else check for augmentation order
            if self.order == "first":
                x = torch.cat([torch.zeros(new_dims).to(x), x], self.augment_idx)
            else:
                x = torch.cat([x, torch.zeros(new_dims).to(x)], self.augment_idx)
        else:
            # if-else check for augmentation order
            if self.order == "first":
                x = torch.cat([self.augment_func(x).to(x), x], self.augment_idx)
            else:
                x = torch.cat([x, self.augment_func(x).to(x)], self.augment_idx)
        return x, ts


class AugmentedVectorField(nn.Module):
    """NeuralODE but augmented state.

    Preprends Augmentations to state for easy integration over time
    """

    def __init__(self, net, augmentation_list: nn.ModuleList, dim):
        super().__init__()
        self.net = net
        self.dim = dim
        self.augmentation_list = augmentation_list

    def forward(self, t, state, augmented_input=True, *args, **kwargs):
        n_aug = len(self.augmentation_list)

        class SharedContext:
            pass

        with torch.set_grad_enabled(True):
            # first dimensions reserved for augmentations
            x = state
            if augmented_input:
                x = x[:, n_aug:].requires_grad_(True)

            # the neural network will handle the data-dynamics here
            if isinstance(self.dim, int):
                dx = self.net(t, x.reshape(-1, self.dim))
            else:
                dx = self.net(t, x.reshape(-1, *self.dim))
            if n_aug == 0:
                return dx
            dx = dx.reshape(dx.shape[0], -1)
            # x_out = x_out.squeeze(dim=1)

            augs = [aug_fn(t, x, dx, SharedContext) for aug_fn in self.augmentation_list]
            augs = torch.stack(augs, dim=1)
        # `+ 0*state` has the only purpose of connecting state[:, 0] to autograd graph
        return torch.cat([augs, dx], 1) + (0 * state if augmented_input else 0)


class CNF(AugmentedVectorField):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        cnf_reg = CNFReg(trace_estimator, noise_dist)
        super().__init__(net, [cnf_reg])


class Old_CNF(nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(
                True
            )  # first dimension reserved to divergence propagation
            # the neural network will handle the data-dynamics here
            x_out = self.net(t, x_in)
            x_out = x_out.squeeze(dim=1)
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph


class Sequential(nn.Sequential):
    """A sequential module which handles multiple inputs."""

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


if __name__ == "__main__":
    # Test Shapes
    class SharedContext:
        pass

    for reg in [
        L1Reg,
        L2Reg,
        SquaredL2Reg,
        JacobianFrobeniusReg,
        JacobianDiagFrobeniusReg,
        JacobianOffDiagFrobeniusReg,
    ]:
        x = torch.ones(2, 3).requires_grad_(True)
        dx = x * 2
        out = reg().forward(torch.ones(1), x, dx, SharedContext)
        assert out.dim() == 1
        assert out.shape[0] == 2
