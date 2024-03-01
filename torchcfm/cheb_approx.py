import typing as T
import numpy as np
import torch
from scipy.special import ive


def expm_multiply(
    L: torch.Tensor,
    X: torch.Tensor,
    coeff: torch.Tensor,
    eigval: T.Union[torch.Tensor, np.ndarray],
):
    """Matrix exponential with chebyshev polynomial approximation."""

    def body(carry, c):
        T0, T1, Y = carry
        T2 = (2.0 / eigval) * torch.matmul(L, T1) - 2.0 * T1 - T0
        Y = Y + c * T2
        return (T1, T2, Y)

    T0 = X
    Y = 0.5 * coeff[0] * T0
    T1 = (1.0 / eigval) * torch.matmul(L, X) - T0
    Y = Y + coeff[1] * T1

    initial_state = (T0, T1, Y)
    for c in coeff[2:]:
        initial_state = body(initial_state, c)

    _, _, Y = initial_state

    return Y


@torch.no_grad()
def compute_chebychev_coeff_all(eigval, t, K):
    return 2.0 * ive(torch.arange(0, K + 1), -t * eigval)
