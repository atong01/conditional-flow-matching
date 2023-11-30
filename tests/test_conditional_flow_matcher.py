"""Tests for Conditional Flow Matcher classers."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import math

import numpy as np
import pytest
import torch

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    pad_t_like_x,
)
from torchcfm.optimal_transport import OTPlanSampler

TEST_SEED = 1994
TEST_BATCH_SIZE = 128
SIGMA_CONDITION = {
    "sb_cfm": lambda x: x <= 0,
}


def random_samples(shape, batch_size=TEST_BATCH_SIZE):
    """Generate random samples of different dimensions."""
    if isinstance(shape, int):
        shape = [shape]
    return [torch.randn(batch_size, *shape), torch.randn(batch_size, *shape)]


def compute_xt_ut(method, x0, x1, t_given, sigma, epsilon):
    if method == "vp_cfm":
        sigma_t = sigma
        mu_t = torch.cos(math.pi / 2 * t_given) * x0 + torch.sin(math.pi / 2 * t_given) * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            math.pi
            / 2
            * (torch.cos(math.pi / 2 * t_given) * x1 - torch.sin(math.pi / 2 * t_given) * x0)
        )
    elif method == "t_cfm":
        sigma_t = 1 - (1 - sigma) * t_given
        mu_t = t_given * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (x1 - (1 - sigma) * computed_xt) / sigma_t

    elif method == "sb_cfm":
        sigma_t = sigma * torch.sqrt(t_given * (1 - t_given))
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            (1 - 2 * t_given)
            / (2 * t_given * (1 - t_given) + 1e-8)
            * (computed_xt - (t_given * x1 + (1 - t_given) * x0))
            + x1
            - x0
        )
    elif method in ["exact_ot_cfm", "i_cfm"]:
        sigma_t = sigma
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = x1 - x0

    return computed_xt, computed_ut


def get_flow_matcher(method, sigma):
    if method == "vp_cfm":
        fm = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif method == "t_cfm":
        fm = TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "sb_cfm":
        fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="sinkhorn")
    elif method == "exact_ot_cfm":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "i_cfm":
        fm = ConditionalFlowMatcher(sigma=sigma)
    return fm


def sample_plan(method, x0, x1, sigma):
    if method == "sb_cfm":
        x0, x1 = OTPlanSampler(method="sinkhorn", reg=2 * (sigma**2)).sample_plan(x0, x1)
    elif method == "exact_ot_cfm":
        x0, x1 = OTPlanSampler(method="exact").sample_plan(x0, x1)
    return x0, x1


@pytest.mark.parametrize("method", ["vp_cfm", "t_cfm", "sb_cfm", "exact_ot_cfm", "i_cfm"])
# Test both integer and floating sigma
@pytest.mark.parametrize("sigma", [0.0, 5e-4, 0.5, 1.5, 0, 1])
@pytest.mark.parametrize("shape", [[1], [2], [1, 2], [3, 4, 5]])
def test_fm(method, sigma, shape):
    batch_size = TEST_BATCH_SIZE

    if method in SIGMA_CONDITION.keys() and SIGMA_CONDITION[method](sigma):
        with pytest.raises(ValueError):
            get_flow_matcher(method, sigma)
        return

    FM = get_flow_matcher(method, sigma)
    x0, x1 = random_samples(shape, batch_size=batch_size)
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)
    _ = FM.compute_lambda(t)

    if method in ["sb_cfm", "exact_ot_cfm"]:
        torch.manual_seed(TEST_SEED)
        np.random.seed(TEST_SEED)
        x0, x1 = sample_plan(method, x0, x1, sigma)

    torch.manual_seed(TEST_SEED)
    t_given_init = torch.rand(batch_size)
    t_given = t_given_init.reshape(-1, *([1] * (x0.dim() - 1)))
    sigma_pad = pad_t_like_x(sigma, x0)
    epsilon = torch.randn_like(x0)
    computed_xt, computed_ut = compute_xt_ut(method, x0, x1, t_given, sigma_pad, epsilon)

    assert torch.all(ut.eq(computed_ut))
    assert torch.all(xt.eq(computed_xt))
    assert torch.all(eps.eq(epsilon))
    assert any(t_given_init == t)
