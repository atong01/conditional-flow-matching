"""Tests for time Tensor t."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import pytest
import torch
import numpy as np
from torchcfm.optimal_transport import OTPlanSampler

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)

seed = 1994
batch_size = 128
sigma = 0.5


@pytest.mark.parametrize(
    "FM",
    [
        ConditionalFlowMatcher(sigma=sigma),
    ],
)
def test_random_sample_location_and_vector_field(FM):
    # Test sample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2, 2)
    x1 = torch.randn(batch_size, 2, 2)

    torch.manual_seed(seed)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)

    torch.manual_seed(seed)
    sigma_t = torch.FloatTensor([sigma]).reshape(-1, *([1] * (x0.dim() - 1))) 
    t_given = torch.rand(batch_size)
    t_given = t_given.reshape(-1, *([1] * (x0.dim() - 1)))

    epsilon = torch.randn_like(x0)
    mu_t = t_given * x1 + (1 - t_given) * x0
    computed_xt = mu_t + sigma_t * epsilon
    computed_ut = x1 - x0

    assert torch.all(ut.eq(computed_ut))
    assert torch.all(xt.eq(computed_xt))
    assert torch.all(eps.eq(epsilon))


@pytest.mark.parametrize(
    "FM",
    [
        ExactOptimalTransportConditionalFlowMatcher(sigma=sigma),
    ],
)
def test_random_sample_location_and_vector_field_OT(FM):
    # Test sample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2, 2)
    x1 = torch.randn(batch_size, 2, 2)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    x0, x1 = OTPlanSampler(method="exact").sample_plan(x0, x1)

    sigma_t = torch.FloatTensor([sigma]).reshape(-1, *([1] * (x0.dim() - 1))) 
    t_given = torch.rand(batch_size)
    t_given = t_given.reshape(-1, *([1] * (x0.dim() - 1)))

    epsilon = torch.randn_like(x0)
    mu_t = t_given * x1 + (1 - t_given) * x0
    computed_xt = mu_t + sigma_t * epsilon
    computed_ut = x1 - x0

    assert torch.all(ut.eq(computed_ut))
    assert torch.all(xt.eq(computed_xt))
    assert torch.all(eps.eq(epsilon))
    

