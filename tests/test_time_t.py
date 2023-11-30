"""Tests for time Tensor t."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import pytest
import torch

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

seed = 1994
batch_size = 128


@pytest.mark.parametrize(
    "FM",
    [
        ConditionalFlowMatcher(sigma=0.0),
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.0),
        TargetConditionalFlowMatcher(sigma=0.0),
        SchrodingerBridgeConditionalFlowMatcher(sigma=0.1),
        VariancePreservingConditionalFlowMatcher(sigma=0.0),
    ],
)
def test_random_Tensor_t(FM):
    # Test sample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2)
    x1 = torch.randn(batch_size, 2)

    torch.manual_seed(seed)
    t_given = torch.rand(batch_size)
    t_given, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=t_given)

    torch.manual_seed(seed)
    t_random, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=None)

    assert any(t_given == t_random)


@pytest.mark.parametrize(
    "FM",
    [
        ExactOptimalTransportConditionalFlowMatcher(sigma=0.0),
        SchrodingerBridgeConditionalFlowMatcher(sigma=0.1),
    ],
)
@pytest.mark.parametrize("return_noise", [True, False])
def test_guided_random_Tensor_t(FM, return_noise):
    # Test guided_sample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2)
    y0 = torch.randint(high=10, size=(batch_size, 1))
    x1 = torch.randn(batch_size, 2)
    y1 = torch.randint(high=10, size=(batch_size, 1))

    torch.manual_seed(seed)
    t_given = torch.rand(batch_size)
    t_given = FM.guided_sample_location_and_conditional_flow(
        x0, x1, y0=y0, y1=y1, t=t_given, return_noise=return_noise
    )[0]

    torch.manual_seed(seed)
    t_random = FM.guided_sample_location_and_conditional_flow(
        x0, x1, y0=y0, y1=y1, t=None, return_noise=return_noise
    )[0]

    assert any(t_given == t_random)
