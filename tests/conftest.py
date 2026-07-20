"""Shared fixtures for the TorchCFM test suite."""

import pytest
import torch

from torchcfm.conditional_flow_matching import (  # noqa: F401  (star import below)
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.conditional_flow_matching import *  # noqa: F401,F403

# A collection of (batch, *feature) shapes used to parametrize tests across the suite.
SHAPES = [(2, 3), (5, 2), (10, 1), (3, 3, 2)]


@pytest.fixture(params=SHAPES)
def shapes(request):
    """Parametrized fixture yielding each (batch, *feature) shape in ``SHAPES``."""
    return request.param


@pytest.fixture
def rng():
    """A seeded ``torch.Generator`` (seed 42) for reproducible randomness."""
    return torch.Generator().manual_seed(42)


@pytest.fixture
def sample_data(rng):
    """Small pair of standard-normal tensors ``(x0, x1)`` of shape ``(64, 2)``."""
    x0 = torch.randn(64, 2, generator=rng)
    x1 = torch.randn(64, 2, generator=rng)
    return x0, x1


@pytest.fixture
def conditional_flow_matcher():
    """Base independent conditional flow matcher (sigma=0.0)."""
    return ConditionalFlowMatcher(sigma=0.0)


@pytest.fixture
def exact_ot_flow_matcher():
    """Exact optimal-transport conditional flow matcher (sigma=0.0)."""
    return ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)


@pytest.fixture
def target_flow_matcher():
    """Target conditional flow matcher (sigma=0.0)."""
    return TargetConditionalFlowMatcher(sigma=0.0)


@pytest.fixture
def schrodinger_bridge_flow_matcher():
    """Schrödinger bridge conditional flow matcher (sigma must be > 0)."""
    return SchrodingerBridgeConditionalFlowMatcher(sigma=1.0)


@pytest.fixture
def variance_preserving_flow_matcher():
    """Variance-preserving (trigonometric interpolant) flow matcher (sigma=0.0)."""
    return VariancePreservingConditionalFlowMatcher(sigma=0.0)
