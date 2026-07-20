"""Tests for utility functions in ``torchcfm.utils``."""

import numpy as np
import pytest
import torch

from torchcfm.models.models import MLP
from torchcfm.utils import eight_normal_sample, sample_8gaussians, sample_moons, torch_wrapper


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_eight_normal_sample_shape_and_finite(n_samples, dim):
    # Arrange & Act
    data = eight_normal_sample(n_samples, dim)
    # Assert
    assert isinstance(data, torch.Tensor)
    assert data.shape == (n_samples, dim)
    assert torch.isfinite(data).all()


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_eight_normal_sample_default_params(n_samples):
    # Arrange & Act — default scale=1, var=1
    data = eight_normal_sample(n_samples, 2)
    # Assert
    assert isinstance(data, torch.Tensor)
    assert data.shape == (n_samples, 2)
    assert torch.isfinite(data).all()


def test_eight_normal_sample_edge_case_single_sample():
    # Arrange & Act — smallest batch size with non-default scale/var
    data = eight_normal_sample(1, 2, scale=5, var=0.5)
    # Assert
    assert data.shape == (1, 2)
    assert torch.isfinite(data).all()


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_sample_moons_shape_and_finite(n_samples):
    # Arrange & Act
    data = sample_moons(n_samples)
    # Assert — sample_moons returns a numpy array of shape (n_samples, 2)
    assert np.asarray(data).shape == (n_samples, 2)
    assert torch.isfinite(torch.as_tensor(data)).all()


def test_sample_moons_edge_case_single_sample():
    # Arrange & Act — smallest batch size
    data = sample_moons(1)
    # Assert
    assert np.asarray(data).shape == (1, 2)
    assert torch.isfinite(torch.as_tensor(data)).all()


@pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
def test_sample_8gaussians_shape_and_finite(n_samples):
    # Arrange & Act
    data = sample_8gaussians(n_samples)
    # Assert
    assert isinstance(data, torch.Tensor)
    assert data.shape == (n_samples, 2)
    assert torch.isfinite(data).all()


def test_sample_8gaussians_edge_case_single_sample():
    # Arrange & Act — smallest batch size
    data = sample_8gaussians(1)
    # Assert
    assert data.shape == (1, 2)
    assert torch.isfinite(data).all()


def test_torch_wrapper_returns_callable_with_correct_shape():
    # Arrange — MLP expects an input of dim (dim + 1) when time_varying
    model = MLP(dim=2, w=16, time_varying=True)
    wrapper = torch_wrapper(model)
    t = torch.tensor(0.5)
    x = torch.randn(10, 2)
    # Act
    out = wrapper(t, x)
    # Assert
    assert isinstance(out, torch.Tensor)
    assert out.shape == (10, 2)
    assert torch.isfinite(out).all()


def test_torch_wrapper_is_torch_module():
    # Arrange
    model = MLP(dim=2, w=16, time_varying=True)
    wrapper = torch_wrapper(model)
    # Assert — torch_wrapper subclasses torch.nn.Module for torchdyn compatibility
    assert isinstance(wrapper, torch.nn.Module)
    assert wrapper.model is model


@pytest.mark.parametrize("batch_size", [1, 5, 32])
def test_torch_wrapper_parametrized(batch_size):
    # Arrange
    model = MLP(dim=2, w=16, time_varying=True)
    wrapper = torch_wrapper(model)
    t = torch.tensor(0.5)
    x = torch.randn(batch_size, 2)
    # Act
    out = wrapper(t, x)
    # Assert
    assert out.shape == (batch_size, 2)
    assert torch.isfinite(out).all()


@pytest.mark.skip(reason="plot_trajectories requires a matplotlib display backend")
def test_plot_trajectories():
    """Skipped: plotting requires an interactive matplotlib display."""
    from torchcfm.utils import plot_trajectories

    traj = torch.randn(5, 100, 2)
    plot_trajectories(traj)
