"""Tests for models in ``torchcfm.models``."""

import torch

from torchcfm.models import MLP
from torchcfm.models.models import GradModel
from torchcfm.models.unet import UNetModel


def test_initialize_models():
    UNetModel(
        dim=(1, 28, 28),
        num_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    )
    MLP(dim=2, time_varying=True, w=64)


def test_mlp_forward_time_varying():
    # Arrange — time_varying MLP expects input of dim (dim + 1)
    model = MLP(dim=2, w=32, time_varying=True)
    x = torch.randn(10, 2)
    t = torch.rand(10, 1)
    xt = torch.cat([x, t], dim=1)
    # Act
    out = model(xt)
    # Assert
    assert out.shape == (10, 2)
    assert torch.isfinite(out).all()


def test_mlp_forward_not_time_varying():
    # Arrange — non-time-varying MLP takes input of dim directly
    model = MLP(dim=2, w=32, time_varying=False)
    x = torch.randn(10, 2)
    # Act
    out = model(x)
    # Assert
    assert out.shape == (10, 2)
    assert torch.isfinite(out).all()


def test_mlp_forward_out_dim():
    # Arrange — explicit out_dim different from dim
    model = MLP(dim=2, out_dim=4, w=32, time_varying=False)
    x = torch.randn(10, 2)
    # Act
    out = model(x)
    # Assert
    assert out.shape == (10, 4)


def test_grad_model_instantiation_and_forward():
    # Arrange — GradModel wraps an "action" and returns the gradient (minus last dim)
    action = MLP(dim=3, w=32, time_varying=False)
    grad_model = GradModel(action)
    x = torch.randn(10, 3)
    # Act
    grad = grad_model(x)
    # Assert — gradient w.r.t. x has shape (10, 3); [:, :-1] yields (10, 2)
    assert grad.shape == (10, 2)
    assert torch.isfinite(grad).all()


def test_grad_model_is_gradient():
    # Arrange — verify the output is actually the gradient of the action
    action = MLP(dim=3, w=32, time_varying=False)
    grad_model = GradModel(action)
    x = torch.randn(5, 3).clone().requires_grad_(True)
    # Act
    grad = grad_model(x)
    # Manually compute the expected gradient
    loss = torch.sum(action(x))
    (expected_grad,) = torch.autograd.grad(loss, x, create_graph=True)
    # Assert — GradModel returns all but the last column of the gradient
    assert torch.allclose(grad, expected_grad[:, :-1])
