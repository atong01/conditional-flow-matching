"""Hungarian coupling must preserve endpoint shapes and matched subset gradients."""

import itertools
import math

import numpy as np
import pytest
import torch

from torchcfm.optimal_transport import OTPlanSampler


def exhaustive_assignment(source, target):
    """Enumerate all minimum-cardinality matchings instead of calling an OT solver."""
    left = source.detach().cpu().numpy().reshape(len(source), -1)
    right = target.detach().cpu().numpy().reshape(len(target), -1)
    cost = ((left[:, None] - right[None, :]) ** 2).sum(-1)
    cardinality = min(len(source), len(target))
    candidates = []
    for rows in itertools.combinations(range(len(source)), cardinality):
        for columns in itertools.permutations(range(len(target)), cardinality):
            objective = sum(float(cost[row, column]) for row, column in zip(rows, columns))
            candidates.append((objective, rows, columns))
    candidates.sort()
    assert (
        candidates[1][0] - candidates[0][0] > 0.01
    ), "The oracle example must have a unique assignment."
    return np.asarray(candidates[0][1]), np.asarray(candidates[0][2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample_shape", [(2,), (1, 2, 2), (2, 2, 2)])
@pytest.mark.parametrize("sizes", [(3, 3), (4, 2), (2, 4)])
@pytest.mark.parametrize("normalize", [False, True])
def test_assignment_preserves_structure_matching_and_gradients(
    dtype, sample_shape, sizes, normalize
):
    source_size, target_size = sizes
    features = math.prod(sample_shape)
    source_centers = torch.tensor([0.0, 4.0, 10.0, 100.0], dtype=dtype)[:source_size]
    target_centers = torch.tensor([9.7, 0.2, 4.3, -50.0], dtype=dtype)[:target_size]
    feature_weights = torch.linspace(0.7, 1.2, features, dtype=dtype)
    source = (
        (source_centers[:, None] * feature_weights)
        .reshape(source_size, *sample_shape)
        .requires_grad_()
    )
    target = (
        (target_centers[:, None] * feature_weights)
        .reshape(target_size, *sample_shape)
        .requires_grad_()
    )
    rows, columns = exhaustive_assignment(source, target)
    sampler = OTPlanSampler(method="exact", normalize_cost=normalize)
    paired_source, paired_target = sampler.sample_plan_with_scipy(source, target)
    assert paired_source.shape == (len(rows), *sample_shape)
    assert paired_target.shape == (len(rows), *sample_shape)
    torch.testing.assert_close(paired_source, source[rows], rtol=0.0, atol=0.0)
    torch.testing.assert_close(paired_target, target[columns], rtol=0.0, atol=0.0)
    weights = torch.linspace(0.1, 0.9, paired_source.numel(), dtype=dtype).reshape_as(
        paired_source
    )
    actual_source_gradient, actual_target_gradient = torch.autograd.grad(
        (weights * paired_source.square()).sum() + 0.3 * paired_target.sin().sum(),
        (source, target),
    )
    expected_source_gradient = torch.zeros_like(source)
    expected_target_gradient = torch.zeros_like(target)
    expected_source_gradient[rows] = 2.0 * weights * source.detach()[rows]
    expected_target_gradient[columns] = 0.3 * target.detach()[columns].cos()
    torch.testing.assert_close(
        actual_source_gradient, expected_source_gradient, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        actual_target_gradient, expected_target_gradient, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "endpoint_shapes",
    [
        ((4,), (4,)),
        ((1, 2, 2), (1, 2, 2)),
        ((4,), (1, 2, 2)),
        ((0,), (0,)),
        ((1, 0, 2), (1, 0, 2)),
    ],
)
@pytest.mark.parametrize("sizes", [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("normalize", [False, True])
def test_empty_assignment_preserves_endpoint_shapes_and_zero_gradients(
    dtype, endpoint_shapes, sizes, normalize
):
    """A matching with an empty endpoint has zero pairs and zero gather gradients."""
    source_shape, target_shape = endpoint_shapes
    source = torch.ones((sizes[0], *source_shape), dtype=dtype, requires_grad=True)
    target = torch.ones((sizes[1], *target_shape), dtype=dtype, requires_grad=True)
    paired_source, paired_target = OTPlanSampler(
        method="exact", normalize_cost=normalize
    ).sample_plan_with_scipy(source, target)

    # Cardinality follows the matching definition, independent of a cost solver.
    assert min(sizes) == 0
    assert paired_source.shape == (0, *source_shape)
    assert paired_target.shape == (0, *target_shape)
    assert paired_source.dtype == paired_target.dtype == dtype
    assert paired_source.device == source.device
    assert paired_target.device == target.device
    assert paired_source.numel() == paired_target.numel() == 0

    source_gradient, target_gradient = torch.autograd.grad(
        paired_source.sum() + paired_target.sum(), (source, target)
    )
    torch.testing.assert_close(source_gradient, torch.zeros_like(source), rtol=0.0, atol=0.0)
    torch.testing.assert_close(target_gradient, torch.zeros_like(target), rtol=0.0, atol=0.0)
