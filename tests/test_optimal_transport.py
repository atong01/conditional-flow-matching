"""Tests for Conditional Flow Matcher classers."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import numpy as np
import ot
import pytest
import torch

from torchcfm.optimal_transport import OTPlanSampler, wasserstein

ot_sampler = OTPlanSampler(method="exact")


def test_sample_map(batch_size=128):
    # Build sparse random OT map
    map = np.eye(batch_size)
    rng = np.random.default_rng()
    permuted_map = rng.permutation(map, axis=1)

    # Sample elements from the OT plan
    # All elements should be sampled only once
    indices = ot_sampler.sample_map(permuted_map, batch_size=batch_size, replace=False)

    # Reconstruct the coupling from the sampled elements
    reconstructed_map = np.zeros((batch_size, batch_size))
    for i in range(batch_size):
        reconstructed_map[indices[0][i], indices[1][i]] = 1
    assert np.array_equal(reconstructed_map, permuted_map)


def test_get_map(batch_size=128):
    x0 = torch.randn(batch_size, 2, 2, 2)
    x1 = torch.randn(batch_size, 2, 2, 2)

    M = torch.cdist(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1)) ** 2
    pot_pi = ot.emd(ot.unif(x0.shape[0]), ot.unif(x1.shape[0]), M.numpy())

    pi = ot_sampler.get_map(x0, x1)

    assert np.array_equal(pi, pot_pi)


def test_sample_plan(batch_size=128, seed=1980):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = torch.randn(batch_size, 2, 2, 2)
    x1 = torch.randn(batch_size, 2, 2, 2)

    pi = ot_sampler.get_map(x0, x1)
    indices_i, indices_j = ot_sampler.sample_map(pi, batch_size=batch_size, replace=True)
    new_x0, new_x1 = x0[indices_i], x1[indices_j]

    torch.manual_seed(seed)
    np.random.seed(seed)

    sampled_x0, sampled_x1 = ot_sampler.sample_plan(x0, x1, replace=True)

    assert torch.equal(new_x0, sampled_x0)
    assert torch.equal(new_x1, sampled_x1)


def test_wasserstein(batch_size=128, seed=1980):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = torch.randn(batch_size, 2, 2, 2)
    x1 = torch.randn(batch_size, 2, 2, 2)

    M = torch.cdist(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1))
    pot_W22 = ot.emd2(ot.unif(x0.shape[0]), ot.unif(x1.shape[0]), (M**2).numpy())
    pot_W2 = np.sqrt(pot_W22)
    W2 = wasserstein(x0, x1, "exact")

    pot_W1 = ot.emd2(ot.unif(x0.shape[0]), ot.unif(x1.shape[0]), M.numpy())
    W1 = wasserstein(x0, x1, "exact", power=1)

    pot_eot = ot.sinkhorn2(
        ot.unif(x0.shape[0]),
        ot.unif(x1.shape[0]),
        M.numpy(),
        reg=0.01,
        numItermax=int(1e7),
    )
    eot = wasserstein(x0, x1, "sinkhorn", reg=0.01, power=1)

    with pytest.raises(ValueError):
        eot = wasserstein(x0, x1, "noname", reg=0.01, power=1)

    assert pot_W2 == W2
    assert pot_W1 == W1
    assert pot_eot == eot
