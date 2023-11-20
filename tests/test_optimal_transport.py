"""Tests for Conditional Flow Matcher classers."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import math

import numpy as np
import pytest
import torch
import ot

from torchcfm.optimal_transport import OTPlanSampler

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
    reconstructed_a = np.zeros((batch_size, batch_size))
    for i in range(batch_size):
        reconstructed_a[indices[0][i], indices[1][i]] = 1
    assert np.array_equal(reconstructed_a, permuted_map)


def test_get_map(batch_size=128):
    x0 = torch.randn(batch_size, 2, 2, 2)
    x1 = torch.randn(batch_size, 2, 2, 2)

    M = torch.cdist(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1))**2
    pot_pi = ot.emd(ot.unif(x0.shape[0]), ot.unif(x1.shape[0]), M.numpy())

    pi = ot_sampler.get_map(x0, x1)

    assert np.array_equal(pi, pot_pi)
