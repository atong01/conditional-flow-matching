"""End-to-end smoke tests for training a tiny CFM model."""

import numpy as np
import pytest
import torch

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.models import MLP
from torchcfm.utils import sample_moons


@pytest.mark.slow
def test_cfm_training_reduces_loss():
    """Train a small MLP with CFM on two moons and assert the loss decreases."""
    # Arrange — seed everything for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 256
    x0 = torch.as_tensor(sample_moons(n_samples)).float()
    x1 = torch.as_tensor(sample_moons(n_samples)).float()

    model = MLP(dim=2, w=32, time_varying=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    fm = ConditionalFlowMatcher(sigma=0.0)

    # Fixed evaluation batch (seeded so initial/final loss are comparable)
    torch.manual_seed(0)
    t_eval, xt_eval, ut_eval = fm.sample_location_and_conditional_flow(x0, x1)
    t_eval = t_eval.reshape(-1, 1)

    def eval_loss():
        vt = model(torch.cat([xt_eval, t_eval], dim=1))
        return torch.nn.functional.mse_loss(vt, ut_eval).item()

    # Act — measure initial loss
    initial_loss = eval_loss()

    # Train for a small number of steps
    n_steps = 100
    for _ in range(n_steps):
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
        t = t.reshape(-1, 1)
        vt = model(torch.cat([xt, t], dim=1))
        loss = torch.nn.functional.mse_loss(vt, ut)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = eval_loss()

    # Assert — training should reduce the loss on the fixed evaluation batch
    assert final_loss < initial_loss
