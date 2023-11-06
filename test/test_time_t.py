"""Tests for time Tensor t"""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import torch
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher
)
from torchcfm.utils import *

seed=1994
batch_size=128

def test_random_Tensor_t(FM):
    x0 = torch.randn(batch_size, 2)
    x1 = torch.randn(batch_size, 2)
    torch.manual_seed(seed)
    t_given = torch.rand(batch_size)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=t_given)
    
    torch.manual_seed(seed)
    t_random, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=None)

    assert any(t_given==t_random)

ICFM = ConditionalFlowMatcher(sigma=0.0)
OTCFM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
FM = TargetConditionalFlowMatcher(sigma=0.0)
SBCFM = SchrodingerBridgeConditionalFlowMatcher(sigma=0.0)
SI = VariancePreservingConditionalFlowMatcher(sigma=0.0)

list_FM = [ICFM, OTCFM, SBCFM, FM, SI]

for FM in list_FM:
    test_random_Tensor_t(FM)
