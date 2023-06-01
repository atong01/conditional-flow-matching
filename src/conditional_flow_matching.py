"""
Implements Conditional Flow Matching Losses
"""
import math

import torch

from .optimal_transport import OTPlanSampler


class ConditionalFlowMatching:
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, x0, x1, t):
        del x0, x1, t
        return self.sigma

    def sample_xt(self, x0, x1, t):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        return mu_t + sigma_t * torch.randn_like(mu_t)

    def compute_conditional_flow(self, x0, x1, t, xt):
        del t, xt
        return x1 - x0

    def compute_location_and_target(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def compute_score(self, x0, x1, t, xt):
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        n = torch.numel(mu_t[0])
        return (
            -n * torch.log(sigma_t)
            - n / 2 * torch.log(2 * math.pi)
            - torch.sum((xt - mu_t) ** 2) / (2 * sigma_t**2)
        )


class ExactOptimalTransportConditionalFlowMatching(ConditionalFlowMatching):
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="exact")

    def compute_location_and_target(self, x0, x1):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().compute_location_and_target(x0, x1)


class TargetConditionalFlowMatching(ConditionalFlowMatching):
    """Lipman et al. 2023 style target OT conditional flow matching."""

    def compute_mu_t(self, x0, x1, t):
        del x0
        return t * x1

    def compute_sigma_t(self, x0, x1, t):
        del x0, x1
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        del x0
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatching(ConditionalFlowMatching):
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * self.sigma**2)

    def compute_sigma_t(self, x0, x1, t):
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        mu_t = self.compute_mu_t(self, x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def compute_location_and_target(self, x0, x1):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().compute_location_and_target(x0, x1)


class VariancePreservingConditionalFlowMatching(ConditionalFlowMatching):
    def compute_mu_t(self, x0, x1, t):
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        return (
            math.pi
            / 2
            * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)
        )


