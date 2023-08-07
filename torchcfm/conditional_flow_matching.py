"""Implements Conditional Flow Matcher Losses."""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math

import torch

from .optimal_transport import OTPlanSampler


def pad_t_like_x(t, x):
    if isinstance(t, float):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: float = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the [GIVE MORE DETAILS] hyper-
        parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        epsilon : Tensor, shape (bs, dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, y0=None, y1=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = torch.rand(x0.shape[0]).type_as(x0)
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        to_return = [t, xt, ut]
        if y0 is not None:
            to_return.append(y0)
        if y1 is not None:
            to_return.append(y1)
        if return_noise:
            to_return.append(eps)
        return to_return

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(XXX) [1].

        Parameters
        ----------
        t : float, shape (bs, 1)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for optimal transport conditional flow matching method. This class implements
    the OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: float = 0.0):
        r"""Initialize the ConditionalFlowMatcher class. It requires the [GIVE MORE DETAILS] hyper-
        parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, y0=None, y1=None, return_noise=False):
        r"""
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        return_noise : bool
            return the noise sample epsilon

        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0=y0, y1=y1)
        return super().sample_location_and_conditional_flow(x0, x1, y0, y1, return_noise)


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """Lipman et al. 2023 style target OT conditional flow matching. This class inherits the
    ConditionalFlowMatcher and override the compute_mu_t, compute_sigma_t and
    compute_conditional_flow functions in order to compute [2]'s flow matching.

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        mean mu_t: t * x1

        References
        ----------
        [3] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        return t * x1

    def compute_sigma_t(self, t):
        """
        Compute the mean of the probability path N(t * x1, 1 -(1 - sigma)t), see (Eq.20) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        standard deviation sigma 1 -(1 - sigma)t

        References
        ----------
        [3] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t), see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) t)/(1 - (1 - sigma)t)

        References
        ----------
        [1] Flow Matching for Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """Child class for Schrödinger bridge conditional flow matching method. This class implements
    the SB-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and
    sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: float = 1.0, ot_method="exact"):
        r"""Initialize the SchrodingerBridgeConditionalFlowMatcher class. It requires the hyper-
        parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : float
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        self.sigma = sigma
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def compute_sigma_t(self, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1, y0=None, y1=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) with respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        return_noise: bool
            return the noise sample epsilon


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, shape (bs, dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.sample_plan_with_labels(x0, x1, y0=y0, y1=y1)
        return super().sample_location_and_conditional_flow(x0, x1, y0, y1, return_noise)


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)
