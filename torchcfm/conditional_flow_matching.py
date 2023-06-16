"""
Implements Conditional Flow Matcher Losses
"""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math

import torch

# ------------------------------------------------------------------------------------
from .optimal_transport import OTPlanSampler

# ------------------------------------------------------------------------------------


def pad_t_like_x(t, x):
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """
    Base class for conditional flow matching methods. This class implements the
    independant conditional flow matching methods from [1] and serves as a parent class
    for all other flow matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: float = 0.0):
        """
        Initialize the ConditionalFlowMatcher class. It requires the [GIVE MORE DETAILS]
        hyper-parameter $\sigma$.

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

    def compute_sigma_t(self, x0, x1, t):
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
        del x0, x1, t
        return self.sigma

    def sample_xt(self, x0, x1, t):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        xt : Tensor, shape (bs, dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        return mu_t + sigma_t * torch.randn_like(mu_t)

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

    def sample_location_and_conditional_flow(self, x0, x1):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = torch.rand(x0.shape[0]).type_as(x0)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def compute_score(self, x0, x1, t, xt):
        """
        Compute the score $\nabla log(pt(x)$

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path p_t

        Returns
        -------
        $\nabla log p_t(x)$ : score

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        n = torch.numel(mu_t[0])
        return (
            -n * torch.log(sigma_t)
            - n / 2 * torch.log(2 * math.pi)
            - torch.sum((xt - mu_t) ** 2) / (2 * sigma_t**2)
        )


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Child class for optimal transport conditional flow matching method. This class implements the
    OT-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the sample_location_and_conditional_flow.
    """

    def __init__(self, sigma: float = 0.0):
        """
        Initialize the ConditionalFlowMatcher class. It requires the [GIVE MORE DETAILS]
        hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        with respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1)


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    Lipman et al. 2023 style target OT conditional flow matching.
    This class inherits the ConditionalFlowMatcher and override the
    compute_mu_t, compute_sigma_t and compute_conditional_flow functions
    in order to compute [2]'s flow matching

    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path tx1, see (Eq.20) [3].

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

    def compute_sigma_t(self, x0, x1, t):
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
        del x0, x1
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
    """
    Child class for Schrödinger bridge conditional flow matching method. This class implements the
    SB-CFM methods from [1] and inherits the ConditionalFlowMatcher parent class.

    It overrides the compute_sigma_t, compute_conditional_flow and sample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Initialize the SchrodingerBridgeConditionalFlowMatcher class. It requires the
        hyper-parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : float
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * self.sigma**2)

    def compute_sigma_t(self, x0, x1, t):
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
        # Potential bug below, should be sigma^2 to be consistent with paper
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field
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
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def sample_location_and_conditional_flow(self, x0, x1):
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


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1)


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        return (
            math.pi
            / 2
            * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)
        )


class SF2M(ConditionalFlowMatcher):
    """
    Child class for simulation-free score and flow matching [2]. This class implements the
    SF2M method from [2] and inherits the ConditionalFlowMatcher parent class.

    It overrides the all functions.
    """

    def __init__(self, sigma: float = 0.1):
        """
        Initialize the ConditionalFlowMatcher class. It requires the [GIVE MORE DETAILS]
        hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : float
        """
        self.sigma = sigma
        self.ot_sampler = OTPlanSampler(method="exact")
        self.entropic_ot_sampler = OTPlanSampler(method="sinkhorn", reg=1.0)

    def F(self, t):
        """
        THE NAME OF THIS FUNCTION HAS TO MEAN SMTH. CURRENTLY NOT ACCEPTABLE.
        """
        t = t * 1.0
        if isinstance(t, float):
            t = torch.tensor(t)
        return t

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
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        ft = self.F(t)
        fone = self.F(1)
        return x0 + (x1 - x0) * ft / fone

    def compute_sigma_t(self, x0, x1, t):
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
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        del x0, x1
        sigma_t = self.F(t) - self.F(t) ** 2 / self.F(1)  # sigma * torch.sqrt(t - t**2)
        return sigma_t

    def sample_xt(self, x0, x1, t):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)

        Returns
        -------
        xt : Tensor, shape (bs, dim)

        References
        ----------
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        return mu_t + sigma_t * torch.randn_like(mu_t)

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
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        ft = self.F(t)  # Find good function name.
        fone = self.F(1)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        my_sigmat = torch.ones_like(t)  # Find good variable name.

        sigma_t_prime = my_sigmat**2 - 2 * ft * my_sigmat**2 / fone
        sigma_t_prime_over_sigma_t = sigma_t_prime / (sigma_t + 1e-8)
        mu_t_prime = (x1 - x0) * my_sigmat**2 / fone
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + mu_t_prime
        return ut

    def sample_location_and_conditional_flow(self, x0, x1):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch


        Returns
        -------
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        # x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        t = torch.rand(x0.shape[0], 1).type_as(x0)
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

    def compute_score(self, x0, x1, t, xt):
        """
        Compute the score $\nabla log(pt(x)$

        Parameters
        ----------
        x0 : Tensor, shape (bs, dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, dim)
            represents the source minibatch
        t : float, shape (bs, 1)
        xt : Tensor, shape (bs, dim)
            represents the samples drawn from probability path p_t

        Returns
        -------
        $\nabla log p_t(x)$ : score

        References
        ----------
        [2] Schrödinger bridge via score and flow matching, Preprint, Tong et al.
        """
        # x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(x0, x1, t)
        eps = (xt - mu_t) / sigma_t  # to get the same noise as in ut
        my_sigmat = torch.ones_like(t)  # Find a good variable name
        return -eps * my_sigmat**2 / 2
