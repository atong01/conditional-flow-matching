"""
Implements Conditional Flow Matching Losses
"""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math

import torch

#------------------------------------------------------------------------------------
##BUG ImportError: attempted relative import with no known parent package
## TO SOLVE LATER AND COMMENT FOR NOW
from optimal_transport import OTPlanSampler
#------------------------------------------------------------------------------------

class ConditionalFlowMatching:
    def __init__(self, sigma: float = 0.0):
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


class ExactOptimalTransportConditionalFlowMatching(ConditionalFlowMatching):
    def __init__(self, sigma: float = 0.0):
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


class TargetConditionalFlowMatching(ConditionalFlowMatching):
    """
    Lipman et al. 2023 style target OT conditional flow matching.
    This class inherits the ConditionalFlowMatching and override the 
    compute_mu_t, compute_sigma_t and compute_conditional_flow functions
    in order to compute [2]'s flow matching
    
    [2] Flow Matching for Generative Modelling, ICLR, Lipman et al.
    """

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

    def sample_location_and_conditional_flow(self, x0, x1):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1)


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


