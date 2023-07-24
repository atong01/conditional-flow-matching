import numpy as np
import torch


class NoiseScheduler:
    """Base Class for noise schedule.

    The noise schedule is a function that maps time to reference process noise level. We can use
    this to determine the Brownian bridge noise schedule.

    We define the noise schedule with __call__ and the Brownian bridge noise schedule with sigma_t.
    We define F as the integral of the squared reference process noise schedule which is a useful
    intermediate quantity.
    """

    def __call__(self, t):
        """Calculate the reference process noise schedule.

        g(t) in the paper.
        """
        raise NotImplementedError

    def F(self, t):
        """Calculate the integral of the squared reference process noise schedule."""
        raise NotImplementedError

    def sigma_t(self, t):
        """Given the reference process noise schedule, calculate the brownian bridge noise
        schedule."""
        return torch.sqrt(self.F(t) - self.F(t) ** 2 / self.F(1))


class ConstantNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, t):
        return self.sigma

    def F(self, t):
        return self.sigma**2 * t


class LinearDecreasingNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, sigma_max: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, t):
        return torch.sqrt(t * self.sigma_min + (1 - t) * self.sigma_max)

    def F(self, t):
        return (t**2) * self.sigma_min / 2 - (t**2) * self.sigma_max / 2 + self.sigma_max * t


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, sigma_min: float, scale: float):
        self.sigma_min = sigma_min
        self.scale = scale

    def __call__(self, t):
        return self.scale * (1 - (t * np.pi * 2).cos()) + self.sigma_min

    def F(self, t):
        antider = t - (t * 2 * np.pi).sin() / (2 * np.pi)
        antider2 = t - 2 * (t * 2 * np.pi).sin() / (2 * np.pi)
        antider2 += t / 2 + (t * 4 * np.pi).sin() / (8 * np.pi)
        return (
            self.scale**2 * antider2
            + t * self.sigma_min**2
            + self.scale * 2 * self.sigma_min * antider
        )
