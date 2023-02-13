from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.distributions import MultivariateNormal
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
from torchdyn.core import NeuralODE

from .components.augmentation import (
    AugmentationModule,
    AugmentedVectorField,
    Sequential,
)
from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import OTPlanSampler
from .components.plotting import plot_paths, plot_scatter_and_flow, store_trajectories
from .utils import get_wandb_logger


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.

        Args:
            net: torch module representing dx/dt = f(t, x) for t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "dim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is used in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            leaveout_timepoint: which (if any) timepoint to leave out during the training phase
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["net", "optimizer", "datamodule", "augmentations"], logger=False
        )
        self.is_trajectory = datamodule.IS_TRAJECTORY
        self.dim = datamodule.dim
        self.net = net(dim=datamodule.dim)
        self.augmentations = augmentations
        self.node = NeuralODE(self.net)
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs)
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            # cnf_estimator=None if self.is_trajectory else "exact",
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs)
        self.val_aug_node = Sequential(
            self.val_augmentations.augmenter,
            NeuralODE(self.val_aug_net, solver="rk4"),
        )
        self.aug_node = Sequential(
            self.augmentations.augmenter,
            NeuralODE(self.aug_net, sensitivity="autograd"),
        )
        self.optimizer = optimizer
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        X = self.unpack_batch(batch)
        X_start = X[:, t_span[0], :]
        traj = self.node.trajectory(X_start, t_span=t_span)
        return traj

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x)

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if self.is_trajectory:
            return torch.stack(batch, dim=1)
        return batch

    def preprocess_batch(self, X, training=False):
        """converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1)
        if self.is_trajectory:
            batch_size, times, dim = X.shape
            if training and self.hparams.leaveout_timepoint > 0:
                # Select random except for the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,))
                t_select[t_select >= self.hparams.leaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            for i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.leaveout_timepoint:
                    ti_next += 1
                x0.append(X[i, ti])
                x1.append(X[i, ti_next])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size, dim = X.shape
            # If no trajectory assume generate from standard normal
            x0 = torch.randn(batch_size, X.shape[1])
            x1 = X
        return x0, x1, t_select

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)

        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        if self.hparams.avg_size > 0:
            t = torch.rand(1, 1).repeat(X.shape[0], 1)
        else:
            t = torch.rand(X.shape[0], 1)
        ut = x1 - x0
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min

        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # t that network sees is incremented by first timepoint
        t = t + t_select[:, None]
        x = mu_t + sigma_t * torch.randn_like(x0)

        if self.hparams.avg_size > 0:
            pt = torch.exp(
                -0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2)
            )  # (1/(sigma_t * np.sqrt(2 * np.pi)) *
            batch_size = x1.shape[0]
            ind = torch.randint(
                batch_size, size=(batch_size, self.hparams.avg_size - 1)
            )  # randomly (non-repreat) sample m-many index
            # always include self
            ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
            pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
            ut_sub = torch.stack([ut[ind[i]] for i in range(batch_size)])
            p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
            ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum

            aug_x = self.aug_net(t[:1], x[:1], augmented_input=False)
            reg, vt = self.augmentations(aug_x)
            return torch.mean(reg), self.criterion(vt, ut[:1])
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        reg, mse = self.step(batch, training=True)
        loss = mse + reg
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        shapes = [b.shape[0] for b in batch]
        if shapes.count(shapes[0]) == len(shapes):
            reg, mse = self.step(batch, training=False)
            loss = mse + reg
            self.log_dict(
                {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
            )
            return {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}
        return {"x": batch}

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            # x is jagged if doing a trajectory
            x = outputs[0]["x"]
            ts = len(x)
            x0 = x[0]
            x_rest = x[1:]

        elif self.is_trajectory:
            v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
            x = v["x"]
            ts = x.shape[1]
            x0 = x[:, 0, :]
            x_rest = x[:, 1:]

        else:
            v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
            x = v["x"]
            # Sample some random points for the plotting function
            rand = torch.randn_like(x)
            # rand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
            x = torch.stack([rand, x], dim=1)
            ts = x.shape[1]
            x0 = x[:, 0, :]
            x_rest = x[:, 1:]

        if False and self.dim == 2:
            plot_scatter_and_flow(
                x,
                self.net,
                title=f"{self.current_epoch}_flow",
                wandb_logger=wandb_logger,
            )

        if self.current_epoch == 0:
            # skip epoch zero for numerical integration reasons
            return

        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)
        aug_dims = self.val_augmentations.aug_dims
        regs = []
        trajs = []
        for i in range(ts - 1):
            if self.is_trajectory and prefix == "test":
                x_start = x[i]
            else:
                x_start = x[:, i, :]
            _, aug_traj = self.val_aug_node(x_start, t_span + i)
            aug, traj = aug_traj[-1, :, :aug_dims], aug_traj[-1, :, aug_dims:]
            # traj = torch.transpose(traj, 0, 1)  # Now [Batch, Time, Dim]
            trajs.append(traj)
            # Mean regs over batch dimension
            regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
        regs = np.stack(regs).mean(axis=0)
        names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
        self.log_dict(dict(zip(names, regs)))

        # Evaluate the fit

        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            names, dists = compute_distribution_distances(trajs[:-1], x_rest[:-1])
        else:
            names, dists = compute_distribution_distances(trajs, x_rest)
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        if self.hparams.leaveout_timepoint >= 0:
            to_add = {
                f"{prefix}/t_out/{key.split('/')[-1]}": val
                for key, val in d.items()
                if key.startswith(f"{prefix}/t{self.hparams.leaveout_timepoint}")
            }
            d.update(to_add)

        self.log_dict(d)

        if False:
            plot_paths(
                x,
                self.net,
                title=f"{self.current_epoch}_paths",
                wandb_logger=wandb_logger,
            )

        if prefix == "test":
            store_trajectories(x, self.net)

        # Reverse trajectories
        # _, aug_traj = self.val_aug_node(x[:, 0, :], t_span[::-1])
        # aug, traj = aug_traj[::100, :, :aug_dims], aug_traj[::100, :, aug_dims:]
        # traj = torch.transpose(traj, 0, 1)  # Now [Batch, Time, Dim]

        # if self.val_augmentations.cnf_estimator:
        # Add in the original log-likelihood.
        #    mn = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        #    aug[-1, :, 0] += mn.log_prob(x[:, 0, :])

        # use sum for the log likelihood
        # regs = torch.sum(aug[-1], dim=0).detach().cpu().numpy()
        # names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
        # self.log_dict(dict(zip(names[:1], regs[:1])))

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        return self.optimizer(params=self.parameters())


class SBCFMLitModule(CFMLitModule):
    """Implements a Schrodinger Bridge based conditional flow matching model.

    This is similar to the OTCFM loss, however with the variance varying with t*(1-t). This has
    provably equal probability flow to the Schrodinger bridge solution when the transport is
    computed with the squared Euclidean distance on R^d.
    """

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""

        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)

        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        if self.hparams.avg_size > 0:
            t = torch.rand(1, 1).repeat(x1.shape[0], 1)
        else:
            t = torch.rand(x1.shape[0], 1)
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min * torch.sqrt(t - t**2)
        x = mu_t + sigma_t * torch.randn_like(x0)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (x - mu_t) + x1 - x0
        # t that network sees is incremented by first timepoint
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2
        t = t + t_select[:, None]

        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut_all = (
            sigma_t_prime_over_sigma_t * (x[:, None, :] - mu_t[None, :, :]) + (x1 - x0)[None, :, :]
        )
        if self.hparams.avg_size > 0:
            pt = torch.exp(
                -0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2)
            )  # (1/(sigma_t * np.sqrt(2 * np.pi)) *
            batch_size = x1.shape[0]
            ind = torch.randint(
                batch_size, size=(batch_size, self.hparams.avg_size - 1)
            )  # randomly (non-repreat) sample m-many index
            # always include self
            ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
            pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
            ut_sub = torch.stack([ut_all[i, ind[i]] for i in range(batch_size)])
            p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
            ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum

        aug_x = self.aug_net(t[:1], x[:1], augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut[:1])


class FMLitModule(CFMLitModule):
    """Implements a Lipman et al. 2023 style flow matching loss.

    This maps the standard normal distribution to the data distribution by using conditional flows
    that are the optimal transport flow from a narrow Gaussian around a datapoint to a standard N(x
    | 0, 1).
    """

    def step(self, batch: Any, training: bool = False):
        # Only works for a single distribiton
        assert not self.is_trajectory
        x1 = batch
        # t = torch.rand(x1.shape[0], 1)
        if self.hparams.avg_size > 0:
            t = torch.rand(1, 1).repeat(x1.shape[0], 1)
        else:
            t = torch.rand(x1.shape[0], 1)
        sigma_min = self.hparams.sigma_min
        mu_t = t * x1
        sigma_t = 1 - (1 - sigma_min) * t
        x = mu_t + sigma_t * torch.randn_like(x1)
        ut = (x1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)
        # batch, batch, dim
        ut_all = (x1[None, :, :] - (1 - sigma_min) * x[:, None, :]) / (1 - (1 - sigma_min) * t)
        if self.hparams.avg_size > 0:
            pt = torch.exp(
                -0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2)
            )  # (1/(sigma_t * np.sqrt(2 * np.pi)) *
            batch_size = x1.shape[0]
            ind = torch.randint(
                batch_size, size=(batch_size, self.hparams.avg_size - 1)
            )  # randomly (non-repreat) sample m-many index
            # always include self
            ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
            pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
            ut_sub = torch.stack([ut_all[i, ind[i]] for i in range(batch_size)])
            p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
            ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum

        aug_x = self.aug_net(t[:1], x[:1], augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut[:1])


class SplineCFMLitModule(CFMLitModule):
    """Implements cubic spline version of OT-CFM."""

    def preprocess_batch(self, X, training=False):
        """converts a batch of data into matched a random pair of (x0, x1)"""
        lotp = self.hparams.leaveout_timepoint
        valid_times = torch.arange(X.shape[1]).type_as(X)
        t_select = torch.zeros(1)
        batch_size, times, dim = X.shape
        # TODO handle leaveout case
        if training and self.hparams.leaveout_timepoint > 0:
            # Select random except for the leftout timepoint
            t_select = torch.randint(times - 2, size=(batch_size,))
            X = torch.cat([X[:, :lotp], X[:, lotp + 1 :]], dim=1)
            valid_times = valid_times[valid_times != lotp]
        else:
            t_select = torch.randint(times - 1, size=(batch_size,))
        traj = torch.from_numpy(self.ot_sampler.sample_trajectory(X)).type_as(X)
        x0 = []
        x1 = []
        for i in range(batch_size):
            x0.append(traj[i, t_select[i]])
            x1.append(traj[i, t_select[i] + 1])
        x0, x1 = torch.stack(x0), torch.stack(x1)
        if training and self.hparams.leaveout_timepoint > 0:
            t_select[t_select >= self.hparams.leaveout_timepoint] += 1

        coeffs = natural_cubic_spline_coeffs(valid_times, traj)
        spline = NaturalCubicSpline(coeffs)
        return x0, x1, t_select, spline

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        assert self.is_trajectory
        X = self.unpack_batch(batch)
        x0, x1, t_select, spline = self.preprocess_batch(X, training)

        t = torch.rand(X.shape[0], 1)
        # t [batch, 1]
        # coeffs [batch, times, dims]
        # t that network sees is incremented by first timepoint
        t = t + t_select[:, None]
        ut = torch.stack([spline.derivative(b[0])[i] for i, b in enumerate(t)], dim=0)
        mu_t = torch.stack([spline.evaluate(b[0])[i] for i, b in enumerate(t)], dim=0)
        sigma_t = self.hparams.sigma_min

        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        x = mu_t + sigma_t * torch.randn_like(x0)
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut)


class CNFLitModule(CFMLitModule):
    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        return super().forward_integrate(batch, t_span + 1)

    def step(self, batch: Any, training: bool = False):
        obs = self.unpack_batch(batch)
        if not self.is_trajectory:
            obs = obs[:, None, :]
        aug_dims = self.augmentations.aug_dims
        even_ts = torch.arange(obs.shape[1]).to(obs) + 1
        self.prior = MultivariateNormal(
            torch.zeros(self.dim).type_as(obs), torch.eye(self.dim).type_as(obs)
        )
        # Minimize the log likelihood by integrating all back to the initial timepoint
        reversed_ts = torch.cat([torch.flip(even_ts, [0]), torch.tensor([0]).type_as(even_ts)])

        # If only one timepoint then Gaussian is at t0, data t1
        # If multiple timepoints then Gaussian is at t_{-1} data is at times 0 to T
        if self.is_trajectory:
            reversed_ts -= 1
        losses = []
        regs = []
        for t in range(len(reversed_ts) - 1):
            # When leaving out a timepoint simply skip it in the backwards integration
            if self.hparams.leaveout_timepoint == t:
                continue
            ts, x = reversed_ts[t:], obs[:, len(even_ts) - t - 1, :]
            _, x = self.aug_node(x, ts)
            x = x[-1]
            # Assume log prob is in zero spot
            delta_logprob, reg, x = self.augmentations(x)
            logprob = self.prior.log_prob(x).to(x) - delta_logprob
            losses.append(-torch.mean(logprob))
            # negative because we are integrating backwards
            regs.append(-reg)
            # Predicted locations
        reg = torch.mean(torch.stack(regs))
        loss = torch.mean(torch.stack(losses))
        return reg, loss
