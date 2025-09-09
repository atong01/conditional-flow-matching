import copy
import math
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms

from .components.augmentation import (
    AugmentationModule,
    AugmentedVectorField,
    Sequential,
)
from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import OTPlanSampler
from .components.plotting import (
    plot_samples,
    plot_trajectory,
    store_trajectories,
)
from .components.schedule import ConstantNoiseScheduler, NoiseScheduler
from .components.solver import FlowSolver
from .utils import get_wandb_logger


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        test_nfe: int = 100,
        plot: bool = False,
        nice_name: str = "CFM",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.

        Note: DDP does not currently work with NeuralODE objects from torchdyn
        in the init so we initialize them every time we need to do a sampling
        step.

        Args:
            net: torch module representing dx/dt = f(t, x) for t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "dim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is used in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            leaveout_timepoint: which (if any) timepoint to leave out during the training phase
            plot: if true, log intermediate plots during validation
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net(dim=self.dim)
        self.augmentations = augmentations
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs, self.dim)
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs, self.dim)
        if neural_ode is not None:
            self.aug_node = Sequential(
                self.augmentations.augmenter,
                neural_ode(self.aug_net),
            )

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        if not isinstance(self.dim, int):
            # Assume this is an image classification dataset where we need to strip the targets
            return batch[0]
        return batch

    def preprocess_batch(self, X, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X.device)
        if self.is_trajectory:
            batch_size, times, dim = X.shape
            if not hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                # resample the OT plan
                # list of length t of tuples of length 2 of tensors of shape
                tmp_ot_list = []
                for t in range(times - 1):
                    if training and t + 1 == self.hparams.leaveout_timepoint:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 2]))
                    else:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 1]))
                    if (
                        training
                        and self.ot_sampler is not None
                        and t != self.hparams.leaveout_timepoint
                    ):
                        tmp_ot = torch.stack(self.ot_sampler.sample_plan(tmp_ot[0], tmp_ot[1]))

                    tmp_ot_list.append(tmp_ot)
                tmp_ot_list = torch.stack(tmp_ot_list)
                # randomly sample a batch

            if training and self.hparams.leaveout_timepoint > 0:
                # Select random except for the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,), device=X.device)
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
                if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                    x0.append(torch.tensor(self.datamodule.timepoint_data[ti][X[i, ti]]))
                    pi = self.datamodule.pi[ti]
                    if training and ti + 1 == self.hparams.leaveout_timepoint:
                        pi = self.datamodule.pi_leaveout[ti]
                    index_batch = X[i][ti]
                    i_next = np.random.choice(
                        pi.shape[1], p=pi[index_batch] / pi[index_batch].sum()
                    )
                    x1.append(torch.tensor(self.datamodule.timepoint_data[ti_next][i_next]))
                else:
                    x0.append(tmp_ot_list[ti][0][i])
                    x1.append(tmp_ot_list[ti][1][i])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size = X.shape[0]
            # If no trajectory assume generate from standard normal
            x0 = torch.randn_like(X)
            x1 = X
        return x0, x1, t_select

    def average_ut(self, x, t, mu_t, sigma_t, ut):
        pt = torch.exp(-0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2))
        batch_size = x.shape[0]
        ind = torch.randint(
            batch_size, size=(batch_size, self.hparams.avg_size - 1)
        )  # randomly (non-repreat) sample m-many index
        # always include self
        ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
        pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
        ut_sub = torch.stack([ut[ind[i]] for i in range(batch_size)])
        p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
        ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum
        # Reduce batch size because they are all the same
        return x[:1], ut[:1], t[:1]

    def calc_mu_sigma(self, x0, x1, t):
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x, t, mu_t, sigma_t
        return x1 - x0

    def calc_loc_and_target(self, x0, x1, t, t_select, training):
        """Computes the loss on a batch of data."""
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t, sigma_t = self.calc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + sigma_t * eps_t
        ut = self.calc_u(x0, x1, x, t_xshape, mu_t, sigma_t)

        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note that this has to be torch.cdist(x, mu) in that order
        # t that network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.shape[1:])
        return x, ut, t, mu_t, sigma_t, eps_t

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None and not self.is_trajectory:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        x, ut, t, mu_t, sigma_t, eps_t = self.calc_loc_and_target(x0, x1, t, t_select, training)

        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)
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

    def image_eval_step(self, batch: Any, batch_idx: int, prefix: str):
        import os

        from torchvision.utils import save_image

        #        val_augmentations = AugmentationModule(
        #            cnf_estimator="hutch",
        #            squared_l2_reg=1,
        #        )
        #        aug_dims = val_augmentations.aug_dims
        #        val_aug_net = AugmentedVectorField(self.net, val_augmentations.regs, self.dim)
        #        val_aug_node = Sequential(
        #            val_augmentations.augmenter,
        #            NeuralODE(val_aug_net, solver="euler", sensitivity="adjoint"),
        #        )
        #        t_span = torch.linspace(1, 0, 101)
        #        x = batch[0]
        #        os.makedirs("regularizations", exist_ok=True)
        #        for k in range(0):
        #            x_norm = cifar10_normalization()(x + (torch.rand_like(x) / 255))
        #            _, aug_traj = val_aug_node(x_norm, t_span)
        #            aug, traj = aug_traj[-1, :, :aug_dims], aug_traj[-1, :, aug_dims:]
        #            mn = MultivariateNormal(
        #                torch.zeros(prod(self.dim)).type_as(traj),
        #                torch.eye(prod(self.dim)).type_as(traj),
        #            )
        #            aug[:, 0] += mn.log_prob(traj.reshape(traj.shape[0], -1))
        #            np.save(
        #                f"regularizations/regs_{k}_{batch_idx}.npy",
        #                aug.detach().cpu().numpy(),
        #            )

        solver = self.partial_solver(self.net, self.dim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 2)
        else:
            raise NotImplementedError(f"Unknown test procedure {self.hparams.test_nfe}")
        traj = solver.odeint(torch.randn(batch[0].shape[0], *self.dim).type_as(batch[0]), t_span)[
            -1
        ]
        os.makedirs("images", exist_ok=True)
        mean = [-x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [255.0 / x for x in [63.0, 62.1, 66.7]]
        inv_normalize = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=std),
                transforms.Normalize(mean=mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        traj = inv_normalize(traj)
        traj = torch.clip(traj, min=0, max=1.0)
        for i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        return {"x": batch[0]}

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        if prefix == "test" and self.is_image:
            self.image_eval_step(batch, batch_idx, prefix)
        shapes = [b.shape[0] for b in batch]

        if not self.is_image and prefix == "val" and shapes.count(shapes[0]) == len(shapes):
            reg, mse = self.step(batch, training=False)
            loss = mse + reg
            self.log_dict(
                {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}

        return {"x": batch}

    def preprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            # x is jagged if doing a trajectory
            x = outputs[0]["x"]
            ts = len(x)
            x0 = x[0]
            x_rest = x[1:]
        elif self.is_trajectory:
            if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                x = [torch.tensor(dd) for dd in self.datamodule.timepoint_data]
                x0 = x[0]
                x_rest = x[1:]
                ts = len(x)
            else:
                v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
                x = v["x"]
                ts = x.shape[1]
                x0 = x[:, 0, :]
                x_rest = x[:, 1:]
        else:
            if isinstance(self.dim, int):
                v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
                x = v["x"]
            else:
                x = [d["x"] for d in outputs][0][0][:100]
            # Sample some random points for the plotting function
            rand = torch.randn_like(x)
            # rand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
            x = torch.stack([rand, x], dim=1)
            ts = x.shape[1]
            x0 = x[:, 0]
            x_rest = x[:, 1:]
        return ts, x, x0, x_rest

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(self.net, self.dim)
        nfe = 0
        x0_tmp = x0.clone()

        if self.is_image:
            traj = solver.odeint(x0, t_span)
            full_trajs.append(traj)
            trajs.append(traj[0])
            trajs.append(traj[-1])
            nfe += solver.nfe

        if not self.is_image:
            solver.augmentations = self.val_augmentations
            for i in range(ts - 1):
                traj, aug = solver.odeint(x0_tmp, t_span + i)
                full_trajs.append(traj)
                traj, aug = traj[-1], aug[-1]
                x0_tmp = traj
                regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
                trajs.append(traj)
                nfe += solver.nfe

        full_trajs = torch.cat(full_trajs)

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the fit
            if (
                self.is_trajectory
                and prefix == "test"
                and isinstance(outputs[0]["x"], list)
                and not hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM")
            ):
                # Redo the solver for each timepoint
                trajs = []
                full_trajs = []
                nfe = 0
                x0_tmp = x0
                for i in range(ts - 1):
                    traj, _ = solver.odeint(x0_tmp, t_span + i)
                    traj = traj[-1]
                    x0_tmp = x_rest[i]
                    trajs.append(traj)
                    nfe += solver.nfe
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
            d[f"{prefix}/nfe"] = nfe

            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            # t_span = torch.linspace(0, 1, 101)
            # traj = solver.odeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            t_span = torch.linspace(0, 1, 21)
            traj = solver.odeint(x0, t_span)
            assert traj.shape[0] == t_span.shape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)
            ]
            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] for i in range(21)}, sync_dist=True)

        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        if prefix == "test" and self.is_image:
            os.makedirs("images", exist_ok=True)
            if len(os.listdir("images")) > 0:
                path = "/home/mila/a/alexander.tong/scratch/trajectory-inference/data/fid_stats_cifar10_train.npz"
                from pytorch_fid import fid_score

                fid = fid_score.calculate_fid_given_paths(["images", path], 256, "cuda", 2048, 0)
                self.log(f"{prefix}/fid", fid)

        ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
        trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix)

        if self.hparams.plot:
            if isinstance(self.dim, int):
                plot_trajectory(
                    x,
                    full_trajs,
                    title=f"{self.current_epoch}_ode",
                    key="ode_path",
                    wandb_logger=wandb_logger,
                )
            else:
                plot_samples(
                    trajs[-1],
                    title=f"{self.current_epoch}_samples",
                    wandb_logger=wandb_logger,
                )

        if prefix == "test" and not self.is_image:
            store_trajectories(x, self.net)

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
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)


class RectifiedFlowLitModule(CFMLitModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        partial_solver: FlowSolver,
        val_augmentations: Optional[AugmentationModule] = None,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        rectify_epochs: Optional[List[int]] = None,
        test_nfe: int = 100,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        plot: bool = False,
        nice_name: str = "Rect",
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
            plot: if true, log intermediate plots during validation
        """
        super(CFMLitModule, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "val_augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net(dim=self.dim)
        self.frozen_net = None
        self.augmentations = augmentations
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs, self.dim)
        self.val_augmentations = val_augmentations
        if val_augmentations is None:
            self.val_augmentations = AugmentationModule(
                l1_reg=1,
                l2_reg=1,
                squared_l2_reg=1,
            )
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs, self.dim)
        if neural_ode is not None:
            self.aug_node = Sequential(
                self.augmentations.augmenter,
                neural_ode(self.aug_net),
            )
        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

    def preprocess_batch(self, X, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X.device)
        if self.is_trajectory:
            batch_size, times, dim = X.shape
            if training and self.hparams.leaveout_timepoint > 0:
                # Select random except for the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,), device=X.device)
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
            batch_size = X.shape[0]
            # If no trajectory assume generate from standard normal
            x0 = torch.randn_like(X)
            x1 = X

        if self.frozen_net is not None:
            # Currently only works for 2 distributions
            assert t_select[0] == 0
            t_span = torch.linspace(0, 1, 100)
            val_node = NeuralODE(self.frozen_net, solver="euler")
            with torch.no_grad():
                _, traj = val_node(x0, t_span)
                x1 = traj[-1]
        return x0, x1, t_select

    def training_epoch_end(self, training_step_outputs):
        if (
            self.hparams.rectify_epochs is not None
            and self.current_epoch in self.hparams.rectify_epochs
        ):
            self.frozen_net = copy.deepcopy(self.net)


class ActionMatchingLitModule(CFMLitModule):
    """Implements Action Matching: Learning Stochastic Dynamics from Samples (Neklyudov et al.
    2022)

    Requires net to have a .energy function where net.energy(t, x): \\mathbb{R}^{d+1} \to
    \\mathbb{R} and net.forward is equal to \nabla_x(net.energy).
    """

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        assert not self.is_trajectory
        energy = self.net.energy
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)

        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        t = torch.rand(X.shape[0]).type_as(X)
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        xt = t_xshape * x1 + (1 - t_xshape) * x0
        # t that network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.shape[1:])

        xt.requires_grad, t_xshape.requires_grad = True, True
        with torch.set_grad_enabled(True):
            st = torch.sum(energy(torch.cat([xt, t_xshape], dim=-1)))
            dsdx, dsdt = torch.autograd.grad(st, (xt, t_xshape), create_graph=True)
        xt.requires_grad, t_xshape.requires_grad = False, False
        a0 = energy(torch.cat([x0, torch.zeros(x0.shape[0], 1)], dim=-1))
        a1 = energy(torch.cat([x1, torch.ones(x1.shape[0], 1)], dim=-1))
        loss = a0 - a1 + 0.5 * (dsdx**2).sum(1, keepdims=True) + dsdt
        loss = loss.mean()
        aug_x = self.aug_net(t, xt, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), loss


class VariancePreservingCFM(CFMLitModule):
    """Implements a variance preserving time schedule as suggested in (Albergo et al.

    2023) here we have an interpolation cos(t pi/2) x_0 + sin(t pi/2) x_1.
    """

    def calc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        mu_t = torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1
        sigma_t = self.hparams.sigma_min
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x, mu_t, sigma_t
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


class SBCFMLitModule(CFMLitModule):
    """Implements a Schrodinger Bridge based conditional flow matching model.

    This is similar to the OTCFM loss, however with the variance varying with t*(1-t). This has
    provably equal probability flow to the Schrodinger bridge solution when the transport is
    computed with the squared Euclidean distance on R^d.
    """

    def calc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min * torch.sqrt(t - t**2)
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del sigma_t
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (x - mu_t) + x1 - x0
        return ut


class SF2MLitModule(CFMLitModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        partial_solver: FlowSolver,
        score_net: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma: Optional[NoiseScheduler] = None,
        sigma_min: float = 0.1,
        outer_loop_epochs: Optional[int] = None,
        score_weight: float = 1.0,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        test_nfe: int = 100,
        test_sde: bool = False,
        plot: bool = False,
        nice_name: Optional[str] = "SF2M",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.

        Args:
            net: torch module representing dx/dt = f(t, x) for t in [1, T] missing dimension.
            score_net: torch module representing the score function of the flow.
            If not supplied it is assumed that the net contains both flow and
            score.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "dim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is used in minibatch.
            sigma: sigma determines the width of the Gaussian smoothing of the data and interpolations.
            leaveout_timepoint: which (if any) timepoint to leave out during the training phase
            plot: if true, log intermediate plots during validation
        """
        super(CFMLitModule, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "sigma_scheduler",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net(dim=self.dim)
        self.separate_score = score_net is not None
        self.score_net = score_net
        if self.separate_score:
            self.score_net = score_net(dim=self.dim)
        self.partial_solver = partial_solver
        self.augmentations = augmentations
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs, self.dim)
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs, self.dim)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sigma = sigma
        if sigma is None:
            self.sigma = ConstantNoiseScheduler(sigma_min)
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * self.sigma.F(1))
        self.criterion = torch.nn.MSELoss()

        # If we are doing outer loops holds the current dataset
        self.stored_data = None
        self.tmp_stored_data = None

    def calc_mu_sigma(self, x0, x1, t):
        # assert not self.is_trajectory
        ft = self.sigma.F(t)
        fone = self.sigma.F(1)
        mu_t = x0 + (x1 - x0) * ft / fone
        # Note this is slightly different than the notebook. Which is correct?
        sigma_t = torch.sqrt(ft - ft**2 / fone)
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        ft = self.sigma.F(t)
        fone = self.sigma.F(1)
        sigma_t_prime = self.sigma(t) ** 2 - 2 * ft * self.sigma(t) ** 2 / fone
        sigma_t_prime_over_sigma_t = sigma_t_prime / (sigma_t + 1e-8)
        mu_t_prime = (x1 - x0) * self.sigma(t) ** 2 / fone
        ut = sigma_t_prime_over_sigma_t * (x - mu_t) + mu_t_prime
        return ut

    def calc_loc_and_target(self, x0, x1, t, t_select, training):
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t, sigma_t = self.calc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + sigma_t * eps_t
        ut = self.calc_u(x0, x1, x, t_xshape, mu_t, sigma_t)

        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note that this has to be torch.cdist(x, mu) in that order
        # t that network sees is incremented by first timepoint
        score_target = eps_t
        # score_target = -eps_t * self.sigma(t_xshape) ** 2 / 2
        t = t + t_select.reshape(-1, *t.shape[1:])
        return x, ut, t, mu_t, sigma_t, score_target

    def forward_flow_and_score(self, t, x):
        if self.separate_score:
            reg, vt = self.augmentations(self.aug_net(t, x, augmented_input=False))
            st = self.score_net(t, x)
            return reg, vt, st
        reg, vtst = self.augmentations(self.aug_net(t, x, augmented_input=False))
        split_idx = vtst.shape[1] // 2
        vt, st = vtst[:, :split_idx], vtst[:, split_idx:]
        return reg, vt, st

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        t_orig = t.clone()

        x, ut, t, mu_t, sigma_t, score_target = self.calc_loc_and_target(
            x0, x1, t, t_select, training
        )

        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)

        reg, vt, st = self.forward_flow_and_score(t, x)
        flow_loss = self.criterion(vt, ut)
        score_loss = self.criterion(
            -sigma_t * st / (self.sigma(t_orig.reshape(sigma_t.shape)) ** 2) * 2,
            score_target,
        )
        return torch.mean(reg) + self.hparams.score_weight * score_loss, flow_loss

    def forward_sde_eval(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 2)
        solver = self.partial_solver(
            self.net, self.dim, score_field=self.score_net, sigma=self.sigma
        )
        if False and self.is_image:
            traj = solver.sdeint(x0, t_span, logqp=False)

        trajs = []
        full_trajs = []
        nfe = 0
        kldiv_total = 0
        x0_tmp = x0.clone()
        for i in range(ts - 1):
            traj, kldiv = solver.sdeint(x0_tmp, t_span + i, logqp=True)
            kldiv_total += torch.mean(kldiv[-1])
            x0_tmp = traj[-1]
            trajs.append(traj[-1])
            full_trajs.append(traj)
            nfe += solver.nfe
        full_trajs = torch.cat(full_trajs)
        if not self.is_image:
            # Evaluate the fit
            if (
                self.is_trajectory
                and prefix == "test"
                and isinstance(outputs[0]["x"], list)
                and not hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM")
            ):
                trajs = []
                full_trajs = []
                nfe = 0
                kldiv_total = 0
                x0_tmp = x0.clone()
                for i in range(ts - 1):
                    traj, kldiv = solver.sdeint(x0_tmp, t_span + i, logqp=True)
                    x0_tmp = x_rest[i]
                    kldiv_total += torch.mean(kldiv[-1])
                    trajs.append(traj[-1])
                    full_trajs.append(traj)
                    nfe += solver.nfe
                names, dists = compute_distribution_distances(trajs[:-1], x_rest[:-1])
            else:
                names, dists = compute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/sde/{name}" for name in names]
            d = dict(zip(names, dists))
            if self.hparams.leaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/sde/t_out/{key.split('/')[-1]}": val
                    for key, val in d.items()
                    if key.startswith(f"{prefix}/sde/t{self.hparams.leaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/sde/nfe"] = nfe
            d[f"{prefix}/sde/kldiv"] = kldiv_total
            self.log_dict(d, sync_dist=True)
        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)
            solver.dt = 0.05
            # solver.dt = 0.01
            traj = solver.sdeint(x0, t_span)
            assert traj.shape[0] == t_span.shape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)
            ]
            self.log_dict(
                {f"{prefix}/sde/kl/mean": torch.stack(kls).mean().item()},
                sync_dist=True,
            )
            self.log_dict({f"{prefix}/sde/kl/tp_{i}": kls[i] for i in range(21)}, sync_dist=True)
        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        super().eval_epoch_end(outputs, prefix)
        wandb_logger = get_wandb_logger(self.loggers)
        ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
        if isinstance(self.dim, int):
            traj, sde_traj = self.forward_sde_eval(ts, x0, x_rest, outputs, prefix)

        if self.hparams.plot:
            if isinstance(self.dim, int):
                plot_trajectory(
                    x,
                    sde_traj,
                    title=f"{self.current_epoch}_sde_traj",
                    key="sde",
                    wandb_logger=wandb_logger,
                )

    def preprocess_batch(self, X, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        if self.stored_data is not None and training:
            # Randomly sample a batch from the stored data.
            idx = torch.randint(self.stored_data.shape[0], size=(X.shape[0],))
            X = self.stored_data[idx]
            t_select = torch.zeros(1, device=X.device)
            return X[:, 0], X[:, 1], t_select
        return super().preprocess_batch(X, training)

    def training_step(self, batch: Any, batch_idx: int):
        # If we are doing outerloops we need to resample and store forward and backwards batches.
        if (
            self.hparams.outer_loop_epochs is not None
            and (self.current_epoch + 1) % self.hparams.outer_loop_epochs == 0
        ):
            X = self.unpack_batch(batch)
            x0, x1, t_select = self.preprocess_batch(X, training=True)
            assert not torch.any(t_select)  # resampling outerloop can only handle 2 timepoints
            solver = self.partial_solver
            t_span = torch.linspace(0, 1, 2)
            solver = self.partial_solver(
                self.net, self.dim, score_field=self.score_net, sigma=self.sigma
            )
            batch_size = x0.shape[0]
            with torch.no_grad():
                forward_traj = solver.sdeint(x0[: batch_size // 2], t_span)
                backward_traj = torch.flip(
                    solver.sdeint(x1[batch_size // 2 :], t_span, reverse=True), (0,)
                )
            stored_traj = torch.cat([forward_traj, backward_traj], dim=1)
            stored_traj = stored_traj.transpose(0, 1)
            if batch_idx == 0:
                self.tmp_stored_data = []
            self.tmp_stored_data.append(stored_traj)
        return super().training_step(batch, batch_idx)

    def training_epoch_end(self, training_step_outputs):
        if (
            self.hparams.outer_loop_epochs is not None
            and (self.current_epoch + 1) % self.hparams.outer_loop_epochs == 0
        ):
            self.stored_data = torch.cat(self.tmp_stored_data, dim=0).detach().clone()

    def image_eval_step(self, batch: Any, batch_idx: int, prefix: str):
        import os

        from torchvision.utils import save_image

        solver = self.partial_solver(self.net, self.dim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 2).type_as(batch[0])
        else:
            raise NotImplementedError(f"Unknown test procedure {self.hparams.test_nfe}")
        if self.hparams.test_sde:
            solver = self.partial_solver(
                self.net, self.dim, score_field=self.score_net, sigma=self.sigma
            )
            solver.dt = 1 / int(self.hparams.test_nfe)
            t_span = torch.linspace(0, 1, 2).type_as(batch[0])
            integrator = solver.sdeint
        else:
            integrator = solver.odeint
        x0 = torch.randn(5 * batch[0].shape[0], *self.dim).type_as(batch[0])
        traj = integrator(x0, t_span)[-1]
        os.makedirs("images", exist_ok=True)
        mean = [-x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [255.0 / x for x in [63.0, 62.1, 66.7]]
        inv_normalize = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=std),
                transforms.Normalize(mean=mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        traj = inv_normalize(traj)
        traj = torch.clip(traj, min=0, max=1.0)
        for i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        os.makedirs("compressed_images", exist_ok=True)
        torch.save(traj.cpu(), f"compressed_images/{batch_idx}.pt")
        return {"x": batch[0]}


class OneWaySF2MLitModule(SF2MLitModule):
    def calc_loc_and_target(self, x0, x1, t, t_select, training):
        x, ut, t, mu_t, sigma_t, score_target = super().calc_loc_and_target(
            x0, x1, t, t_select, training
        )
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        eps_t = -score_target * 2 / (self.sigma(t_xshape) ** 2)
        forward_target = (
            x1 - x0 - (self.sigma(t_xshape) * torch.sqrt(t_xshape / (1 - t_xshape + 1e-6))) * eps_t
        )
        return x, forward_target, t, mu_t, sigma_t, None

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        x, forward_target, t, _, _, _ = self.calc_loc_and_target(x0, x1, t, t_select, training)
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        forward_scaling = (1 + self.sigma(t_xshape) ** 2 * t_xshape / (1 - t_xshape + 1e-6)) ** -1
        reg, vt, st = self.forward_flow_and_score(t, x)
        forward_flow_loss = torch.mean(forward_scaling * (vt - forward_target) ** 2)
        return torch.mean(reg), forward_flow_loss

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 101).type_as(x0)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(
            self.net, self.dim, score_field=self.score_net, sigma=self.sigma
        )
        nfe = 0
        x0_tmp = x0.clone()
        for i in range(ts - 1):
            if not self.is_image:
                solver.augmentations = self.val_augmentations
                traj, aug = solver.sdeint(x0_tmp, t_span + i)
                aug = aug[-1]
                regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
            else:
                traj = solver.sdeint(x0_tmp, t_span + i)
            full_trajs.append(traj)
            traj = traj[-1]
            x0_tmp = traj
            trajs.append(traj)
            nfe += solver.nfe

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the fit
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
            d[f"{prefix}/nfe"] = nfe
            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)  # 101
            traj = solver.odeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            assert traj.shape[0] == t_span.shape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)
            ]
            # others = torch.stack([self.datamodule.detailed_evaluation(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)])

            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] for i in range(21)}, sync_dist=True)

        full_trajs = torch.cat(full_trajs)
        return trajs, full_trajs


class DSBMLitModule(SF2MLitModule):
    """Based on SF2M module except directly regresses against the target SDE drift rather than
    separating the ODE and Score components."""

    def calc_loc_and_target(self, x0, x1, t, t_select, training):
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1))).clone()
        x, ut, t_plus_t_select, mu_t, sigma_t, eps_t = super().calc_loc_and_target(
            x0, x1, t, t_select, training
        )
        forward_target = (
            x1 - x0 - (self.sigma(t_xshape) * torch.sqrt(t_xshape / (1 - t_xshape + 1e-6))) * eps_t
        )
        backward_target = (
            x0
            - x1
            - (self.sigma(t_xshape) * torch.sqrt((1 - t_xshape) / (t_xshape + 1e-6))) * eps_t
        )
        return x, forward_target, t_plus_t_select, mu_t, sigma_t, backward_target

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        forward_scaling = (1 + self.sigma(t) ** 2 * t / (1 - t + 1e-6)) ** -1
        backward_scaling = (1 + self.sigma(t) ** 2 * (1 - t) / (t + 1e-6)) ** -1
        x, forward_target, t, _, _, backward_target = self.calc_loc_and_target(
            x0, x1, t, t_select, training
        )
        # print(forward_target, backward_target, x0, x1, t, t_select)
        reg, vt, st = self.forward_flow_and_score(t, x)
        forward_flow_loss = torch.mean(forward_scaling[:, None] * (vt - forward_target) ** 2)
        backward_flow_loss = torch.mean(backward_scaling[:, None] * (st - backward_target) ** 2)
        if not torch.isfinite(forward_flow_loss) or not torch.isfinite(backward_flow_loss):
            raise ValueError("Loss Not Finite")

        return torch.mean(reg) + backward_flow_loss, forward_flow_loss

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(
            self.net, self.dim, score_field=self.score_net, sigma=self.sigma
        )
        nfe = 0
        x0_tmp = x0.clone()
        for i in range(ts - 1):
            if not self.is_image:
                solver.augmentations = self.val_augmentations
                traj, aug = solver.odeint(x0_tmp, t_span + i)
            else:
                traj = solver.odeint(x0_tmp, t_span + i)
            full_trajs.append(traj)
            if not self.is_image:
                traj, aug = traj[-1], aug[-1]
            else:
                traj = traj[-1]
                aug = torch.tensor(0.0)
            x0_tmp = traj
            regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
            trajs.append(traj)
            nfe += solver.nfe

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the fit
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
            d[f"{prefix}/nfe"] = nfe
            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)  # 101
            traj = solver.odeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            assert traj.shape[0] == t_span.shape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)
            ]
            # others = torch.stack([self.datamodule.detailed_evaluation(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)])

            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] for i in range(21)}, sync_dist=True)

        full_trajs = torch.cat(full_trajs)
        return trajs, full_trajs


class DSBMSharedLitModule(SF2MLitModule):
    """Based on SF2M module except directly regresses against the target SDE drift rather than
    separating the ODE and Score components."""

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        X = self.unpack_batch(batch)
        x0, x1, t_select = self.preprocess_batch(X, training)
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X.shape[0]).type_as(X)
        else:
            t = torch.rand(X.shape[0]).type_as(X)
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        x, ut, t, mu_t, sigma_t, score_target = self.calc_loc_and_target(
            x0, x1, t, t_select, training
        )

        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        forward_flow_loss = self.criterion(vt + sigma_t * self.score_net(t, x), ut + score_target)
        backward_flow_loss = self.criterion(
            -vt + sigma_t * self.score_net(t, x), -ut + score_target
        )
        # flow_loss = self.criterion(vt + sigma_t * self.score_net, ut + score_target)
        # score_loss = self.criterion(sigma_t * self.score_net(t, x), score_target)
        return torch.mean(reg) + backward_flow_loss, forward_flow_loss


class FMLitModule(CFMLitModule):
    """Implements a Lipman et al.

    2023 style flow matching loss.     This maps the standard normal distribution to the data
    distribution by using conditional flows     that are the optimal transport flow from a narrow
    Gaussian around a datapoint to a standard N(x     | 0, 1).
    """

    def calc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        del x0
        sigma_min = self.hparams.sigma_min
        mu_t = t * x1
        sigma_t = 1 - (1 - sigma_min) * t
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x0, mu_t, sigma_t
        sigma_min = self.hparams.sigma_min
        ut = (x1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)
        return ut


class SplineCFMLitModule(CFMLitModule):
    """Implements cubic spline version of OT-CFM."""

    def preprocess_batch(self, X, training=False):
        from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

        """Converts a batch of data into matched a random pair of (x0, x1)"""
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
            # ts, x = self.aug(reversed_ts[t:], obs[:, len(even_ts) - t - 1, :])
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
