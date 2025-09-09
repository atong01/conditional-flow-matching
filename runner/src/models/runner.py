from typing import Any, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule

from torchcfm import ConditionalFlowMatcher

from .components.augmentation import AugmentationModule
from .components.distribution_distances import compute_distribution_distances
from .components.plotting import plot_trajectory, store_trajectories
from .components.solver import FlowSolver
from .utils import get_wandb_logger


class CFMLitModule(LightningModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        flow_matcher: ConditionalFlowMatcher,
        solver: FlowSolver,
        scheduler: Optional[Any] = None,
        plot: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "flow_matcher",
                "solver",
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
        self.solver = solver
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.scheduler = scheduler
        self.criterion = torch.nn.MSELoss()
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if not isinstance(self.dim, int):
            # Assume this is an image classification dataset where we need to strip the targets
            return batch[0]
        return batch

    def preprocess_batch(self, batch, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        X = self.unpack_batch(batch)
        # If no trajectory assume generate from standard normal
        x0 = torch.randn_like(X)
        x1 = X
        return x0, x1

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        x0, x1 = self.preprocess_batch(batch, training)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self.net(t, xt)
        return torch.nn.functional.mse_loss(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch, training=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        loss = self.step(batch, training=True)
        self.log(f"{prefix}/loss", loss)
        return {"loss": loss, "x": batch}

    def preprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
        x = v["x"]

        # Sample some random points for the plotting function
        rand = torch.randn_like(x)
        x = torch.stack([rand, x], dim=1)
        ts = x.shape[1]
        x0 = x[:, 0]
        x_rest = x[:, 1:]
        return ts, x, x0, x_rest

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a trajectory
        t_span = torch.linspace(0, 1, 101)
        solver = self.solver(self.net, self.dim)
        solver.augmentations = self.val_augmentations
        traj, aug = solver.odeint(x0, t_span)
        full_trajs = [traj]
        traj, aug = traj[-1], aug[-1]
        regs = [torch.mean(aug, dim=0).detach().cpu().numpy()]
        trajs = [traj]
        nfe = solver.nfe
        full_trajs = torch.cat(full_trajs)

        regs = np.stack(regs).mean(axis=0)
        names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
        self.log_dict(dict(zip(names, regs)), sync_dist=True)

        names, dists = compute_distribution_distances(trajs, x_rest)
        names = [f"{prefix}/{name}" for name in names]
        d = dict(zip(names, dists))
        d[f"{prefix}/nfe"] = nfe
        self.log_dict(d, sync_dist=True)
        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
        trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix)

        if self.hparams.plot:
            plot_trajectory(
                x,
                full_trajs,
                title=f"{self.current_epoch}_ode",
                key="ode_path",
                wandb_logger=wandb_logger,
            )
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
