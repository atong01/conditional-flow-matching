from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule
from torch import autograd

from .components.distribution_distances import compute_distribution_distances
from .utils import get_wandb_logger


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()


def plot(x, y, x_pred, y_pred, savename=None, wandb_logger=None):
    x = to_numpy(x)[:, 0]
    y = to_numpy(y)[:, 0]
    x_pred = to_numpy(x_pred)[:, 0]
    y_pred = to_numpy(y_pred)[:, 0]

    import matplotlib.pyplot as plt

    plt.scatter(y[:, 0], y[:, 1], color="C1", alpha=0.5, label=r"$Y$")
    plt.scatter(x[:, 0], x[:, 1], color="C2", alpha=0.5, label=r"$X$")
    plt.scatter(x_pred[:, 0], x_pred[:, 1], color="C3", alpha=0.5, label=r"$\nabla g(Y)$")
    plt.scatter(y_pred[:, 0], y_pred[:, 1], color="C4", alpha=0.5, label=r"$\nabla f(X)$")
    plt.legend()
    if savename:
        plt.savefig(savename)
    if wandb_logger:
        wandb_logger.log_image(key="match", images=[f"{savename}.png"])
    plt.close()


class ICNNLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        f_net: Any,
        g_net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        reg: int = 0.1,
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
        self.save_hyperparameters(ignore=["net", "optimizer", "datamodule"], logger=False)
        self.is_trajectory = datamodule.IS_TRAJECTORY
        self.dim = datamodule.dim
        self.f = f_net(dim=datamodule.dim)
        self.g = g_net(dim=datamodule.dim)
        self.optimizer = optimizer
        self.reg = reg
        self.criterion = torch.nn.MSELoss()

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if self.is_trajectory:
            return torch.stack(batch, dim=1)
        return batch

    def preprocess_batch(self, X):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1)
        if self.is_trajectory:
            batch_size, times, dim = X.shape
            if times > 2:
                raise NotImplementedError("ICNN not implemented for times > 2")
            t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            for i in range(batch_size):
                x0.append(X[i, t_select[i]])
                x1.append(X[i, t_select[i] + 1])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size, dim = X.shape
            # If no trajectory assume generate from standard normal
            x0 = torch.randn(batch_size, X.shape[1])
            x1 = X
        x0.requires_grad_()
        x1.requires_grad_()
        return x0, x1, t_select

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        X = self.unpack_batch(batch)
        x, y, t_select = self.preprocess_batch(X)

        if optimizer_idx == 0:
            fx = self.f(x)
            gy = self.g(y)
            grad_gy = torch.autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[
                0
            ]
            f_grad_gy = self.f(grad_gy)
            y_dot_grad_gy = torch.sum(torch.mul(y, grad_gy), axis=1, keepdim=True)
            loss = torch.mean(f_grad_gy - y_dot_grad_gy)
            if self.reg > 0:
                reg = self.reg * torch.sum(
                    torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in self.g.Wzs])
                )
                loss += reg
        if optimizer_idx == 1:
            fx = self.f(x)
            gy = self.g(y)
            grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[0]
            f_grad_gy = self.f(grad_gy)
            loss = torch.mean(fx - f_grad_gy)
            if self.reg > 0:
                reg = self.reg * torch.sum(
                    torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in self.f.Wzs])
                )
                loss += reg

        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        X = self.unpack_batch(batch)
        x, y, t_select = self.preprocess_batch(X)
        w2, f_loss, g_loss = compute_w2(self.f, self.g, x, y, return_loss=True)
        self.log_dict(
            {
                f"{prefix}/model_w2": w2,
                f"{prefix}/loss": w2,
                f"{prefix}/f_loss": f_loss,
                f"{prefix}/g_loss": g_loss,
            },
            on_step=False,
            on_epoch=True,
        )
        return {
            "loss": w2,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "x": self.unpack_batch(batch),
        }

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        def transport(model, x):
            return autograd.grad(torch.sum(model(x)), x)[0]

        def y_to_x(y):
            return transport(self.g, y)

        def x_to_y(x):
            return transport(self.f, x)

        v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
        x = v["x"]
        wandb_logger = get_wandb_logger(self.loggers)

        if not self.is_trajectory:
            # Sample some random points for the plotting function
            rand = torch.randn_like(x)
            x = torch.stack([rand, x], dim=1)
        x.requires_grad_()
        x0 = x[:, :1]
        x1 = x[:, 1:]
        pred = x_to_y(x0)
        _, dists = compute_distribution_distances(x0, pred)
        w1, w2 = dists[:2]
        self.log_dict({f"{prefix}/L2": w1, f"{prefix}/squared_L2": w2})

        # Evaluate the fit
        names, dists = compute_distribution_distances(pred, x[:, 1:])
        names = [f"{prefix}/{name}" for name in names]
        self.log_dict(dict(zip(names, dists)))

        x_pred = y_to_x(x1)
        plot(
            x0,
            x1,
            x_pred,
            pred,
            savename=f"{self.current_epoch}_match",
            wandb_logger=wandb_logger,
        )

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
        f_opt = self.optimizer(params=self.f.parameters())
        g_opt = self.optimizer(params=self.g.parameters())
        return [
            {"optimizer": g_opt, "frequency": 10},
            {"optimizer": f_opt, "frequency": 1},
        ]

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)


def compute_w2(f, g, x, y, return_loss=False):
    fx = f(x)
    gy = g(y)
    grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[0]

    f_grad_gy = f(grad_gy)
    y_dot_grad_gy = torch.sum(torch.multiply(y, grad_gy), axis=1, keepdim=True)

    x_squared = torch.sum(torch.pow(x, 2), axis=1, keepdim=True)
    y_squared = torch.sum(torch.pow(y, 2), axis=1, keepdim=True)

    w2 = torch.mean(f_grad_gy - fx - y_dot_grad_gy + 0.5 * x_squared + 0.5 * y_squared)
    if not return_loss:
        return w2
    g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
    f_loss = torch.mean(fx - f_grad_gy)
    return w2, f_loss, g_loss
