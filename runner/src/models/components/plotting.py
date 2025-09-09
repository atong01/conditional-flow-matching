import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scprep
import torch


def plot_scatter(obs, model, title="fig", wandb_logger=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    batch_size, ts, dim = obs.shape
    obs = obs.reshape(-1, dim).detach().cpu().numpy()
    ts = np.tile(np.arange(ts), batch_size)
    scprep.plot.scatter2d(obs, c=ts, ax=ax)
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    if wandb_logger:
        wandb_logger.log_image(key=title, images=[f"figs/{title}.png"])
    plt.close()


def plot_scatter_and_flow(obs, model, title="stream", wandb_logger=None):
    batch_size, ts, dim = obs.shape
    device = obs.device
    obs = obs.reshape(-1, dim).detach().cpu().numpy()
    diff = obs.max() - obs.min()
    wmin = obs.min() - diff * 0.1
    wmax = obs.max() + diff * 0.1
    points = 50j
    points_real = 50
    Y, X, T = np.mgrid[wmin:wmax:points, wmin:wmax:points, 0 : ts - 1 : 7j]
    gridpoints = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), requires_grad=True, device=device
    ).type(torch.float32)
    times = torch.tensor(T.flatten(), requires_grad=True, device=device).type(torch.float32)[
        :, None
    ]
    out = model(times, gridpoints)
    out = out.reshape([points_real, points_real, 7, dim])
    out = out.cpu().detach().numpy()
    # Stream over time
    fig, axes = plt.subplots(1, 7, figsize=(20, 4), sharey=True)
    axes = axes.flatten()
    tts = np.tile(np.arange(ts), batch_size)
    for i in range(7):
        scprep.plot.scatter2d(obs, c=tts, ax=axes[i])
        axes[i].streamplot(
            X[:, :, 0],
            Y[:, :, 0],
            out[:, :, i, 0],
            out[:, :, i, 1],
            color=np.sum(out[:, :, i] ** 2, axis=-1),
        )
        axes[i].set_title(f"t = {np.linspace(0,ts-1,7)[i]:0.2f}")
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="flow", images=[f"figs/{title}.png"])


def store_trajectories(obs: Union[torch.Tensor, list], model, title="trajs", start_time=0):
    n = 2000
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        # For consistency with DSB
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, 20 * (ts - 1)))
        traj = traj.cpu().detach().numpy()
        os.makedirs("figs", exist_ok=True)
        np.save(f"figs/{title}.npy", traj)


def plot_trajectory(
    obs: Union[torch.Tensor, list],
    traj: torch.Tensor,
    title="traj",
    key="traj",
    start_time=0,
    n=200,
    wandb_logger=None,
):
    plt.figure(figsize=(6, 6))
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
        scprep.plot.scatter2d(obs, c=tts)
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c="purple", marker="x")
    for i in range(20):
        plt.plot(traj[:, i, 0], traj[:, i, 1], c="red", alpha=0.5)
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key=key, images=[f"figs/{title}.png"])


def plot_paths(
    obs: Union[torch.Tensor, list],
    model,
    title="paths",
    start_time=0,
    n=200,
    wandb_logger=None,
):
    plt.figure(figsize=(6, 6))
    if isinstance(obs, list):
        data, labels = [], []
        for t, xi in enumerate(obs):
            xi = xi.detach().cpu().numpy()
            data.append(xi)
            labels.append(t * np.ones(xi.shape[0]))
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        scprep.plot.scatter2d(data, c=labels)
        start = obs[0][:n]
        ts = len(obs)
    else:
        batch_size, ts, dim = obs.shape
        start = obs[:n, start_time, :]
        obs = obs.reshape(-1, dim).detach().cpu().numpy()
        tts = np.tile(np.arange(ts), batch_size)
        scprep.plot.scatter2d(obs, c=tts)
    from torchdyn.core import NeuralODE

    with torch.no_grad():
        node = NeuralODE(model)
        traj = node.trajectory(start, t_span=torch.linspace(0, ts - 1, max(20 * ts, 100)))
        traj = traj.cpu().detach().numpy()
    # plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.3, alpha=0.2, c="black", label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=6, alpha=1, c="purple", marker="x")
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{title}.png")
    plt.close()
    if wandb_logger:
        wandb_logger.log_image(key="paths", images=[f"figs/{title}.png"])


def plot_samples(trajs, title="samples", wandb_logger=None):
    import PIL
    from torchvision.utils import save_image

    images = trajs[:100]
    os.makedirs("figs", exist_ok=True)
    save_image(images, fp=f"figs/{title}.jpg", nrow=10, normalize=True, padding=0)
    if wandb_logger:
        try:
            wandb_logger.log_image(key="paths", images=[f"figs/{title}.jpg"])
        except PIL.UnidentifiedImageError:
            print(f"ERROR logging {title}")
