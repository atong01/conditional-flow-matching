import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_trajectories(data, pred, graph, dataset, title=[1, 2.1]):
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.3))
    fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
    assert data.shape[-1] == pred.shape[-1]
    for i in range(data.shape[-1]):
        axs[0].plot(data[0, :, i].squeeze())
        axs[1].plot(pred[0, :, i].squeeze())
    title = f"{dataset}: Epoch = {title[0]}, Loss = {title[1]:1.3f}"
    axs[1].set_title(title)
    cax = axs[2].matshow(graph)
    fig.colorbar(cax)
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    plt.close()


def plot_graph_dist(graph_mu, graph_thresh, graph_std, ground_truth, path):
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.5))
    # fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
    axs[0].set_title("Ground Truth")
    axs[1].set_title("Graph means")
    axs[2].set_title("Graph post-threshold")
    axs[3].set_title("Graph std")

    print(graph_mu.shape, ground_truth.shape)

    g = [ground_truth, graph_mu, graph_thresh, graph_std]
    for col in range(4):
        ax = axs[col]
        pcm = ax.matshow(g[col], cmap="viridis")
        fig.colorbar(pcm, ax=ax)

    if not os.path.exists(path + "/figs"):
        os.mkdir(path + "/figs")
    plt.savefig(f"{path}/figs/graph_dist_plot.png")
    plt.close()


def plot_traj_dist(data, pred, dataset, title=[1, 2.1]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.3))
    fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
    assert data.shape[-1] == pred.shape[-1]
    for i in range(data.shape[-1]):
        axs[0].plot(data[0, :, i].squeeze())
        axs[1].plot(pred[0, :, i].squeeze())
    title = f"{dataset}: Epoch = {title[0]}, Loss = {title[1]:1.3f}"
    axs[1].set_title(title)
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    plt.close()


def plot_cnf(data, traj, graph, dataset, title):
    n = 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    data = data.reshape([-1, *data.shape[2:]])
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5)
    ax.scatter(traj[:n, -1, 0], traj[:n, -1, 1], s=10, alpha=0.8, c="black")
    # ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    # ax.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    ax.scatter(traj[:n, :, 0], traj[:n, :, 1], s=0.2, alpha=0.2, c="olive")
    # ax.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    ax.scatter(traj[:n, 0, 0], traj[:n, 0, 1], s=4, alpha=1, c="blue")
    ax.legend(["data", "Last Timepoint", "Flow", "Posterior"])

    ax = axes[1]
    cax = ax.matshow(graph)
    fig.colorbar(cax)
    title = f"{dataset}: Epoch = {title[0]}, Loss = {title[1]:1.3f}"
    ax.set_title(title)
    if not os.path.exists("figs"):
        os.mkdir("figs")
    plt.savefig(f"figs/{title}.png")
    plt.close()


def plot_pca_traj(data, traj, graph, adata, dataset, title):
    """
    Args:
        data: np.array [N, T, D]
        traj: np.array [N, T, D]
        graph: np.array [D, D]
    """
    n = 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    # data = data.reshape([-1, *data.shape[2:]])

    def pca_transform(x, d=2):
        return (x - adata.var["means"].values) @ adata.varm["PCs"][:, :d]

    traj = pca_transform(traj)

    for t in range(data.shape[1]):
        pcd = pca_transform(data[:, t])
        ax.scatter(pcd[:, 0], pcd[:, 1], alpha=0.5)
    ax.scatter(traj[:n, -1, 0], traj[:n, -1, 1], s=10, alpha=0.8, c="black")
    ax.scatter(traj[:n, :, 0], traj[:n, :, 1], s=0.2, alpha=0.2, c="olive")
    ax.scatter(traj[:n, 0, 0], traj[:n, 0, 1], s=4, alpha=1, c="blue")
    ax.legend(
        [
            *[f"T={i}" for i in range(data.shape[1])],
            "Last Timepoint",
            "Flow",
            "Posterior",
        ]
    )

    ax = axes[1]
    cax = ax.matshow(graph)
    fig.colorbar(cax)
    title = f"{dataset}: Epoch = {title[0]}, Loss = {title[1]:1.3f}"
    ax.set_title(title)
    if not os.path.exists("figs_pca"):
        os.mkdir("figs_pca")
    plt.savefig(f"figs_pca/{title}.png")
    np.save(f"figs_pca/{title}.npy", graph)
    plt.close()


def to_torch(arr):
    if isinstance(arr, list):
        return torch.tensor(np.array(arr)).float()
    elif isinstance(arr, (np.ndarray, np.generic)):
        return torch.tensor(arr).float()
    else:
        raise NotImplementedError(f"to_torch not implemented for type: {type(arr)}")
