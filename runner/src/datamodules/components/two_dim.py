# Adapted from From DSB
# https://github.com/JTT94/diffusion_schrodinger_bridge/blob/main/bridge/data/two_dim.py
import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import TensorDataset

# checker/pinwheel/8gaussians can be found at
# https://github.com/rtqichen/ffjord/blob/994864ad0517db3549717c25170f9b71e96788b1/lib/toy_data.py#L8


def data_distrib(npar, data, random_state=42):
    np.random.seed(random_state)

    if data == "mixture":
        init_sample = torch.randn(npar, 2)
        p = init_sample.shape[0] // 2
        init_sample[:p, 0] = init_sample[:p, 0] - 7.0
        init_sample[p:, 0] = init_sample[p:, 0] + 7.0

    if data == "scurve":
        X, y = datasets.make_s_curve(n_samples=npar, noise=0.1, random_state=None)
        init_sample = torch.tensor(X)[:, [0, 2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor

    if data == "swiss":
        X, y = datasets.make_swiss_roll(n_samples=npar, noise=0.1, random_state=None)
        init_sample = torch.tensor(X)[:, [0, 2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor

    if data == "moon":
        X, y = datasets.make_moons(n_samples=npar, noise=0.1, random_state=None)
        scaling_factor = 7.0
        init_sample = torch.tensor(X)
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor

    if data == "circle":
        X, y = datasets.make_circles(n_samples=npar, noise=0.0, random_state=None, factor=0.5)
        init_sample = torch.tensor(X) * 10

    if data == "checker":
        x1 = np.random.rand(npar) * 4 - 2
        x2_ = np.random.rand(npar) - np.random.randint(0, 2, npar) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 7.5
        init_sample = torch.from_numpy(x)

    if data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = npar // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        x = 7.5 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        init_sample = torch.from_numpy(x)

    if data == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(npar):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset *= 3
        init_sample = torch.from_numpy(dataset)

    init_sample = init_sample.float()

    return init_sample


def two_dim_ds(npar, data_tag):
    init_sample = data_distrib(npar, data_tag)
    init_ds = TensorDataset(init_sample)
    return init_ds
