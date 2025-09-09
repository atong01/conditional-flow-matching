"""Random data generators.

Largely from
https://github.com/AmirTag/OT-ICNN/blob/6caa9b982596a101b90a8a947d10f35f18c7de4e/2_dim_experiments/W2-minimax-tf.py
"""

import random

import numpy as np
import sklearn


def generate_uniform_around_centers(centers, variance):
    num_center = len(centers)

    return centers[np.random.choice(num_center)] + variance * np.random.uniform(-1, 1, (2))


def generate_cross(centers, variance):
    num_center = len(centers)
    x = variance * np.random.uniform(-1, 1)
    y = (np.random.randint(2) * 2 - 1) * x

    return centers[np.random.choice(num_center)] + [x, y]


def sample_data(dataset, batch_size, scale, var):
    if dataset == "25gaussians":
        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        np.random.shuffle(dataset)
        # dataset /= 2.828 # stdev
        while True:
            for i in range(len(dataset) / batch_size):
                yield dataset[i * batch_size : (i + 1) * batch_size]

    elif dataset == "swissroll":
        while True:
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
            data = data.astype("float32")[:, [0, 2]]
            # data /= 7.5 # stdev plus a little
            yield data

    elif dataset == "8gaussians":
        scale = scale
        variance = var
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
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * variance
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_five":
        scale = scale
        variance = var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_four":
        scale = scale
        variance = var
        centers = scale * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpleGaussian":
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "unif_square":
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.uniform(-var, var, 2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpletranslatedGaussian":
        while True:
            dataset = []
            for i in range(batch_size):
                point = scale * np.array([1.0, 1.0]) + np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpletranslated_scaled_Gaussian":
        while True:
            dataset = []
            for i in range(batch_size):
                point = scale * np.array([1.0, 1.0]) + var * np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "circle-S1":
        while True:
            dataset = []
            for i in range(batch_size):
                angle = np.random.rand() * 2 * np.pi
                point = scale * np.array([np.cos(angle), np.sin(angle)])
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            yield dataset

    elif dataset == "semi-circle-S1":
        while True:
            dataset = []
            for i in range(batch_size):
                angle = np.random.rand() * np.pi
                point = scale * np.array([np.cos(angle), np.sin(angle)])
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            yield dataset

    elif dataset == "checker_board_five_cross":
        scale = scale
        variance = var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_cross(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_five_expanded":
        scale = scale
        variance = 2 * var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset
