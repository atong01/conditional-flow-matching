import numpy as np
import scanpy as sc
import torch
from functools import partial
from torch.utils.data import random_split

from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader

def adata_dataset(path, embed_name="X_pca", label_name="sample_labels", max_dim=100):
    adata = sc.read_h5ad(path)
    labels = adata.obs[label_name].astype("category")
    ulabels = labels.cat.categories
    return adata.obsm[embed_name][:, :max_dim], labels, ulabels

def split(timepoint_data):
    """split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
    train_val_test_split = [0.8, 0.1, 0.1]
    if isinstance(train_val_test_split, int):
        split_timepoint_data = list(map(lambda x: (x, x, x), timepoint_data))
        return split_timepoint_data
    splitter = partial(
        random_split,
        lengths=train_val_test_split,
        generator=torch.Generator().manual_seed(42),
    )
    split_timepoint_data = list(map(splitter, timepoint_data))
    return split_timepoint_data

def combined_loader(split_timepoint_data, index, shuffle=False, load_full=False):
    tp_dataloaders = [
        DataLoader(
            dataset=datasets[index],
            batch_size=128,
            shuffle=shuffle,
            drop_last=True,
        )
        for datasets in split_timepoint_data
    ]
    return CombinedLoader(tp_dataloaders, mode="min_size")

def train_dataloader(split_timepoint_data):
    return combined_loader(split_timepoint_data, 0, shuffle=True)

def val_dataloader(split_timepoint_data):
    return combined_loader(split_timepoint_data, 1, shuffle=False, load_full=False)

def test_dataloader(split_timepoint_data):
    return combined_loader(split_timepoint_data, 2, shuffle=False, load_full=True)