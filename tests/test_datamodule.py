from pathlib import Path

import pytest
import torch

from src.datamodules.distribution_datamodule import TwoDimDataModule, SKLearnDataModule, TorchDynDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
@pytest.mark.parametrize("train_val_test_split", [300, [200, 50, 50]])
@pytest.mark.parametrize("datamodule", [TwoDimDataModule, SKLearnDataModule, TorchDynDataModule])
def test_datamodule(batch_size, train_val_test_split, datamodule):
    data_dir = "data/"

    dm = datamodule(data_dir=data_dir, batch_size=batch_size, train_val_test_split=train_val_test_split)

    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 300

    batch = next(iter(dm.train_dataloader()))
    x = batch
    assert x.dim() == 3
    assert x.shape[0] == batch_size
    assert x.shape[-1] == 2
    assert dm.dim == 2
    assert x.dtype == torch.float32
