import pytest
import torch

from src.datamodules.distribution_datamodule import (
    SKLearnDataModule,
    TorchDynDataModule,
    TwoDimDataModule,
)


@pytest.mark.parametrize("batch_size", [32, 128])
@pytest.mark.parametrize("train_val_test_split", [400, [1000, 100, 100]])
@pytest.mark.parametrize(
    "datamodule,system",
    [
        (SKLearnDataModule, "scurve"),
        (SKLearnDataModule, "moons"),
        (TorchDynDataModule, "gaussians"),
    ],
)
def test_single_datamodule(batch_size, train_val_test_split, datamodule, system):
    dm = datamodule(
        batch_size=batch_size, train_val_test_split=train_val_test_split, system=system
    )

    assert dm.data_train is not None and dm.data_val is not None and dm.data_test is not None
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 1200

    batch = next(iter(dm.train_dataloader()))
    x = batch
    assert x.dim() == 2
    assert x.shape[0] == batch_size
    assert x.shape[-1] == 2
    assert dm.dim == 2
    assert x.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [32, 128])
@pytest.mark.parametrize("train_val_test_split", [300, [200, 50, 50]])
@pytest.mark.parametrize(
    "datamodule,system",
    [
        (TwoDimDataModule, "moon-8gaussians"),
    ],
)
def test_trajectory_datamodule(batch_size, train_val_test_split, datamodule, system):
    dm = datamodule(
        batch_size=batch_size, train_val_test_split=train_val_test_split, system=system
    )
    # assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x = batch
    assert len(x) == 2
    for t in range(len(dm.timepoint_data)):
        xt = x[t]
        assert xt.dim() == 2
        assert xt.shape[0] == batch_size
        assert xt.shape[-1] == 2
        assert xt.dtype == torch.float32
    assert dm.dim == 2
