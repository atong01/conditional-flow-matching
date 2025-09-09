import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "runner/src/train.py"
overrides = ["logger=[]"]
dir_overrides = ["paths.data_dir", "hydra.sweep.dir"]


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.xfail(
    reason="Currently failing experiments with fast_dev_run which messes with gradients"
)
def test_xfail_fast_dev_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = (
        [
            startfile,
            "-m",
            "experiment=glob(*)",
            "++trainer.fast_dev_run=true",
        ]
        + overrides
        + [f"{d}={tmp_path}" for d in dir_overrides]
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = (
        [
            startfile,
            "-m",
            "experiment=cfm",
            "model=cfm,otcfm,sbcfm,fm",
            "++trainer.fast_dev_run=true",
            "++trainer.limit_val_batches=0.25",
        ]
        + overrides
        + [f"{d}={tmp_path}" for d in dir_overrides]
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = (
        [
            startfile,
            "-m",
            "hydra.sweep.dir=" + str(tmp_path),
            "model.optimizer.lr=0.005,0.01",
            "++trainer.fast_dev_run=true",
        ]
        + overrides
        + [f"{d}={tmp_path}" for d in dir_overrides]
    )

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.xfail(reason="DDP is not working yet")
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = (
        [
            startfile,
            "-m",
            "trainer=ddp_sim",
            "trainer.max_epochs=3",
            "+trainer.limit_train_batches=0.01",
            "+trainer.limit_val_batches=0.1",
            "+trainer.limit_test_batches=0.1",
            "model.optimizer.lr=0.005,0.01,0.02",
        ]
        + overrides
        + [f"{d}={tmp_path}" for d in dir_overrides]
    )
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
@pytest.mark.skip(reason="Too slow for easy esting, pathway currently not used")
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = (
        [
            startfile,
            "-m",
            "hparams_search=optuna",
            "hydra.sweep.dir=" + str(tmp_path),
            "hydra.sweeper.n_trials=3",
            "hydra.sweeper.sampler.n_startup_trials=2",
            # "++trainer.fast_dev_run=true",
        ]
        + overrides
        + [f"{d}={tmp_path}" for d in dir_overrides]
    )
    run_sh_command(command)


@RunIf(wandb=True, sh=True)
@pytest.mark.slow
@pytest.mark.xfail(reason="wandb import is still bad without API key")
def test_optuna_sweep_ddp_sim_wandb(tmp_path):
    """Test optuna sweep with wandb and ddp sim."""
    command = [
        startfile,
        "-m",
        "hparams_search=optuna",
        "hydra.sweeper.n_trials=5",
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=0.01",
        "+trainer.limit_val_batches=0.1",
        "+trainer.limit_test_batches=0.1",
        "logger=wandb",
    ] + [f"{d}={tmp_path}" for d in dir_overrides]
    run_sh_command(command)
