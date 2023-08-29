<div align="center">

# Conditional Flow Matching

<!---[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

<!---[![contributors](https://img.shields.io/github/contributors/atong01/conditional-flow-matching.svg)](https://github.com/atong01/conditional-flow-matching/graphs/contributors) -->

[![OT-CFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2302.00482)
[![SF2M Preprint](http://img.shields.io/badge/paper-arxiv.2307.03672-B31B1B.svg)](https://arxiv.org/abs/2307.03672)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/atong01/conditional-flow-matching/actions/workflows/test.yaml/badge.svg)](https://github.com/atong01/conditional-flow-matching/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/atong01/conditional-flow-matching/branch/main/graph/badge.svg)](https://codecov.io/gh/atong01/conditional-flow-matching/)
[![code-quality](https://github.com/atong01/conditional-flow-matching/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/atong01/conditional-flow-matching/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/atong01/conditional-flow-matching#license)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

## Description

Conditional Flow Matching (CFM) is a fast way to train continuous normalizing flow (CNF) models. CFM is a simulation-free training objective for continuous normalizing flows that allows conditional generative modeling and speeds up training and inference.

This repository contains the code to reproduce the main experiments and illustrations of two preprints:

- [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482). We introduce **Optimal Transport Conditional Flow Matching** (OT-CFM), a CFM variant that approximates the dynamical formulation of optimal transport (OT). Based on OT theory, OT-CFM leverages the static optimal transport plan as well as the optimal probability paths and vector fields to approximate dynamic OT.
- [Simulation-free Schrödinger bridges via score and flow matching](https://arxiv.org/abs/2307.03672). We propose **Simulation-Free Score and Flow Matching** (\[SF\]<sup>2</sup>M). \[SF\]<sup>2</sup>M leverages OT-CFM as well as score-based methods to approximate Schrödinger bridges, a stochastic version of optimal transport.

If you find this code useful in your research, please cite the following papers (expand for BibTeX):

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
</summary>

```bibtex
@article{tong2023improving,
  title={Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author={Tong, Alexander and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and {Rector-Brooks}, Jarrid and Fatras, Kilian and Wolf, Guy and Bengio, Yoshua},
  year={2023},
  journal={arXiv preprint 2302.00482}
}
```

</details>

<details>
<summary>
A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023.
</summary>

```bibtex
@article{tong2023simulation,
   title={Simulation-Free Schr{\"o}dinger Bridges via Score and Flow Matching},
   author={Tong, Alexander and Malkin, Nikolay and Fatras, Kilian and Atanackovic, Lazar and Zhang, Yanlei and Huguet, Guillaume and Wolf, Guy and Bengio, Yoshua},
   year={2023},
   journal={arXiv preprint 2307.03672}
}
```

</details>

## Examples

![My Image](assets/8gaussians-to-moons.gif)

![My Image](assets/gaussian-to-moons.gif)

The density, vector field, and trajectories of simulation-free CNF training schemes: mapping 8 Gaussians to two moons (above) and a single Gaussian to two moons (below).

The first two methods, variance-preserving SDE (VP-SDE) and flow matching (FM), require a Gaussian source distribution so do not appear in the above example mapping 8 Gaussians distribution to the two moons distribution. Action matching with the same architecture (3x64 MLP with SeLU activations) underfits with the ReLU, SiLU, and SiLU activations as suggested in the [example code](https://github.com/necludov/jam), but it seems to fit better under our training setup (Action-Matching (Swish)).

The models to produce the GIFs are stored in `examples/models` and can be visualized with this notebook: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/model-comparison-plotting.ipynb).

We also have included an example of unconditional MNIST generation in `examples/notebooks/mnist_example.ipynb` for both deterministic and stochastic generation. [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/mnist_example.ipynb).

## The torchcfm Package

In our version 1 update we have extracted implementations of the relevant flow matching variants into a package `torchcfm`. This allows abstraction of the choice of the conditional distribution `q(z)`. `torchcfm` supplies the following loss functions:

- `ConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)
- `ExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi(x_0, x_1)$ where $\\pi$ is an exact optimal transport joint. This is used in \[Tong et al. 2023a\] and \[Poolidan et al. 2023\] as "OT-CFM" and "Multisample FM with Batch OT" respectively.
- `TargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ as defined in Lipman et al. 2023, learns a flow from a standard normal Gaussian to data using conditional flows which optimally transport the Gaussian to the datapoint (Note that this does not result in the marginal flow being optimal transport.
- `SchrodingerBridgeConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi\_\\epsilon(x_0, x_1)$ where $\\pi\_\\epsilon$ is a an entropically regularized OT plan, although in practice this is often approximated by a minibatch OT plan (See Tong et al. 2023b). The flow-matching variant of this where the marginals are equivalent to the Schrodinger Bridge marginals is known as `SB-CFM` \[Tong et al. 2023a\]. When the score is also known and the bridge is stochastic is called \[SF\]2M \[Tong et al. 2023b\]
- `VariancePreservingConditionalFlowMatcher`: $z = (x_0, x_1) $q(z) = q(x_0) q(x_1)$ but with conditional Gaussian probability paths which preserve variance over time using a trigonometric interpolation as presented in \[Albergo et al. 2023a\].

## V0 -> V1

In abstracting out the relevant losses from our `pytorch-lightning` implementation we have moved all of the `pytorch-lightning` code to `runner`. Every command that worked before for lightning should now be run from within the `runner` directory. V0 is now frozen in the `V0` branch.

Major Changes:

- Added code for the new Simulation-free Score and Flow Matching (SF)2M preprint
- Created `torchcfm` pip installable package
- Moved `pytorch-lightning` implementation and experiments to `runner` directory
- Moved `notebooks` -> `examples`
- Added image generation implementation in both lightning and a notebook in `examples`

## Related Work

Relevant papers on simulation-free training of flow models:

- Flow Matching for Generative Modeling (Lipman et al. 2023) [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al. 2023) [Paper](https://openreview.net/forum?id=XVjTT1nw5z) [Code](https://github.com/gnobitab/RectifiedFlow.git)
- Building Normalizing Flows with Stochastic Interpolants (Albergo et al. 2023a) [Paper](https://openreview.net/forum?id=li7qeBbCR1t)
- Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (Albergo et al. 2023b) [Paper](https://arxiv.org/abs/2303.08797)
- Action Matching: Learning Stochastic Dynamics From Samples (Neklyudov et al. 2022) [Paper](https://arxiv.org/abs/2210.06662) [Code](https://github.com/necludov/jam)
- Riemannian Flow Matching on General Geometries (Chen et al. 2023) [Paper](https://arxiv.org/abs/2302.03660)
- Multisample Flow Matching: Straightening Flows with Minibatch Couplings (Pooladian et al. 2023) [Paper](https://arxiv.org/abs/2304.14772)

## Code Contributions

This repo is extracted from a larger private codebase which loses the original commit history which contains work from other authors on the paper.

## How to run

Run a simple minimal example here [![Run in Google Colab](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/training-8gaussians-to-moons.ipynb). Or install the more efficient code locally with these steps.

Install dependencies

```bash
# clone project
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Note that `torchdyn==1.0.4` is broken on pypi. It may be necessary to install `torchdyn==1.0.3` until this is updated.

Train model with default configuration

```bash
cd runner

# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

You can also train a large set of models in parallel with SLURM as shown in `scripts/two-dim-cfm.sh` which trains the models used in the first 3 lines of Table 2.

## Project Structure

The directory structure of new project looks like this:

```

│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── examples              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
│── runner                    <- Shell scripts
|   ├── scripts                   <- Shell scripts
│   ├── configs                   <- Hydra configuration files
│   │   ├── callbacks                <- Callbacks configs
│   │   ├── debug                    <- Debugging configs
│   │   ├── datamodule               <- Datamodule configs
│   │   ├── experiment               <- Experiment configs
│   │   ├── extras                   <- Extra utilities configs
│   │   ├── hparams_search           <- Hyperparameter search configs
│   │   ├── hydra                    <- Hydra configs
│   │   ├── launcher                 <- Hydra launcher configs
│   │   ├── local                    <- Local configs
│   │   ├── logger                   <- Logger configs
│   │   ├── model                    <- Model configs
│   │   ├── paths                    <- Project paths configs
│   │   ├── trainer                  <- Trainer configs
│   │   │
│   │   ├── eval.yaml             <- Main config for evaluation
│   │   └── train.yaml            <- Main config for training
│   ├── src                       <- Source code
│   │   ├── datamodules              <- Lightning datamodules
│   │   ├── models                   <- Lightning models
│   │   ├── utils                    <- Utility scripts
│   │   │
│   │   ├── eval.py                  <- Run evaluation
│   │   └── train.py                 <- Run training
│   │
│   ├── tests                     <- Tests of any kind
│
|── torchcfm                    <- Shell scripts
|   ├── conditional_flow_matching.py      <- CFM classes
│   ├── models                            <- Model architectures
│   │   ├── models                           <- Models for 2D examples
│   │   ├── Unet                             <- Unet models for image examples
|
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>

<!-- deepspeed support still in beta
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like W&B or Tensorboard</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).

> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.

> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# log second gradient norm of the model
python train.py +trainer.track_grad_norm=2

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) doesn't require you to add any boilerplate to your code, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml).

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> **Note**: This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

</details>

<details>
<summary><b>Run tests</b></summary>

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist","experiment_X"]
```

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

<br>

## ❤️  Contributions

Have a question? Found a bug? Missing a specific feature? Feel free to file a new issue, discussion or PR with respective title and description.

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!

## License

Conditional-Flow-Matching is licensed under the MIT License.

```
MIT License

Copyright (c) 2023 Alexander Tong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
