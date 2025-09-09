<div align="center">

# TorchCFM: a Conditional Flow Matching library

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
[![Downloads](https://static.pepy.tech/badge/torchcfm)](https://pepy.tech/project/torchcfm)
[![Downloads](https://static.pepy.tech/badge/torchcfm/month)](https://pepy.tech/project/torchcfm)

</div>

## Description

Conditional Flow Matching (CFM) is a fast way to train continuous normalizing flow (CNF) models. CFM is a simulation-free training objective for continuous normalizing flows that allows conditional generative modeling and speeds up training and inference. CFM's performance closes the gap between CNFs and diffusion models. To spread its use within the machine learning community, we have built a library focused on Flow Matching methods: TorchCFM. TorchCFM is a library showing how Flow Matching methods can be trained and used to deal with image generation, single-cell dynamics, tabular data and soon SO(3) data.

<p align="center">
<img src="assets/169_generated_samples_otcfm.png" width="600"/>
<img src="assets/8gaussians-to-moons.gif" />
</p>

The density, vector field, and trajectories of simulation-free CNF training schemes: mapping 8 Gaussians to two moons (above) and a single Gaussian to two moons (below). Action matching with the same architecture (3x64 MLP with SeLU activations) underfits with the ReLU, SiLU, and SiLU activations as suggested in the [example code](https://github.com/necludov/jam), but it seems to fit better under our training setup (Action-Matching (Swish)).

The models to produce the GIFs are stored in `examples/models` and can be visualized with this notebook: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/model-comparison-plotting.ipynb).

We also have included an example of unconditional MNIST generation in `examples/notebooks/mnist_example.ipynb` for both deterministic and stochastic generation. [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/mnist_example.ipynb).

## The torchcfm Package

In our version 1 update we have extracted implementations of the relevant flow matching variants into a package `torchcfm`. This allows abstraction of the choice of the conditional distribution `q(z)`. `torchcfm` supplies the following loss functions:

- `ConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$
- `ExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi(x_0, x_1)$ where $\\pi$ is an exact optimal transport joint. This is used in \[Tong et al. 2023a\] and \[Poolidan et al. 2023\] as "OT-CFM" and "Multisample FM with Batch OT" respectively.
- `TargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ as defined in Lipman et al. 2023, learns a flow from a standard normal Gaussian to data using conditional flows which optimally transport the Gaussian to the datapoint (Note that this does not result in the marginal flow being optimal transport).
- `SchrodingerBridgeConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi\_\\epsilon(x_0, x_1)$ where $\\pi\_\\epsilon$ is an entropically regularized OT plan, although in practice this is often approximated by a minibatch OT plan (See Tong et al. 2023b). The flow-matching variant of this where the marginals are equivalent to the Schrodinger Bridge marginals is known as `SB-CFM` \[Tong et al. 2023a\]. When the score is also known and the bridge is stochastic is called \[SF\]2M \[Tong et al. 2023b\]
- `VariancePreservingConditionalFlowMatcher`: $z = (x_0, x_1)$ $q(z) = q(x_0) q(x_1)$ but with conditional Gaussian probability paths which preserve variance over time using a trigonometric interpolation as presented in \[Albergo et al. 2023a\].

## How to cite

This repository contains the code to reproduce the main experiments and illustrations of two preprints:

- [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482). We introduce **Optimal Transport Conditional Flow Matching** (OT-CFM), a CFM variant that approximates the dynamical formulation of optimal transport (OT). Based on OT theory, OT-CFM leverages the static optimal transport plan as well as the optimal probability paths and vector fields to approximate dynamic OT.
- [Simulation-free Schrödinger bridges via score and flow matching](https://arxiv.org/abs/2307.03672). We propose **Simulation-Free Score and Flow Matching** (\[SF\]<sup>2</sup>M). \[SF\]<sup>2</sup>M leverages OT-CFM as well as score-based methods to approximate Schrödinger bridges, a stochastic version of optimal transport.

If you find this code useful in your research, please cite the following papers (expand for BibTeX):

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
</summary>

```bibtex
@article{tong2024improving,
title={Improving and generalizing flow-based generative models with minibatch optimal transport},
author={Alexander Tong and Kilian FATRAS and Nikolay Malkin and Guillaume Huguet and Yanlei Zhang and Jarrid Rector-Brooks and Guy Wolf and Yoshua Bengio},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=CD9Snc73AW},
note={Expert Certification}
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

## V0 -> V1

Major Changes:

- **Added cifar10 examples with an FID of 3.5**
- Added code for the new Simulation-free Score and Flow Matching (SF)2M preprint
- Created `torchcfm` pip installable package
- Moved `pytorch-lightning` implementation and experiments to `runner` directory
- Moved `notebooks` -> `examples`
- Added image generation implementation in both lightning and a notebook in `examples`

## Implemented papers

List of implemented papers:

- Flow Matching for Generative Modeling (Lipman et al. 2023) [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al. 2023) [Paper](https://openreview.net/forum?id=XVjTT1nw5z) [Code](https://github.com/gnobitab/RectifiedFlow.git)
- Building Normalizing Flows with Stochastic Interpolants (Albergo et al. 2023a) [Paper](https://openreview.net/forum?id=li7qeBbCR1t)
- Action Matching: Learning Stochastic Dynamics From Samples (Neklyudov et al. 2022) [Paper](https://arxiv.org/abs/2210.06662) [Code](https://github.com/necludov/jam)
- Concurrent work to our OT-CFM method: Multisample Flow Matching: Straightening Flows with Minibatch Couplings (Pooladian et al. 2023) [Paper](https://arxiv.org/abs/2304.14772)
- Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees (Jolicoeur-Martineau et al.) [Paper](https://arxiv.org/abs/2309.09968) [Code](https://github.com/SamsungSAILMontreal/ForestDiffusion)
- Soon: SE(3)-Stochastic Flow Matching for Protein Backbone Generation (Bose et al.) [Paper](https://arxiv.org/abs/2310.02391)

## How to run

Run a simple minimal example here [![Run in Google Colab](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/notebooks/training-8gaussians-to-moons.ipynb). Or install the more efficient code locally with these steps.

TorchCFM is now on [pypi](https://pypi.org/project/torchcfm/)! You can install it with:

```bash
pip install torchcfm
```

To use the full library with the different examples, you can install dependencies:

```bash
# clone project
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching

# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# install torchcfm
pip install -e .
```

To run our jupyter notebooks, use the following commands after installing our package.

```bash
# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebook
python -m ipykernel install --user --name=torchcfm

# launch our notebooks with the torchcfm kernel
```

## Project Structure

The directory structure looks like this:

```

│
├── examples              <- Jupyter notebooks
|   ├── cifar10           <- Cifar10 experiments
│   ├── notebooks         <- Diverse examples with notebooks
│
│── runner                    <- Everything related to the original version (V0) of the library
│
|── torchcfm                  <- Code base of our Flow Matching methods
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

## ❤️  Code Contributions

This toolbox has been created and is maintained by

- [Alexander Tong](http://alextong.net)
- [Kilian Fatras](http://kilianfatras.github.io)

It was initiated from a larger private codebase which loses the original commit history which contains work from other authors of the papers.

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
