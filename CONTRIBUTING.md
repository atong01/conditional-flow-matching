# Contributing to TorchCFM

Thanks for your interest in contributing to TorchCFM! This document covers the
basics of setting up a development environment and the process for submitting
changes. By participating, you are expected to uphold the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

TorchCFM targets Python 3.10. The recommended workflow uses conda:

```bash
# clone your fork
git clone https://github.com/<your-username>/conditional-flow-matching.git
cd conditional-flow-matching

# create a conda environment with python 3.10
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch following https://pytorch.org/get-started/locally/

# install runtime dependencies and the package in editable mode
pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) lets you iterate on `torchcfm/`
without reinstalling. Most examples live in `examples/` as Jupyter notebooks; to
run them, register the conda env as a kernel:

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=torchcfm
```

## Code style and pre-commit

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, with a
line length of 99, and [pyupgrade](https://github.com/asottile/pyupgrade). Both
configurations are in [`pyproject.toml`](pyproject.toml) and wired up in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml). After cloning, install the
hooks once:

```bash
pip install pre-commit
pre-commit install
```

Pre-commit will then run automatically on every `git commit`. You can also run
it manually on all files:

```bash
pre-commit run --all-files
```

The ruff configuration selects the `C`, `E`, `F`, `I`, and `W` rule families,
ignores `E501` (line length is enforced by the formatter), and sorts imports
with `isort`. Run the formatter and linter directly with:

```bash
ruff format .
ruff check --fix .
```

## Testing

Tests live in the `tests/` directory and are run with
[pytest](https://docs.pytest.org/). After installing the package, run the full
suite with:

```bash
pytest tests/ -v
```

Some tests are marked `slow` and can be skipped during quick iteration:

```bash
pytest tests/ -v -m "not slow"
```

Please add or update tests alongside any change to the `torchcfm` package.
Doctests are enabled in [`pyproject.toml`](pyproject.toml)
(`--doctest-modules`), so docstrings in `torchcfm/` are executed as part of the
suite.

## Pull request process

1. **Fork** the repository and clone your fork locally (see the dev setup).
2. **Create a branch** from `main` for your change:
   ```bash
   git checkout -b fix/my-improvement
   ```
3. **Make your changes**, keeping commits focused. Add or update tests as
   needed.
4. **Run pre-commit and tests locally** before pushing:
   ```bash
   pre-commit run --files <changed-files>
   pytest tests/ -v
   ```
5. **Push** to your fork and **open a pull request** against `main`. In the PR
   description, describe what changed and why, and link any related issues.
6. A maintainer will review your PR. Please address review feedback by pushing
   additional commits (avoid force-pushing unless requested).

## Reporting bugs

Before opening an issue, please verify that:

- The problem reproduces on the current `main` branch.
- Your Python and PyTorch versions are recent.
- You are not seeing an already-reported issue (search existing issues first).

When filing a bug report, include:

- A minimal reproducible example (code + the exact command you ran).
- Your environment: OS, Python version, PyTorch version, and `torchcfm` version
  (see `torchcfm/version.py`).
- The full traceback, if an error is raised.

## Adding a new flow matching method

Flow matching losses are implemented as classes in
[`torchcfm/conditional_flow_matching.py`](torchcfm/conditional_flow_matching.py).
All matchers inherit from the base `ConditionalFlowMatcher` and override the
relevant methods (e.g. `compute_mu_t`, `compute_conditional_flow`,
`sample_xt`, `compute_score`). The existing concrete matchers are:

- `ExactOptimalTransportConditionalFlowMatcher` — minibatch OT coupling (OT-CFM).
- `TargetConditionalFlowMatcher` — Gaussian-to-data flow (Lipman et al. 2023).
- `SchrodingerBridgeConditionalFlowMatcher` — entropic OT / Schrödinger bridge.
- `VariancePreservingConditionalFlowMatcher` — variance-preserving (trigonometric)
  interpolation.

To add a new method:

1. Subclass `ConditionalFlowMatcher` (or one of its subclasses) in
   `conditional_flow_matching.py` and implement the conditional probability
   path and vector field.
2. Document the class and its methods with NumPy-style docstrings — these become
   doctests.
3. Add a test under `tests/` that checks your matcher produces tensors of the
   expected shape and that the loss is finite.
4. Update the README's matcher list and, if applicable, add an example notebook
   under `examples/`.

## Questions

For questions about usage or suggestions for improvements, please [open an issue](https://github.com/atong01/conditional-flow-matching/issues).
