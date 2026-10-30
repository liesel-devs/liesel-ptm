# Penalized Transformation Models in Liesel

[![pre-commit](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml)
[![pytest-cov](tests/coverage.svg)](https://github.com/liesel-devs/liesel-ptm/actions/workflows/pytest.yml)


Penalized Transformation Models are a class of flexible structured additive
distributional regression models. This is a Python library for estimating these models
with Markov Chain Monte Carlo (MCMC) methods.

Learn more in the paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2024). Bayesian penalized transformation models: Structured additive location-scale regression for arbitrary conditional distributions (No. arXiv:2404.07440). arXiv. `https://doi.org/10.48550/arXiv.2404.07440 <https://doi.org/10.48550/arXiv.2404.07440>`_

Application examples are included in the supplementary materials for the paper, which
can be found here: https://github.com/liesel-devs/ptm-supplement

## Installation

The library can be installed from GitHub:

```
$ pip install git+https://github.com/liesel-devs/liesel-ptm.git
```


## Getting Started


Construct predictors and terms with [liesel-gam](https://github.com/liesel-devs/liesel-gam).
The reference examples cover the supported workflows:

- [Modern GPTM](demo/reference_modern.py): location, scale, and shape predictors; optimization and VI.
- [Original PTM](demo/reference_original.py): shared shape and computed intercepts.

```sh
uv run python -m demo.reference_modern
uv run python -m demo.reference_original
```
