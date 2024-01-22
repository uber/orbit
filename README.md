<p align="center">
  &emsp;
  <a href="https://join.slack.com/t/orbit-support/shared_invite/zt-1207qlxjl-fhiX_8vTu1Fsa1ao1vGFEA">Join&nbsp;Slack</a>
  &emsp; | &emsp;
  <a href="https://orbit-ml.readthedocs.io/en/stable/">Documentation</a>
  &emsp; | &emsp;
  <a href="https://eng.uber.com/orbit/">Blog - Intro</a>
  &emsp; | &emsp;
  <a href="https://eng.uber.com/the-new-version-of-orbit-v1-1-is-released/">Blog - v1.1</a>
</p>

![Orbit banner](https://raw.githubusercontent.com/uber/orbit/dev/docs/img/orbit-banner.png)

-------------------------------------------------------------------------------------------------------------------------------------
<!--- BADGES: START --->
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/uber/orbit)
[![PyPI](https://img.shields.io/pypi/v/orbit-ml)][#pypi-package]
[![Build Status](https://github.com/uber/orbit/workflows/build/badge.svg?branch=dev)](https://github.com/uber/orbit/actions)
[![Documentation Status](https://readthedocs.org/projects/orbit-ml/badge/?version=latest)](https://orbit-ml.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/orbit-ml)][#pypi-package]
[![Downloads](https://pepy.tech/badge/orbit-ml)](https://pepy.tech/project/orbit-ml)
[![Conda Recipe](https://img.shields.io/static/v1?logo=conda-forge&style=flat&color=green&label=recipe&message=orbit-ml)][#conda-forge-feedstock]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/orbit-ml?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/orbit-ml?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![PyPI - License](https://img.shields.io/pypi/l/orbit-ml?logo=pypi&style=flat&color=green)][#github-license]

[#github-license]: https://github.com/uber/orbit/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/orbit-ml/
[#conda-forge-package]: https://anaconda.org/conda-forge/orbit-ml
[#conda-forge-feedstock]: https://github.com/conda-forge/orbit-ml-feedstock
<!--- BADGES: END --->


# User Notice

The default page of the repo is on `dev` branch. To install the dev version, please check the section `Installing from Dev Branch`. If you are looking for a **stable version**, please refer to the `master` branch [here](https://github.com/uber/orbit/tree/master).


# Disclaimer

This project

- is stable and being incubated for long-term support. It may contain new experimental code, for which APIs are subject to change.
- requires [cmdstanpy](https://mc-stan.org/cmdstanpy/) as one of the core dependencies for Bayesian sampling.

# Orbit: A Python Package for Bayesian Forecasting

Orbit is a Python package for Bayesian time series forecasting and inference. It provides a
familiar and intuitive initialize-fit-predict interface for time series tasks, while utilizing probabilistic programming languages under the hood.

For details, check out our documentation and tutorials:
- HTML (stable): https://orbit-ml.readthedocs.io/en/stable/
- HTML (latest): https://orbit-ml.readthedocs.io/en/latest/

Currently, it supports concrete implementations for the following models:

-  Exponential Smoothing (ETS)
-  Local Global Trend (LGT)
-  Damped Local Trend (DLT)
-  Kernel Time-based Regression (KTR)

It also supports the following sampling/optimization methods for model estimation/inferences:

-  Markov-Chain Monte Carlo (MCMC) as a full sampling method
-  Maximum a Posteriori (MAP) as a point estimate method
-  Variational Inference (VI) as a hybrid-sampling method on approximate
   distribution


##  Installation

### Installing Stable Release

Install the library either from PyPi or from the source with `pip`. 
Alternatively, you can also install it from Anaconda with `conda`:

**With pip**

1. Installing from PyPI

   ```sh
   $ pip install orbit-ml
   ```

2. Install from source

   ```sh
   $ git clone https://github.com/uber/orbit.git
   $ cd orbit
   $ pip install -r requirements.txt
   $ pip install .
   ```

**With conda**

The library can be installed from the conda-forge channel using conda.

```sh
$ conda install -c conda-forge orbit-ml
```

### Installing from Dev Branch

```sh
$ pip install git+https://github.com/uber/orbit.git@dev
```

## Quick Start with Damped-Local-Trend (DLT) Model

### FULL Bayesian Prediction

```python
from orbit.utils.dataset import load_iclaims
from orbit.models import DLT
from orbit.diagnostics.plot import plot_predicted_data

# log-transformed data
df = load_iclaims()
# train-test split
test_size = 52
train_df = df[:-test_size]
test_df = df[-test_size:]

dlt = DLT(
  response_col='claims', date_col='week',
  regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
  seasonality=52,
)
dlt.fit(df=train_df)

# outcomes data frame
predicted_df = dlt.predict(df=test_df)

plot_predicted_data(
  training_actual_df=train_df, predicted_df=predicted_df,
  date_col=dlt.date_col, actual_col=dlt.response_col,
  test_actual_df=test_df
)
```

![full-pred](docs/img/dlt-mcmc-pred.png)

## Demo

Nowcasting with Regression in DLT:

[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwinnglabs/ts-playground/blob/master/Orbit_Tutorial.ipynb)

Backtest on M3 Data:

[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwinnglabs/ts-playground/blob/master/orbit_m3_backtest.ipynb)

More examples can be found under [tutorials](./docs/tutorials)
and [examples](./examples).

# Contributing

We welcome community contributors to the project. Before you start, please read our
[code of conduct](CODE_OF_CONDUCT.md) and check out
[contributing guidelines](CONTRIBUTING.md) first.


# Versioning

We document versions and changes in our [changelog](./docs/changelog.rst).


# References

## Presentations

Check out the ongoing [deck](https://docs.google.com/presentation/d/1WfTtXAW3rud4TX9HtB3NkE6buDE8tWk6BKZ2hRNXjCI/edit?usp=sharing) for scope and roadmap of the project. An older deck used in the [meet-up](https://www.meetup.com/UberEvents/events/279446143/) during July 2021 can also be found [here](https://docs.google.com/presentation/d/1R0Ol8xahIE6XlrAjAi0ewu4nRxo-wQn8w6U7z-uiOzI/edit?usp=sharing).


## Citation

To cite Orbit in publications, refer to the following whitepaper:

[Orbit: Probabilistic Forecast with Exponential Smoothing](https://arxiv.org/abs/2004.08492)

Bibtex:

```
@misc{
    ng2020orbit,
    title={Orbit: Probabilistic Forecast with Exponential Smoothing},
    author={Edwin Ng,
        Zhishi Wang,
        Huigang Chen,
        Steve Yang,
        Slawek Smyl},
    year={2020}, eprint={2004.08492}, archivePrefix={arXiv}, primaryClass={stat.CO}
}
```

##  Papers

- Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip,
  P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. The Journal of Machine Learning
  Research, 20(1):973–978, 2019.
- Hoffman, M.D. and Gelman, A. The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo.
  J. Mach. Learn. Res., 15(1), pp.1593-1623, 2014.
- Hyndman, R., Koehler, A. B., Ord, J. K., and Snyder, R. D. Forecasting with exponential smoothing:
  the state space approach. Springer Science & Business Media, 2008.
- Smyl, S. Zhang, Q. Fitting and Extending Exponential Smoothing Models with Stan.
  International Symposium on Forecasting, 2015.

## Related projects

- [Pyro](https://github.com/pyro-ppl/pyro)
- [Stan](https://github.com/stan-dev/stan)
- [Rlgt](https://cran.r-project.org/web/packages/Rlgt/index.html)
- [forecast](https://github.com/robjhyndman/forecast)
- [prophet](https://facebook.github.io/prophet/)
