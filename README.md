# ORBIT 

## Disclaimer
This project may contain experimental code and may not be ready for general use. Support and/or new releases may be limited.

## Intro
**ORBIT** is a package for **O**bject-o**R**iented **B**ayes**I**an **T**imes-series
model. It provides fast and robust forecast on generic time-series data in a Bayesian style with
flexible object-oriented design.

Current version supports two variations of exponential smoothing model:
1. Local-Global-Trend (LGT)
2. Damped-Trend (DT)

For numerical methods, current version provides three options:
1. Maximum a Posteriori (MAP), point estimate method
2. Variational Inference (VI), hybrid-sampling method on approximate distribution
3. Markov-Chain Monte Carlo (MCMC), full sampling method


# Workflow
We structure our process into three main steps: init, fit and predict.
![](docs/img/ORBIT-Workflow.png)
On one hand, the abstraction of modeling (written in .stan file) and prediction allow us to
extend different forms of model.  On the other, a base Bayesian time-series modeling workflow
is inherited as a commonly-shared arm for models.


# Quick Start
## Load and transform data
```python
import pandas as pd
import numpy as np
DATA_FILE = "data/iclaims.example.csv"
df = pd.read_csv(DATA_FILE, parse_dates=['week'])
df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']] = \
  df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']].apply(np.log, axis=1)

test_size=52
train_df=df[:-test_size]
orbit
test_df=df[-test_size-1:]
```
## Local-Global-Trend (LGT) Model with FULL Bayesian Prediction
```python
from uTS.lgt import LGT
from uTS.utils.utils import plot_predicted_data

lgt_mcmc = LGT(response_col='claims', date_col='week', seasonality=52,
    num_warmup=4000, num_sample=500, sample_method='mcmc', predict_method='full',
    n_bootstrap_draws=500)
lgt_mcmc.fit(df=train_df)
predicted_df = lgt_mcmc.predict(df=test_df)
plot_predicted_data(training_actual_df=train_df, predicted_df=predicted_df,
                    date_col=lgt_mcmc.date_col, actual_col=lgt_mcmc.response_col, pred_col=50,
                    test_actual_df=test_df)
```

![](docs/img/lgt-mcmc-pred.png)


# Installation

## Prerequisites

Install dependencies:
```
$ pip install -r requirements.txt
```

Install from pip:
```
$ pip install uTS
```

# Related projects
* [statsmodels](https://www.statsmodels.org/stable/index.html)
Common statistical models in Python
* [forecast](https://cran.r-project.org/web/packages/forecast/index.html)
Rob Hyndman's forecast package in R
* [Rlgt](https://cran.r-project.org/web/packages/Rlgt/index.html)
Bayesian Exponential Smoothing Models in R
* [stan](https://mc-stan.org/)
Open-source software to provide numeric solutions of Bayesian model
* [Pyro](https://pyro.ai/)
Another open-source API for Bayesian models built based on PyTorch
