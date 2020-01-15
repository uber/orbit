# Orbit 
Test
**Disclaimer: This project may contain experimental code and may not be ready for general use. Support and/or new releases may be limited.**

Orbit is a Python package for general time series modeling and inference using Bayesian sampling methods for model estimation. It provides a familiar and intuitive initialize-fit-predict interface for working with time series tasks, while utilizing advanced probabilistic modeling under the hood.

The initial release supports concrete implementation for the following models:

* Local Global Trend (LGT)
* Damped Local Trend (DLT)

Both models, which are variants of exponential smoothing, support seasonality and exogenous (time-independent) features.

The initial release also supports the following sampling methods for model estimation:

* Maximum a Posteriori (MAP) as a point estimate method
* Variational Inference (VI) as a hybrid-sampling method on approximate distribution
* Markov-Chain Monte Carlo (MCMC) as a full sampling method

# Call for Contributions

The object-oriented design of the package is meant for easy extension of models, while abstracting the estimation procedure.

See <separate contribution doc> for details.


# Quick Start
## Load and transform data
```python
import pandas as pd
import numpy as np
DATA_FILE = "data/iclaims.example.csv"
df = pd.read_csv(DATA_FILE, parse_dates=['week'])

# log transformation
df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']] = \
  df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']].apply(np.log, axis=1)

test_size=52
train_df=df[:-test_size]
test_df=df[-test_size:]
```

## Local-Global-Trend (LGT) Model with FULL Bayesian Prediction
```python
from orbit.lgt import LGT
from orbit.utils.utils import plot_predicted_data

lgt_mcmc = LGT(response_col='claims', date_col='week', seasonality=52)
lgt_mcmc.fit(df=train_df)

# predicted df
predicted_df = lgt_mcmc.predict(df=test_df)

# plot predictions
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
$ pip install orbit
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
