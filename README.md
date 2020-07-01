# Orbit

**Disclaimer: This project may contain experimental code and may not be ready for general use. Support and/or new releases may be limited.**
**Disclaimer: Orbit requires PyStan which is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html), which is a free, copyleft license for software.**

Orbit is a Python package for general time series modeling and inference using Bayesian sampling methods for model estimation. It provides a familiar and intuitive initialize-fit-predict interface for working with time series tasks, while utilizing advanced probabilistic modeling under the hood.

The initial release supports concrete implementation for the following models:

* Local Global Trend (LGT)
* Damped Local Trend (DLT)

Both models, which are variants of exponential smoothing, support seasonality and exogenous (time-independent) features.

The initial release also supports the following sampling methods for model estimation:

* Markov-Chain Monte Carlo (MCMC) as a full sampling method
* Maximum a Posteriori (MAP) as a point estimate method
* Variational Inference (VI) as a hybrid-sampling method on approximate distribution

# Call for Contributions

The object-oriented design of the package is meant for easy extension of models, while abstracting the estimation procedure.

See [contributing guidelines](to_be_added) for details.

# Quick Start

## Load data
```python
import pandas as pd
import numpy as np

DATA_FILE = "data/iclaims_example.csv"
df = pd.read_csv(DATA_FILE, parse_dates=['week'])

test_size=52
train_df=df[:-test_size]
test_df=df[-test_size:]
```

## Local-Global-Trend (LGT) Model with FULL Bayesian Prediction
```python
from orbit.lgt import LGT
from orbit.utils.plot import plot_predicted_data

lgt_mcmc = LGT(response_col='claims',
               date_col='week',
               regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
               seasonality=52,
               infer_method='mcmc',
               predict_method='full',
               num_warmup=4000,
               num_sample=1000,
               is_multiplicative=True)
lgt_mcmc.fit(df=train_df)

# predicted df
predicted_df = lgt_mcmc.predict(df=test_df)

# plot predictions
plot_predicted_data(training_actual_df=train_df, predicted_df=predicted_df,
                    date_col=lgt_mcmc.date_col, actual_col=lgt_mcmc.response_col,
                    pred_col=50, test_actual_df=test_df)
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

# References

## Documentation

[Orbit API Documentation](<to_be_added>)

## Citation

To cite Orbit in publications:

White paper:
[Orbit: Probabilistic Forecast with Exponential Smoothing](https://arxiv.org/abs/2004.08492)

Bibtex:
>@article{orbit2020,
>  title={Orbit: Probabilistic Forecast with Exponential Smoothing},\
>  author={Ng, Edwin and Wang, Zhishi and Chen, Huigang and Yang, Steve and Smyl, Slawek},\
>  journal={arXiv preprint arXiv:2004.08492},\
>  year={2020}\
>}


## Papers/Books

* Hyndman, R. J. and Athanasopoulos, G. Forecasting: principles and practice. OTexts, 2018.
* Scott, S. L. and Varian, H. R. Predicting the Present with Bayesian Structural Time Series.
International Journal of Mathematical Modeling and Optimization 5 4–23, 2014.
* Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., and Riddell, A. Stan: A probabilistic programming language. Journal of statistical software, 76(1), 2017.
* Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. The Journal of Machine Learning Research, 20(1):973–978, 2019.
* Box, G. E. and Jenkins, G. M. Some recent advances in forecasting and control. Journal of the Royal Statistical Society. Series C (Applied Statistics), 17(2):91–109, 1968.


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
