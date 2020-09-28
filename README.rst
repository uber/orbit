.. image:: docs/img/orbit-icon-small.png

-------------------------------------------

**Disclaimer: This project may contain experimental code and may not be
ready for general use. Support and/or new releases may be limited.**

**Disclaimer: Orbit requires PyStan as a system dependency. PyStan is
licensed under** `GPLv3 <https://www.gnu.org/licenses/gpl-3.0.html>`__ **,
which is a free, copyleft license for software.**

Orbit is a Python package for time series modeling and inference
using Bayesian sampling methods for model estimation. It provides a
familiar and intuitive initialize-fit-predict interface for working with
time series tasks, while utilizing probabilistic modeling under
the hood.

The initial release supports concrete implementation for the following
models:

-  Local Global Trend (LGT)
-  Damped Local Trend (DLT)

Both models, which are variants of exponential smoothing, support
seasonality and exogenous (time-independent) features.

The initial release also supports the following sampling methods for
model estimation:

-  Markov-Chain Monte Carlo (MCMC) as a full sampling method
-  Maximum a Posteriori (MAP) as a point estimate method
-  Variational Inference (VI) as a hybrid-sampling method on approximate
   distribution


Quick Start
===========

Load data
---------

.. code:: python

    import pandas as pd
    import numpy as np

    DATA_FILE = "./examples/data/iclaims_example.csv"
    df = pd.read_csv(DATA_FILE, parse_dates=['week'])
    df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']] =\
        df[['claims', 'trend.unemploy', 'trend.filling', 'trend.job']].apply(np.log)

    test_size=52
    train_df=df[:-test_size]
    test_df=df[-test_size:]

Local-Global-Trend (LGT) Model with FULL Bayesian Prediction
------------------------------------------------------------

.. code:: python

    from orbit.models.lgt import LGTFull
    from orbit.diagnostics.plot import plot_predicted_data

    lgt = LGTFull(
        response_col='claims',
        date_col='week',
        regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
        seasonality=52,
        prediction_percentiles=[5, 95],
    )
    lgt.fit(df=train_df)

    # predicted df
    predicted_df = lgt.predict(df=test_df)

    # plot predictions
    plot_predicted_data(
        training_actual_df=train_df, predicted_df=predicted_df,
        date_col=lgt.date_col, actual_col=lgt.response_col,
        pred_col='prediction', test_actual_df=test_df
    )

.. image:: docs/img/lgt-mcmc-pred.png


Installation
============

Prerequisites
-------------

Install from PyPi:

.. code:: bash

    pip install orbit

Install from GitHub:

.. code:: bash

    git clone https://github.com/uber/orbit.git
    cd orbit
    pip install -r requirements.txt
    pip install .
