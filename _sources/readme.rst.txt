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


Tutorials
=============

To learn more, please dive in the _tutorials.

.. _tutorials: /tutorials