About Orbit
============

Orbit is a Python package for Bayesian time series modeling and inference. It provides a
familiar and intuitive initialize-fit-predict interface for working with
time series tasks, while utilizing probabilistic programing languages under
the hood.

Currently, it supports the following models:

-  Exponential Smoothing (ETS)
-  Local Global Trend (LGT)
-  Damped Local Trend (DLT)

It also supports the following sampling methods for
model estimation:

-  Markov-Chain Monte Carlo (MCMC) as a full sampling method
-  Maximum a Posteriori (MAP) as a point estimate method
-  Variational Inference (VI) as a hybrid-sampling method on approximate
   distribution

Quick Example
-------------

Orbit APIs follow a Scikit-learn stype API design, with a user-friendly interface. After instantiating a model
object, one can use .fit and .predict for model training and prediction. Below is a quick illustration using the DLT model.

.. code:: python

    from orbit.models.dlt import DLTFull

    dlt = DLTFull(
        response_col='claims',
        date_col='week',
        regressor_col=['trend.unemploy', 'trend.filling', 'trend.job'],
        seasonality=52,
    )

    dlt.fit(df=train_df)

    predicted_df = dlt.predict(df=test_df)


Citation
--------

To cite Orbit in publications, refer to the following whitepaper:

`Orbit: Probabilistic Forecast with Exponential Smoothing <https://arxiv.org/abs/2004.08492>`__


Bibtex:

.. code-block:: console

    @misc{
        ng2020orbit,
        title={Orbit: Probabilistic Forecast with Exponential Smoothing},
        author={Edwin Ng,
            Zhishi Wang,
            Huigang Chen,
            Steve Yang,
            Slawek Smyl
        },
        year={2020}, eprint={2004.08492}, archivePrefix={arXiv}, primaryClass={stat.CO}
    }



