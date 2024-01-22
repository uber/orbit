About Orbit
============

Orbit is a Python package for Bayesian time series modeling and inference. It provides a
familiar and intuitive initialize-fit-predict interface for working with
time series tasks, while utilizing probabilistic programing languages under
the hood.

Currently, it supports the following models:

-  Damped Local Trend (DLT)
-  Exponential Smoothing (ETS)
-  Local Global Trend (LGT)
-  Kernel-based Time-varying Regression (KTR)

It also supports the following sampling methods for model estimation:

-  Markov-Chain Monte Carlo (MCMC) as a full sampling method
-  Maximum a Posteriori (MAP) as a point estimate method
-  Stochastic Variational Inference (SVI) as a hybrid-sampling method on approximate
   distribution
   
Under the hood, the package is leveraging probabilistic program such as `pyro <https://pyro.ai/>`__ and `cmdstanpy
<https://mc-stan.org/cmdstanpy/>`__.


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


Blog Post
---------
1. Introducing Orbit, An Open Source Package for Time Series Inference and Forecasting [
`Link <https://eng.uber.com/orbit/>`__]
2. The New Version of Orbit (v1.1) is Released: The Improvements, Design Changes, and Exciting Collaborations [
`Link <https://eng.uber.com/the-new-version-of-orbit-v1-1-is-released/>`__]
