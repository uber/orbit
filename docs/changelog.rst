.. :changelog:

Changelog
=========

1.1.5.1 (2026-05-20) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.5.1>`__)
-------------------------------------------------------------------------------------------------
:Python Version Support:
  - Drop Python 3.9–3.11; require Python ≥ 3.12
  - Add support for Python 3.13 (#894)
  - Update Trove classifiers for Python 3.12 and 3.13

:Core Changes:
  - Bump minimum dependency versions (numpy≥2.1.0, pandas≥2.2.3, matplotlib≥3.9.2, etc.) for Python 3.13 wheel availability
  - Replace deprecated ``np.in1d`` with ``np.isin`` (removed in NumPy 2.x)
  - Fix deprecated pandas frequency aliases (``M`` → ``ME``) in tests
  - Fix chained assignment for pandas Copy-on-Write compatibility
  - Fix invalid escape sequence in ``rmsse`` docstring

:CI/CD:
  - Update GitHub Actions (``checkout`` to v4, ``setup-python`` to v5)
  - Update test matrix to [3.12, 3.13]
  - Update ``tox`` envlist to ``py312, py313``
  - Update ``black`` target-version to ``py312, py313``

1.1.5.0 (2026-03-02) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.5.0>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Drop ``pkg_resources`` dependency; use ``importlib.metadata`` / ``importlib_resources`` instead
  - Fix defensive initialization of ``xlim`` / ``ylim`` in ``orbit.diagnostics.plot``
  - Fix support for various array types in model fitting
  - Update path-lib resolution for Stan model files
  - Consolidate package version into a single source of truth in ``pyproject.toml``; ``__version__`` is now read via ``importlib.metadata``

:CI/CD:
  - Consolidate lint and unit-test GitHub Actions workflows (#875)
  - Bump Jinja2 from 3.1.3 to 3.1.4 in docs (#868)

:Documentation:
  - Fix typo in regression_penalty tutorial (#884)

1.1.4.9 (2024-03-31) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.9>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Revert requirements change that caused installation regressions
  - Additional conda-environment fix (#867)

1.1.4.8 (2024-03-28) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.8>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Drop support for Python ≤ 3.8
  - Add conda-environment exception in setup to skip Stan compilation when running under conda (#866)
  - Update requirements

1.1.4.7 (2024-03-22) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.7>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Fix CmdStanPy version pinning and force-compile logic (#865)

1.1.4.6 (2024-03-19) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.6>`__)
-------------------------------------------------------------------------------------------------
.. note:: v1.1.4.5 was an internal pre-release superseded by this version.

:CI/CD:
  - Update PyPI deploy workflow to build ``sdist`` (#862)
  - Add retry for ``pip install .`` step in deploy workflow (#863)
  - Remove redundant ``pip install`` step from deploy GitHub Action

:Documentation:
  - Update README

1.1.4.4 (2024-03-13) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.4>`__)
-------------------------------------------------------------------------------------------------
.. note:: v1.1.4.4alpha was a pre-release superseded by this version.

:Core Changes:
  - Fix installation failure caused by pre-compile approach on Stan files

:Documentation:
  - Add ``Dockerfile`` for a quick-start environment ready for model fitting and forecasting
  - Refresh documentation and notebooks for v1.1.4.4

:CI/CD:
  - Black lint fixes (#859)
  - Fix ``isort`` linting

1.1.4.3 (2024-02-15) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.3>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Fix Linux wheel deployment for platform-specific builds (#840, #841, #842, #843, #844)
  - Update ``CMDSTAN_VERSION`` to 2.32.1 to resolve ``stan-dev/cmdstan#1158`` (#821)
  - CmdStanPy integration improvements (#824)

:Documentation:
  - Documentation and dependency update (#833)
  - Bump Jinja2 from 3.0.3 to 3.1.3 in docs (#828)

1.1.4.2 (2023-01-29) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.2>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Maintenance version bump

1.1.4.1 (2023-01-28) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4.1>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - Hot fix on package requirements
  - Lint workflow update
  - Reduce number of Python versions in unit-test matrix

1.1.4 (2024-01-21) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.4>`__)
-------------------------------------------------------------------------------------------------
:Core Changes:
  - replace stan sampling engine `PyStan2` by `cmdstanpy` (https://github.com/uber/orbit/pull/801)
  - update installation procedures and dependencies (https://github.com/uber/orbit/pull/833), (https://github.com/uber/orbit/pull/835), (https://github.com/uber/orbit/pull/821)
  - update installation process such that it pre-compile all stan files during wheel building (https://github.com/uber/orbit/pull/833), (https://github.com/uber/orbit/pull/835)

:Documentation:
  - update read the doc process and underlying doc with respect to new changes (https://github.com/uber/orbit/pull/836), (https://github.com/uber/orbit/pull/838)
  - prune old examples and duplicates under the `example/` folder (https://github.com/uber/orbit/pull/838)

1.1.3 (2022-11-30) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.3>`__)
-------------------------------------------------------------------------------------------------
:Core changes:

  - add python 3.8 unit tests (https://github.com/uber/orbit/pull/752)
  - optimize interface to be compatible with arviz (https://github.com/uber/orbit/pull/755)
  - requirements update (https://github.com/uber/orbit/pull/763)
  - code clean up (https://github.com/uber/orbit/pull/765)
  - dlt global trend prior adjustment (https://github.com/uber/orbit/pull/786)

:Documentation:
  - tutorial refresh (https://github.com/uber/orbit/pull/795)

:Utilities:
  - uses tqdm in parameters tuning (https://github.com/uber/orbit/pull/762)
  - residuals plot (https://github.com/uber/orbit/pull/758)
  - simpler stan compile interface (https://github.com/uber/orbit/pull/769)

1.1.2 (2022-04-28) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.2>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Add Conda installation option (#679)
  - Suppress the lengthy Stan logging message (#696)
  - WBIC for pyro SVI sampling and BIC for MAP optimization (#719, #710)
  - Backtest module to include confidence intervals (#724)
  - Allow configuration for compiled Stan model path (#713)
  - Box plot for regression coefficient comparison (#737)
  - Bounded logistic growth for DLT model (#712)
  - Enhance regression output reporting (#739)

:Documentation:
  - Add blacking linting to Github action workflow (#708)
  - Tutorial enhancement

:Utilities:
  - Add a new method `make_future_df` to prepare data frame for forecasting (#695)

1.1.2alpha (2022-04-06) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.2alpha>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Add Conda installation option (#679)
  - Suppress the lengthy Stan logging message (#696)
  - WBIC for pyro SVI sampling and BIC for MAP optimization (#719, #710)
  - Backtest module to include confidence intervals (#724)
  - Allow configuration for compiled Stan model path (#713)
  - Box plot for regression coefficient comparison (#737)
  - Bounded logistic growth for DLT model (#712)
  - Enhance regression output reporting (#739)

:Documentation:
  - Add blacking linting to Github action workflow (#708)
  - Tutorial enhancement

:Utilities:
  - Add a new method `make_future_df` to prepare data frame for forecasting (#695)

1.1.1 (2022-03-03) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.1>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - fix the mplstyle file path bug (#714)

1.1.0 (2022-01-11) (`release notes <https://github.com/uber/orbit/releases/tag/v1.1.0>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Redesign the model class structure with three core components: model template, estimator, and forecaster
    (#506, #507, #508, #513)
  - Introduce the Kernel-based Time-varying Regression (KTR) model (#515)
  - Implement the negative coefficient for LGT and KTR (#600, #601, #609)
  - Allow to handle missing values in response for LGT and DLT (#645)
  - Implement WBIC value for model candidate selection (#654)

:Documentation:
  - A new series of tutorials for KTR (#558, #559)
  - Migrate the CI from TravisCI to Github Actions (#556)
  - Missing value handle tutorial (#645)
  - WBIC tutorial (#663)

:Utilities:
  - New Plotting Palette (#571, #589)
  - Redesign the diagnostic plotting (#581, #607)
  - Raise a warning when date index is not evenly distributed (#639)

1.0.17 (2021-08-30) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.17>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Use global mean instead of median in ktrx model before next major release

1.0.16 (2021-08-27) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.16>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Bug fix and code improvement before next major release (#540, #541, #546)

1.0.15 (2021-08-02) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.15>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Prediction functionality refactoring (#430)
  - KTRLite model enhancement and interface cleanup (#440)
  - More flexible scheduling config in Backtester (#447)
  - Allow extraction of training related metrics (e.g. ELBO loss) in Pyro SVI (#443)
  - Add a flag to keep the posterior samples or not in aggregated model (#465)
  - Bug fix and code improvement (#428, #438, #459, #470)

:Documentation:
  - Clean up and standardize example notebooks (#462)
  - Tutorial update and enhancement (#431, #474)

:Utilities:
  - Diagnostic plot with Arviz (#433)
  - Refine plotting palette (#434, #473)
  - Create an orbit-featured plotting style (#434)

1.0.13 (2021-04-02) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.13>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - Implement a new model KTRLite (#380)
  - Refactoring of BaseTemplate (#382, #384)
  - Add MAPTemplate, FullBayesianTemplate, and AggregatedPosteriorTemplate (#394)
  - Remove dependency of scikit-learn (#379, #381)

:Documentation:
  - Add changelogs, release process, and contribution guidance (#363, #369, #370, #372)
  - Setup documentation deployment via TravisCI (#291)
  - New tutorial of making your own model (#389)
  - Tutorial enhancement (#383, #388)

:Utilities:
  - New EDA plot utilities (#403, #407, #408)
  - More options for exisiting plot utilities (#396)

1.0.12 (2021-02-19) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.12>`__)
-------------------------------------------------------------------------------------------------
- Documentation update (#354, #362)
- Providing prediction intervals for point posteriors such as AggregatedPosterior and MAP (#357, #359)
- Abstract classes created to refactor posteriors estimation as templates (#360)
- Automating documentation and tutorials; migrating docs to readthedocs (#291)

1.0.11 (2021-02-18) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.11>`__)
-------------------------------------------------------------------------------------------------
:Core changes:
  - a simple ETS class is created (#280,  #296)
  - DLT is replacing LGT as the model used in the quick start and general demos (#305)
  - DLT and LGT are refactored to inherit from ETS  (#280)
  - DLT now supports regression with strictly positive/negative signs (#296)
  - deprecation on regression with LGT  (#305)
  - dependency update; remove enum34 and update other dependencies versions (#301)
  - fixed pickle error  (#342)

:Documentation:
  - updated tutorials (#309, #329, #332)
  - docstring cleanup with inherited classes (#350)

:Utilities:
  - include the provide hyper-parameters tuning (#288)
  - include dataloader with a few standard datasets  (#352, #337, #277, #248)
  - plotting functions now returns the plot object (#327, #325, #287, #279)

1.0.10 (2020-11-15) (Initial Release)
-------------------------------------
- dpl v2 for travis config (#295)

1.0.9 (2020-11-15)
------------------
- debug travis pypi deployment (#293)
- Debug travis package deployment (#294)

1.0.8 (2020-11-15)
-------------------
- debug travis pypi deployment (#293)

1.0.7 (2020-11-14)
-------------------
- #279
- reorder fourier series calculation to match the df (#286)
- plot utility enhancement (#287)
- Setup TravisCI deployment for PyPI (#292)

1.0.6 (2020-11-13)
-------------------
- #251
- #257
- #259
- #263
- #248
- #264
- #265
- #270
- #273
- #277
- #281
- #282
