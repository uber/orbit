.. :changelog:

Changelog
=========
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
