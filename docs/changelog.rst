.. :changelog:

Changelog
=========
1.0.13 (2021-04-02) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.13>`_)
-------------------
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

1.0.12 (2021-02-19) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.12>`_)
-------------------
- Documentation update (#354, #362)
- Providing prediction intervals for point posteriors such as AggregatedPosterior and MAP (#357, #359)
- Abstract classes created to refactor posteriors estimation as templates (#360)
- Automating documentation and tutorials; migrating docs to readthedocs (#291)

1.0.11 (2021-02-18) (`release notes <https://github.com/uber/orbit/releases/tag/v1.0.11>`_)
-------------------
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
-------------------
- dpl v2 for travis config (#295)

1.0.9 (2020-11-15)
-------------------
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
