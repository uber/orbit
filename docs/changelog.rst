.. :changelog:

Changelog
=========
1.0.12 (2021-02-19)
-------------------
- Documentation update (#354, #362)
- Providing prediction intervals for point posteriors such as AggregatedPosterior and MAP (#357, #359)
- Abstract classes created to refactor posteriors estimation as templates (#360)
- Automating documentation and tutorials; migrating docs to readthedocs (#291)

1.0.11 (2021-02-18)
-------------------
- Core changes:
  - a simple ETS class is created (#280,  #296)
  - DLT is replacing LGT as the model used in the quick start and general demos (#305)
  - DLT and LGT are refactored to inherit from ETS  (#280)
  - DLT now supports regression with strictly positive/negative signs (#296)
  - deprecation on regression with LGT  (#305)
  - dependency update; remove enum34 and update other dependencies versions (#301)
  - fixed pickle error  (#342)

- Documentation:
  - updated tutorials (#309, #329, #332)
  - docstring cleanup with inherited classes (#350)

- Utilities:
  - include the provide hyper-parameters tuning (#288)
  - include dataloader with a few standard datasets  (#352, #337, #277, #248)
  - plotting functions now returns the plot object (#327, #325, #287, #279)

1.0.10 (2020-11-15)
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

1.0.1 (2020-09-10)
-------------------
- Minor plot enhancements (#209) 

1.0.0 (2020-09-09)
-------------------
- 1.0.0 is a redesign of the Orbit package class design and the first version intended for public release
