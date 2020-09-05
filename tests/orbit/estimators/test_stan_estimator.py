from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorVI, StanEstimatorMAP


def test_stan_estimator_mcmc(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    mcmc_estimator = StanEstimatorMCMC()

    # extract posterior samples
    stan_extract = mcmc_estimator.fit(model_name=stan_model_name, model_param_names=model_param_names,
                                      data_input=data_input)

    expected_extract_keys = model_param_names[:] + ['lp__']

    assert set(expected_extract_keys) == set(stan_extract.keys())


def test_stan_estimator_vi(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    vi_estimator = StanEstimatorVI()

    # extract posterior samples
    stan_extract = vi_estimator.fit(model_name=stan_model_name, model_param_names=model_param_names,
                                    data_input=data_input)

    expected_extract_keys = model_param_names[:] + ['lp__']

    assert set(expected_extract_keys) == set(stan_extract.keys())


def test_stan_estimator_map(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    map_estimator = StanEstimatorMAP()

    # extract posterior samples
    stan_extract = map_estimator.fit(model_name=stan_model_name, model_param_names=model_param_names,
                                     data_input=data_input)

    expected_extract_keys = model_param_names[:]  # note no `lp__` for optimizing() method

    assert set(expected_extract_keys) == set(stan_extract.keys())
