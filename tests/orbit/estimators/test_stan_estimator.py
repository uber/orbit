from orbit.estimators.stan_estimator import StanEstimatorMCMC, StanEstimatorMAP


def test_stan_estimator_mcmc(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    mcmc_estimator = StanEstimatorMCMC(num_warmup=50)

    # extract posterior samples
    posteriors, training_metrics = mcmc_estimator.fit(
        model_name=stan_model_name,
        model_param_names=model_param_names,
        data_input=data_input,
        sampling_temperature=1.0,
    )

    assert set(model_param_names + ["loglk"]) == set(posteriors.keys())


def test_stan_estimator_map(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    map_estimator = StanEstimatorMAP()

    # extract posterior samples
    posteriors, training_metrics = map_estimator.fit(
        model_name=stan_model_name,
        model_param_names=model_param_names,
        data_input=data_input,
    )

    assert set(model_param_names + ["loglk"]) == set(posteriors.keys())
