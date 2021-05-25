from orbit.estimators.pyro_estimator import PyroEstimatorVI, PyroEstimatorMAP


def test_pyro_estimator_vi(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    vi_estimator = PyroEstimatorVI(num_steps=50)

    # extract posterior samples
    posteriors, training_metrics = vi_estimator.fit(
        model_name=stan_model_name,
        model_param_names=model_param_names,
        data_input=data_input,
    )

    assert set(model_param_names) == set(posteriors.keys())


def test_pyro_estimator_map(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    map_estimator = PyroEstimatorMAP(num_steps=50)

    # extract posterior samples
    posteriors, training_metrics = map_estimator.fit(
        model_name=stan_model_name,
        model_param_names=model_param_names,
        data_input=data_input,
    )

    assert set(model_param_names) == set(posteriors.keys())
