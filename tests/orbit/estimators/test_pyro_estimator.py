from orbit.estimators.pyro_estimator import PyroEstimatorSVI


def test_pyro_estimator_vi(stan_estimator_lgt_model_input):
    stan_model_name, model_param_names, data_input = stan_estimator_lgt_model_input

    # create estimator
    vi_estimator = PyroEstimatorSVI(num_steps=50)

    # extract posterior samples
    posteriors, training_metrics = vi_estimator.fit(
        model_name=stan_model_name,
        model_param_names=model_param_names,
        data_input=data_input,
        sampling_temperature=1.0,
    )

    assert set(model_param_names) == set(posteriors.keys())
