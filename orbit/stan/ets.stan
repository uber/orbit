data {
  // indicator of which method stan using
  int<lower=0, upper=1> WITH_MCMC;
  // The sampling tempature t_star; this is currently not used.
  real<lower=0> T_STAR;
  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector[NUM_OF_OBS] RESPONSE;
  array[NUM_OF_OBS] int IS_VALID_RES;
  // Smoothing Hyper-Params
  real<upper=1> LEV_SM_INPUT;
  real<upper=1> SEA_SM_INPUT;
  // Seasonality Hyper-Params
  int SEASONALITY;
  real<lower=0> SEASONALITY_SD;
  real<lower=0> RESPONSE_SD;
}
transformed data {
  int IS_SEASONAL;
  int<lower=0, upper=1> LEV_SM_SIZE;
  int<lower=0, upper=1> SEA_SM_SIZE;
  real t_star_inv;
  t_star_inv = 1.0 / T_STAR;
  
  LEV_SM_SIZE = 0;
  SEA_SM_SIZE = 0;
  
  IS_SEASONAL = 0;
  
  if (SEASONALITY > 1) 
    IS_SEASONAL = 1;
  
  if (LEV_SM_INPUT < 0) 
    LEV_SM_SIZE = 1;
  if (SEA_SM_INPUT < 0) 
    SEA_SM_SIZE = 1 * IS_SEASONAL;
}
parameters {
  // smoothing parameters
  //level smoothing parameter
  array[LEV_SM_SIZE] real<lower=0, upper=1> lev_sm_dummy;
  //seasonality smoothing parameter
  array[SEA_SM_SIZE] real<lower=0, upper=1> sea_sm_dummy;
  
  // initial seasonality
  vector<lower=-1, upper=1>[IS_SEASONAL ? SEASONALITY - 1 : 0] init_sea;
  
  real<lower=0, upper=RESPONSE_SD> obs_sigma;
}
transformed parameters {
  vector[NUM_OF_OBS] l; // local level
  vector[NUM_OF_OBS] yhat; // response prediction
  // seasonality vector with 1-cycle upfront as the initial condition
  vector[(NUM_OF_OBS + SEASONALITY) * IS_SEASONAL] s;
  // smoothing parameters
  real<lower=0, upper=1> lev_sm;
  real<lower=0, upper=1> sea_sm;
  
  // log likelihood of observations ~ 1-step ahead forecast
  vector[NUM_OF_OBS] loglk_1step;
  loglk_1step = rep_vector(0, NUM_OF_OBS);
  
  if (LEV_SM_SIZE > 0) {
    lev_sm = lev_sm_dummy[1];
  } else {
    lev_sm = LEV_SM_INPUT;
  }
  if (IS_SEASONAL) {
    if (SEA_SM_SIZE > 0) {
      sea_sm = sea_sm_dummy[1];
    } else {
      sea_sm = SEA_SM_INPUT;
    }
  } else {
    sea_sm = 0.0;
  }
  
  // states initial condition
  if (IS_SEASONAL) {
    real sum_init_sea;
    sum_init_sea = 0;
    for (i in 1 : (SEASONALITY - 1)) {
      sum_init_sea += init_sea[i];
      s[i] = init_sea[i];
    }
    // making sure the first cycle components sum up to zero
    s[SEASONALITY] = -1 * sum_init_sea;
    s[SEASONALITY + 1] = init_sea[1];
  }
  
  if (IS_SEASONAL) {
    l[1] = RESPONSE[1] - s[1];
  } else {
    l[1] = RESPONSE[1];
  }
  yhat[1] = RESPONSE[1];
  
  for (t in 2 : NUM_OF_OBS) {
    real s_t; // a transformed variable of seasonal component at time t
    if (IS_SEASONAL) {
      s_t = s[t];
    } else {
      s_t = 0.0;
    }
    // forecast process
    yhat[t] = l[t - 1] + s_t;
    // the log probs of each overservation for WBIC
    if (IS_VALID_RES[t]) {
      loglk_1step[t] = normal_lpdf(RESPONSE[t] | yhat[t], obs_sigma);
    }
    
    // update process
    if (IS_VALID_RES[t]) {
      l[t] = lev_sm * (RESPONSE[t] - s_t) + (1 - lev_sm) * l[t - 1];
    } else {
      l[t] = lev_sm * (yhat[t] - s_t) + (1 - lev_sm) * l[t - 1];
    }
    // with parameterization as mentioned in 7.3 "Forecasting: Principles and Practice"
    // we can safely use "l[t]" instead of "l[t-1] + damped_factor_dummy * b[t-1]" where 0 < sea_sm < 1
    // otherwise with original one, use 0 < sea_sm < 1 - lev_sm
    if (IS_SEASONAL) {
      if (IS_VALID_RES[t]) {
        s[t + SEASONALITY] = sea_sm * (RESPONSE[t] - l[t])
                             + (1 - sea_sm) * s_t;
      } else {
        s[t + SEASONALITY] = sea_sm * (yhat[t] - l[t]) + (1 - sea_sm) * s_t;
      }
    }
  }
}
model {
  //prior for residuals
  obs_sigma ~ cauchy(0, RESPONSE_SD);
  // prior for seasonality
  for (i in 1 : (SEASONALITY - 1)) 
    init_sea[i] ~ normal(0, SEASONALITY_SD);
  // Likelihood
  for (t in 2 : NUM_OF_OBS) {
    if (IS_VALID_RES[t]) {
      target += t_star_inv * loglk_1step[t];
    }
  }
}
generated quantities {
  matrix[NUM_OF_OBS, 1] loglk;
  loglk[ : , 1] = loglk_1step;
}

