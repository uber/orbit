data {
  // indicator of which method stan using
  int<lower=0,upper=1> WITH_MCMC;

  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector[NUM_OF_OBS] RESPONSE;
  
  // Regression Data
  int<lower=0> NUM_OF_RR; // number of regular regressors
  matrix[NUM_OF_OBS, NUM_OF_RR] RR_MAT; // regular coef regressors, more volatile range
  vector[NUM_OF_RR] RR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_RR] RR_SIGMA_PRIOR;
  int<lower=0> NUM_OF_PR; // number of positive regressors
  matrix[NUM_OF_OBS, NUM_OF_PR] PR_MAT; // positive coef regressors, less volatile range
  vector<lower=0>[NUM_OF_PR] PR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_PR] PR_SIGMA_PRIOR;
  int<lower=0> NUM_OF_NR; // number of negative regressors
  matrix[NUM_OF_OBS, NUM_OF_NR] NR_MAT; // negative coef regressors, less volatile range
  vector<lower=0>[NUM_OF_NR] NR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_NR] NR_SIGMA_PRIOR;

  // Regression Hyper Params
  // 0 As Fixed Ridge Penalty, 1 As Lasso
  int <lower=0,upper=2> REG_PENALTY_TYPE;

  // Smoothing Hyper-Params
  real<upper=1> LEV_SM_INPUT;
  real<upper=1> SEA_SM_INPUT;

  // Seasonality Hyper-Params
  int SEASONALITY;// 4 for quarterly, 12 for monthly, 52 for weekly
  
  real<lower=0> RESPONSE_SD;
}
transformed data {
  int IS_SEASONAL;
  int<lower=0,upper=1> LEV_SM_SIZE;
  int<lower=0,upper=1> SEA_SM_SIZE;
  
  LEV_SM_SIZE = 0;
  SEA_SM_SIZE = 0;

  IS_SEASONAL = 0;

  if (SEASONALITY > 1) IS_SEASONAL = 1;

  if (LEV_SM_INPUT < 0) LEV_SM_SIZE = 1;
  if (SEA_SM_INPUT < 0) SEA_SM_SIZE = 1 * IS_SEASONAL;

}
parameters {
  // regression parameters
  real<lower=0> rr_sigma[NUM_OF_RR];
  real<lower=0> pr_sigma[NUM_OF_PR];
  real<lower=0> nr_sigma[NUM_OF_NR];
  vector[NUM_OF_RR] rr_beta;
  vector<lower=0>[NUM_OF_PR] pr_beta;
  vector<upper=0>[NUM_OF_NR] nr_beta;
  
  // smoothing parameters
  //level smoothing parameter
  real<lower=0,upper=1> lev_sm_dummy[LEV_SM_SIZE];
  //seasonality smoothing parameter
  real<lower=0,upper=1> sea_sm_dummy[SEA_SM_SIZE];

  // initial seasonality
  vector<lower=-1,upper=1>[IS_SEASONAL ? SEASONALITY - 1:0] init_sea;
  
 real<lower=0, upper=RESPONSE_SD> obs_sigma;
}
transformed parameters {
  vector[NUM_OF_OBS] l; // local level
  vector[NUM_OF_OBS] rr; //regular regression component
  vector[NUM_OF_OBS] pr; //positive regression component
  vector[NUM_OF_OBS] nr; //negative regression component
  vector[NUM_OF_OBS] r; //regression component
  vector[NUM_OF_OBS] yhat; // response prediction
  // seasonality vector with 1-cycle upfront as the initial condition
  vector[(NUM_OF_OBS + SEASONALITY) * IS_SEASONAL] s;
  // smoothing parameters
  real<lower=0,upper=1> lev_sm;
  real<lower=0,upper=1> sea_sm;

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

  // compute regression components
  if (NUM_OF_RR>0)
    rr = RR_MAT * rr_beta;
  else
    rr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_PR > 0)
    pr = PR_MAT * pr_beta;
  else
    pr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_NR > 0)
    nr = NR_MAT * nr_beta;
  else
    nr = rep_vector(0, NUM_OF_OBS);
  r = rr + pr + nr;

  // states initial condition
  if (IS_SEASONAL) {
    real sum_init_sea;
    sum_init_sea = 0;
    for(i in 1:(SEASONALITY - 1)){
        sum_init_sea += init_sea[i];
        s[i] = init_sea[i];
    }
    // making sure the first cycle components sum up to zero
    s[SEASONALITY] = -1 * sum_init_sea;
    s[SEASONALITY + 1] = init_sea[1];
  }

  if (IS_SEASONAL) {
    l[1] = RESPONSE[1] - s[1] - r[1];
  } else {
    l[1] = RESPONSE[1] - r[1];
  }
  yhat[1] = RESPONSE[1];

  for (t in 2:NUM_OF_OBS) {
    real s_t; // a transformed variable of seasonal component at time t
    if (IS_SEASONAL) {
      s_t = s[t];
    } else {
      s_t = 0.0;
    }
    // forecast process
    yhat[t] = l[t-1] + s_t + r[t];

    // update process
    l[t] = lev_sm * (RESPONSE[t] - s_t - r[t]) + (1 - lev_sm) * l[t-1];
    // with parameterization as mentioned in 7.3 "Forecasting: Principles and Practice"
    // we can safely use "l[t]" instead of "l[t-1] + damped_factor_dummy * b[t-1]" where 0 < sea_sm < 1
    // otherwise with original one, use 0 < sea_sm < 1 - lev_sm
    if (IS_SEASONAL)
        s[t + SEASONALITY] = sea_sm * (RESPONSE[t] - l[t]  - r[t]) + (1 - sea_sm) * s_t;
  }

}
model {
  //prior for residuals
  obs_sigma ~ cauchy(0, RESPONSE_SD);
  for (t in 2:NUM_OF_OBS) {
    RESPONSE[t] ~ normal(yhat[t], obs_sigma);
  }

  // prior for seasonality
  for (i in 1:(SEASONALITY - 1))
    init_sea[i] ~ normal(0, 0.33); // 33% lift is with 1 sd prob.

  // regression prior
  // see these references for details
  // 1. https://jrnold.github.io/bayesian_notes/shrinkage-and-regularized-regression.html
  // 2. https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#33_wide_weakly_informative_prior
  if (NUM_OF_RR > 0) {
    if (REG_PENALTY_TYPE == 0) {
      // fixed penalty ridge
      rr_beta ~ normal(RR_BETA_PRIOR, RR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      rr_beta ~ double_exponential(RR_BETA_PRIOR, RR_SIGMA_PRIOR);
    }
  }
  if (NUM_OF_PR > 0) {
    if (REG_PENALTY_TYPE== 0) {
      // fixed penalty ridge
      pr_beta ~ normal(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      pr_beta ~ double_exponential(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    }
  }
  if (NUM_OF_NR > 0) {
    if (REG_PENALTY_TYPE== 0) {
      // fixed penalty ridge
      nr_beta ~ normal(NR_BETA_PRIOR, NR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      nr_beta ~ double_exponential(NR_BETA_PRIOR, NR_SIGMA_PRIOR);
    }
  }
}
generated quantities {
  vector[NUM_OF_RR + NUM_OF_PR + NUM_OF_NR] beta;
  int idx;
  idx = 1;
  // compute regression
  if (NUM_OF_RR + NUM_OF_PR + NUM_OF_NR > 0) {
    if (NUM_OF_RR > 0) {
      beta[idx:idx+NUM_OF_RR-1] = rr_beta;
      idx += NUM_OF_RR;
    }
    if (NUM_OF_PR > 0) {
      beta[idx:idx+NUM_OF_PR-1] = pr_beta;
      idx += NUM_OF_PR;
    }
    if (NUM_OF_NR > 0) {
      beta[idx:idx+NUM_OF_NR-1] = nr_beta;
    }
  }
}
