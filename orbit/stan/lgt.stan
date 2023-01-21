// A variation from Holt-Winters’ seasonal method
// R. Hyndman's Exponential Smoothing Book (2008) with equation 3.16a-3.16e
// Additive Trend, Additive Seasonal and Additive Error model
// as known as ETS(A,A,A)
// Additional Regression Components are added as `r[t]`
// normalized seasonal component using chapter 8.1 in initial condition
// lgt version provided an additional global power trend suggested by S. Smyl

// rr stands for regular regressor(s) where the coef follows normal distribution
// pr stands for positive regressor(s) where the coef follows truncated normal distribution

// --- Code Style for .stan ---
// Upper case for Input
// lower case for intermediate variables and parameters to draw
functions {
  
}
data {
  // indicator of which method stan using
  int<lower=0, upper=1> WITH_MCMC;
  // The sampling tempature t_star; this is currently not used.
  real<lower=0> T_STAR;
  
  // Data Input
  // Response Data
  // number of observations
  int<lower=1> NUM_OF_OBS;
  vector[NUM_OF_OBS] RESPONSE;
  array[NUM_OF_OBS] int IS_VALID_RES;
  
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
  vector<upper=0>[NUM_OF_NR] NR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_NR] NR_SIGMA_PRIOR;
  
  // Regression Hyper Params
  // 0 As Fixed Ridge Penalty, 1 As Lasso, 2 As Auto-Ridge
  int<lower=0, upper=2> REG_PENALTY_TYPE;
  real<lower=0> AUTO_RIDGE_SCALE;
  real<lower=0> LASSO_SCALE;
  
  // Smoothing Hyper-Params
  real<upper=1> LEV_SM_INPUT;
  real<upper=1> SLP_SM_INPUT;
  real<upper=1> SEA_SM_INPUT;
  
  // Residuals Tuning Hyper-Params
  // this re-parameterization is sugggested by stan org and improves sampling
  // efficiently (on uniform instead of heavy-tail)
  // - 0.1 is made for point optimization to dodge boundary case
  real<lower=0> CAUCHY_SD; // derived by MAX(RESPONSE)/constant
  real<lower=1> MIN_NU;
  real<lower=1> MAX_NU;
  
  // Seasonality Hyper-Params
  int SEASONALITY; // 4 for quarterly, 12 for monthly, 52 for weekly
  real<lower=0> SEASONALITY_SD;
}
transformed data {
  int<lower=0, upper=1> IS_SEASONAL;
  // SIGMA_EPS is a offset to dodge lower boundary case;
  real<lower=0> SIGMA_EPS;
  int USE_VARY_SIGMA;
  int<lower=0, upper=1> LEV_SM_SIZE;
  int<lower=0, upper=1> SLP_SM_SIZE;
  int<lower=0, upper=1> SEA_SM_SIZE;
  // for WBIC
  real t_star_inv;
  t_star_inv = 1.0 / T_STAR;
  
  USE_VARY_SIGMA = 0;
  SIGMA_EPS = 1e-5;
  IS_SEASONAL = 0;
  LEV_SM_SIZE = 0;
  SLP_SM_SIZE = 0;
  SEA_SM_SIZE = 0;
  if (SEASONALITY > 1) 
    IS_SEASONAL = 1;
  // Only auto-ridge is using pr_sigma and rr_sigma
  if (REG_PENALTY_TYPE == 2) 
    USE_VARY_SIGMA = 1;
  
  if (LEV_SM_INPUT < 0) 
    LEV_SM_SIZE = 1;
  if (SLP_SM_INPUT < 0) 
    SLP_SM_SIZE = 1;
  if (SEA_SM_INPUT < 0) 
    SEA_SM_SIZE = 1 * IS_SEASONAL;
}
parameters {
  // regression parameters
  array[NUM_OF_PR * (USE_VARY_SIGMA)] real<lower=0> pr_sigma;
  array[NUM_OF_RR * (USE_VARY_SIGMA)] real<lower=0> rr_sigma;
  array[NUM_OF_NR * (USE_VARY_SIGMA)] real<lower=0> nr_sigma;
  vector<lower=0>[NUM_OF_PR] pr_beta;
  vector<upper=0>[NUM_OF_NR] nr_beta;
  vector[NUM_OF_RR] rr_beta;
  
  // smoothing parameters
  //level smoothing parameter
  array[LEV_SM_SIZE] real<lower=0, upper=1> lev_sm_dummy;
  //slope smoothing parameter
  array[SLP_SM_SIZE] real<lower=0, upper=1> slp_sm_dummy;
  //seasonality smoothing parameter
  array[SEA_SM_SIZE] real<lower=0, upper=1> sea_sm_dummy;
  
  // residual tuning parameters
  // use 5*CAUCHY_SD to dodge upper boundary case
  array[1 - WITH_MCMC] real<lower=SIGMA_EPS, upper=5 * CAUCHY_SD> obs_sigma_dummy;
  // this re-parameterization is sugggested by stan org and improves sampling
  // efficiently (on uniform instead of heavy-tail)
  // - 0.2 is made to dodge boundary case (tanh(pi/2 - 0.2) roughly equals 5 to be
  // consistent with MAP estimation)
  array[WITH_MCMC] real<lower=0, upper=pi() / 2 - 0.2> obs_sigma_unif_dummy;
  real<lower=MIN_NU, upper=MAX_NU> nu;
  
  // lgt parameters
  real<lower=0, upper=1> lt_coef; // local trend coef
  real<lower=-0.5, upper=0.5> gt_coef; // global trend coef
  real<lower=0, upper=1> gt_pow; // global trend parameter
  
  //initial seasonality
  vector<lower=-1, upper=1>[IS_SEASONAL ? SEASONALITY - 1 : 0] init_sea;
}
transformed parameters {
  real<lower=SIGMA_EPS, upper=5 * CAUCHY_SD> obs_sigma;
  vector<lower=0>[NUM_OF_OBS] l; // levs
  vector[NUM_OF_OBS] b; // slope
  vector[NUM_OF_OBS] pr; //positive regression component
  vector[NUM_OF_OBS] rr; //regular regression component
  vector[NUM_OF_OBS] r; //regression component
  vector[NUM_OF_OBS] lgt_sum; // integrated trend - sum of local & global trend
  vector[NUM_OF_OBS] yhat; // response prediction
  //seasonality vector with 1-cycle upfront as the initial condition
  vector[(NUM_OF_OBS + SEASONALITY) * IS_SEASONAL] s;
  // smoothing parameters
  real<lower=0, upper=1> lev_sm;
  real<lower=0, upper=1> slp_sm;
  real<lower=0, upper=1> sea_sm;
  
  // log likelihood of observations ~ 1-step ahead forecast
  vector[NUM_OF_OBS] loglk_1step;
  loglk_1step = rep_vector(0, NUM_OF_OBS);
  
  if (LEV_SM_SIZE > 0) {
    lev_sm = lev_sm_dummy[1];
  } else {
    lev_sm = LEV_SM_INPUT;
  }
  if (SLP_SM_SIZE > 0) {
    slp_sm = slp_sm_dummy[1];
  } else {
    slp_sm = SLP_SM_INPUT;
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
  if (NUM_OF_PR > 0) 
    pr = PR_MAT * pr_beta;
  else 
    pr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_RR > 0) 
    rr = RR_MAT * rr_beta;
  else 
    rr = rep_vector(0, NUM_OF_OBS);
  r = pr + rr;
  
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
  b[1] = 0;
  if (IS_SEASONAL) {
    l[1] = RESPONSE[1] - s[1] - r[1];
  } else {
    l[1] = RESPONSE[1] - r[1];
  }
  lgt_sum[1] = l[1];
  yhat[1] = RESPONSE[1];
  for (t in 2 : NUM_OF_OBS) {
    real s_t; // a transformed variable of seasonal component at time t
    if (IS_SEASONAL) {
      s_t = s[t];
    } else {
      s_t = 0.0;
    }
    
    // forecast process
    lgt_sum[t] = l[t - 1] + gt_coef * abs(l[t - 1]) ^ gt_pow
                 + lt_coef * b[t - 1];
    yhat[t] = lgt_sum[t] + s_t + r[t];
    
    // update process
    // l[t] update equation with l[t-1] ONLY by excluding b[t-1];
    // It is intentionally different from the Holt-Winter form
    // The change is suggested from Slawek's original SLGT model
    if (IS_VALID_RES[t]) {
      l[t] = fmax(lev_sm * (RESPONSE[t] - s_t - r[t])
                  + (1 - lev_sm) * l[t - 1], 0);
    } else {
      l[t] = fmax(lev_sm * (yhat[t] - s_t - r[t]) + (1 - lev_sm) * l[t - 1],
                  0);
    }
    b[t] = slp_sm * (l[t] - l[t - 1]) + (1 - slp_sm) * b[t - 1];
    // with parameterization as mentioned in 7.3 "Forecasting: Principles and Practice"
    // we can safely use "l[t]" instead of "l[t-1] + b[t-1]" where 0 < sea_sm < 1
    // otherwise with original one, use 0 < sea_sm < 1 - lev_sm
    if (IS_SEASONAL) {
      if (IS_VALID_RES[t]) {
        s[t + SEASONALITY] = sea_sm * (RESPONSE[t] - l[t] - r[t])
                             + (1 - sea_sm) * s_t;
      } else {
        s[t + SEASONALITY] = sea_sm * (yhat[t] - l[t] - r[t])
                             + (1 - sea_sm) * s_t;
      }
    }
  }
  if (WITH_MCMC) {
    // eqv. to obs_sigma ~ cauchy(SIGMA_EPS, CAUCHY_SD) T[SIGMA_EPS, ];
    obs_sigma = SIGMA_EPS + CAUCHY_SD * tan(obs_sigma_unif_dummy[1]);
  } else {
    obs_sigma = obs_sigma_dummy[1];
  }
  
  // the log probs of each overservation for WBIC
  for (t in 2 : NUM_OF_OBS) {
    if (IS_VALID_RES[t]) {
      loglk_1step[t] = student_t_lpdf(RESPONSE[t] | nu, yhat[t], obs_sigma);
    }
  }
}
model {
  // prior for residuals
  if (WITH_MCMC == 0) {
    // for MAP, set finite boundary else use uniform with transformation
    obs_sigma_dummy[1] ~ cauchy(SIGMA_EPS, CAUCHY_SD) T[SIGMA_EPS, 5
                                                                   * CAUCHY_SD];
  }
  for (t in 2 : NUM_OF_OBS) {
    if (IS_VALID_RES[t]) {
      target += t_star_inv * loglk_1step[t];
    }
  }
  
  // prior for seasonality
  for (i in 1 : (SEASONALITY - 1)) 
    init_sea[i] ~ normal(0, SEASONALITY_SD); // 33% lift is with 1 sd prob.
  
  // regression prior
  // see these references for details
  // 1. https://jrnold.github.io/bayesian_notes/shrinkage-and-regularized-regression.html
  // 2. https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#33_wide_weakly_informative_prior
  if (NUM_OF_PR > 0) {
    if (REG_PENALTY_TYPE == 0) {
      pr_beta ~ normal(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      pr_beta ~ double_exponential(PR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      for (i in 1 : NUM_OF_PR) {
        //weak prior for sigma
        pr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0, ];
      }
      //weak prior for betas
      pr_beta ~ normal(PR_BETA_PRIOR, pr_sigma);
    }
  }
  if (NUM_OF_NR > 0) {
    if (REG_PENALTY_TYPE == 0) {
      nr_beta ~ normal(NR_BETA_PRIOR, NR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      nr_beta ~ double_exponential(NR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      for (i in 1 : NUM_OF_NR) {
        nr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0, ];
      }
      //weak prior for betas
      nr_beta ~ normal(NR_BETA_PRIOR, nr_sigma);
    }
  }
  if (NUM_OF_RR > 0) {
    if (REG_PENALTY_TYPE == 0) {
      // fixed penalty ridge
      rr_beta ~ normal(RR_BETA_PRIOR, RR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      rr_beta ~ double_exponential(RR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      for (i in 1 : NUM_OF_RR) {
        rr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0, ];
      }
      //weak prior for betas
      rr_beta ~ normal(RR_BETA_PRIOR, rr_sigma);
    }
  }
}
generated quantities {
  vector[NUM_OF_PR + NUM_OF_NR + NUM_OF_RR] beta;
  matrix[NUM_OF_OBS - 1, 1] loglk;
  
  int idx;
  idx = 1;
  // compute regression
  if (NUM_OF_PR + NUM_OF_NR + NUM_OF_RR > 0) {
    if (NUM_OF_PR > 0) {
      beta[idx : idx + NUM_OF_PR - 1] = pr_beta;
      idx += NUM_OF_PR;
    }
    if (NUM_OF_NR > 0) {
      beta[idx : idx + NUM_OF_NR - 1] = nr_beta;
      idx += NUM_OF_NR;
    }
    if (NUM_OF_RR > 0) {
      beta[idx : idx + NUM_OF_RR - 1] = rr_beta;
    }
    // truncate small numeric values
    for (iidx in 1 : NUM_OF_PR + NUM_OF_NR + NUM_OF_RR) {
      if (abs(beta[iidx]) <= 1e-5) 
        beta[iidx] = 0;
    }
  }
  loglk[ : , 1] = loglk_1step[2 : ];
}

