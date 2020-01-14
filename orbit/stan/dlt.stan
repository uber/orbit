// Holt-Winters’ seasonal method
// Additive Trend, Additive Seasonal and Additive Error model
// as known as ETS(A,A,A)
// Hyndman Exponential Smoothing Book page. 46
// Using equation 3.16a-3.16e
// Additional Regression Components are added as r[t]
// normalized seasonal component using chapter 8.1 in initial components
// can consider dynamic normalization later
// lgt version provided an additional global power trend suggested by Slawek

// damped trend method suppress the global trend
// and follows Section 7.2 from Hyndman's blog

// rr stands for regular regressor(s) where the coef follows normal distribution
// pr stands for positive regressor(s) where the coef follows truncated normal distribution

// --- Code Style for .stan ---
// Upper case for Input
// lower case for intermediate variables and variables we are interested

data {
  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector<lower=0>[NUM_OF_OBS] RESPONSE;
  // Regression Data
  int<lower=0> NUM_OF_PR; // number of positive regressors
  matrix[NUM_OF_OBS, NUM_OF_PR] PR_MAT; // positive coef regressors, less volatile range
  vector<lower=0>[NUM_OF_PR] PR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_PR] PR_SIGMA_PRIOR;
  int<lower=0> NUM_OF_RR; // number of regular regressors
  matrix[NUM_OF_OBS, NUM_OF_RR] RR_MAT; // regular coef regressors, more volatile range
  vector[NUM_OF_RR] RR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_RR] RR_SIGMA_PRIOR;

  // Trend Hyper-Params
  real<lower=0,upper=1>   LEV_SM_MIN;
  real<lower=0,upper=1>   LEV_SM_MAX;
  real<lower=0,upper=1>   SLP_SM_MIN;
  real<lower=0,upper=1>   SLP_SM_MAX;

  // Regression Hyper-Params
  real <lower=0> BETA_MAX;
  int<lower=0,upper=1> FIX_REG_COEF_SD;
  real<lower=0,upper=10> REG_SIGMA_SD;

  // Residuals Tuning Hyper-Params
  real<lower=0> CAUCHY_SD; //using max(RESPONSE)/300, not very sensitive
  //real<lower=0> MIN_SIGMA;
  //real<lower=0> MIN_VAL;
  real<lower=1> MIN_NU; real<lower=1> MAX_NU;

  // Damped Trend Hyper-Params
  real DAMPED_FACTOR_MIN;
  real DAMPED_FACTOR_MAX;
  real DAMPED_FACTOR_FIXED;

  // Seasonality Hyper-Params
  real<lower=-1,upper=1> SEA_MIN;
  real<lower=-1,upper=1> SEA_MAX;
  real<lower=0,upper=1> SEA_SM_MIN;
  real<lower=0,upper=1> SEA_SM_MAX;
  int SEASONALITY;// 4 for quarterly, 12 for monthly, 52 for weekly
}
transformed data {
  int IS_SEASONAL;
  int DAMPED_FACTOR_SIZE;

  DAMPED_FACTOR_SIZE = 1;
  IS_SEASONAL = 0;
  if (SEASONALITY > 1) IS_SEASONAL = 1;
  if (DAMPED_FACTOR_FIXED > 0) DAMPED_FACTOR_SIZE = 0;
}
parameters {
  // regression parameters
  real<lower=0> pr_sigma[NUM_OF_PR * (1 - FIX_REG_COEF_SD)];
  real<lower=0> rr_sigma[NUM_OF_RR * (1 - FIX_REG_COEF_SD)];
  vector<lower=0,upper=BETA_MAX>[NUM_OF_PR] pr_beta;
  vector<lower=-1 * BETA_MAX,upper=BETA_MAX>[NUM_OF_RR] rr_beta;

  real<lower=LEV_SM_MIN,upper=LEV_SM_MAX> lev_sm; //level smoothing parameter
  real<lower=SLP_SM_MIN,upper=SLP_SM_MAX> slp_sm; //slope smoothing parameter

  // residual tuning parameters
  // real<lower=0> obs_sigma;
  real<lower=0, upper=pi()/2> obs_sigma_unif_dummy;
  real<lower=MIN_NU,upper=MAX_NU> nu;

  // trend parameters
  real gl; // global level
  real<lower=-1,upper=1> gb; // global slope
  real<lower=DAMPED_FACTOR_MIN,upper=DAMPED_FACTOR_MAX> damped_factor[DAMPED_FACTOR_SIZE];

  // seasonal parameters
  //seasonality smoothing parameter
  real<lower=SEA_SM_MIN,upper=SEA_SM_MAX> sea_sm[IS_SEASONAL ? 1:0];
  //initial seasonality
  vector<lower=SEA_MIN,upper=SEA_MAX>[IS_SEASONAL ? SEASONALITY - 1:0] init_sea;

}
transformed parameters {
  real<lower=0> obs_sigma;
  vector[NUM_OF_OBS] l; // local level
  vector[NUM_OF_OBS] b; // local slope
  vector[NUM_OF_OBS] pr; //positive regression component
  vector[NUM_OF_OBS] rr; //regular regression component
  vector[NUM_OF_OBS] r; //regression component
  vector[NUM_OF_OBS] gt_sum; // sum of global trend
  vector[NUM_OF_OBS] lt_sum; // sum of local trend
  vector[NUM_OF_OBS] yhat; // response prediction
  //seasonality vector with 1-cycle upfront as the initial condition
  vector[(NUM_OF_OBS + SEASONALITY) * IS_SEASONAL] s;
  real damped_factor_dummy;

  // compute regression
  if (NUM_OF_PR > 0)
    pr = PR_MAT * pr_beta;
  else
    pr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_RR>0)
    rr = RR_MAT * rr_beta;
  else
    rr = rep_vector(0, NUM_OF_OBS);
  r = pr + rr;

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
  gt_sum[1] = gl;
  b[1] = 0;
  if (IS_SEASONAL) {
    l[1] = RESPONSE[1] - gl - s[1] - r[1];
  } else {
    l[1] = RESPONSE[1] - gl - r[1];
  }
  lt_sum[1] = l[1];
  yhat[1] = RESPONSE[1];

  // sequential sampling on state variables
  // a dummy for damped factor to split cases whether they are direct input or not;
  if (DAMPED_FACTOR_SIZE > 0) {
    damped_factor_dummy = damped_factor[1];
  } else {
    damped_factor_dummy = DAMPED_FACTOR_FIXED;
  }

  for (t in 2:NUM_OF_OBS) {
    real s_t; // a transformed variable of seasonal component at time t
    if (IS_SEASONAL) {
      s_t = s[t];
    } else {
        s_t = 0.0;
    }
    lt_sum[t] = l[t-1] + damped_factor_dummy * b[t-1];
    gt_sum[t] = gt_sum[t-1] + gb;
    yhat[t] = gt_sum[t] + lt_sum[t] + s_t + r[t];

    l[t] = lev_sm * (RESPONSE[t] - gt_sum[t] - s_t - r[t]) + (1 - lev_sm) * lt_sum[t];
    b[t] = slp_sm * (l[t] - l[t-1]) + (1 - slp_sm) * damped_factor_dummy * b[t-1];
    // with parameterization as mentioned in 7.3 "Forecasting: Principles and Practice"
    // we can safely use "l[t]" instead of "l[t-1] + b[t-1]" where 0 < sea_sm < 1
    // otherwise with original one, use 0 < sea_sm < 1 - lev_sm
    if (IS_SEASONAL)
        s[t + SEASONALITY] = sea_sm[1] * (RESPONSE[t] - gt_sum[t] - l[t] - r[t]) + (1 - sea_sm[1]) * s_t;
  }

  obs_sigma = CAUCHY_SD * tan(obs_sigma_unif_dummy); // obs_sigma ~ cauchy(0, CAUCHY_SD);
}
model {
  // prior for residuals
  // obs_sigma ~ cauchy(0, CAUCHY_SD) T[0,];
  if (NUM_OF_PR > 0) {
    if (FIX_REG_COEF_SD == 0) {
      //weak prior for sigma
      for(i in 1:NUM_OF_PR) {
        pr_sigma[i] ~ cauchy(PR_SIGMA_PRIOR[i], REG_SIGMA_SD) T[0,];
      }
      //weak prior for betas
      pr_beta ~ normal(PR_BETA_PRIOR, pr_sigma);
    } else { //straight point prior for sigma
      //weak prior for betas
      pr_beta ~ normal(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    }
  }
  if (NUM_OF_RR > 0) {
    if (FIX_REG_COEF_SD == 0) {
      //weak prior for sigma
      for(j in 1:NUM_OF_RR) {
        rr_sigma[j] ~ cauchy(RR_SIGMA_PRIOR[j], REG_SIGMA_SD) T[0,];
      }
      //weak prior for betas
      rr_beta ~ normal(RR_BETA_PRIOR, rr_sigma);
    } else { //straight point prior for sigma
      //weak prior for betas
      rr_beta ~ normal(RR_BETA_PRIOR, RR_SIGMA_PRIOR);
    }
  }
  for (i in 1:(SEASONALITY - 1))
    init_sea[i] ~ normal(0, 0.33); // 33% lift is with 1 sd prob.
  for (t in 2:NUM_OF_OBS) {
    RESPONSE[t] ~ student_t(nu, yhat[t], obs_sigma);
  }
}
