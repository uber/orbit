data {
  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector[NUM_OF_OBS] RESPONSE;
  real INIT_LEV;
  real<lower=0> INIT_LEV_SD;
  real<lower=0> RESPONSE_SD;
  
  real<lower=0,upper=1> DAMPED_FACTOR;
  // Seasonality Hyper-Params
  int SEASONALITY;// 4 for quarterly, 12 for monthly, 52 for weekly
  int<lower=0,upper=1> NORM_SEAS;
  
  // step size
  real<lower=0> TIME_DELTA;
  
  // Residuals Tuning Hyper-Params
  real<lower=0> CAUCHY_SD; // derived by MAX(RESPONSE)/constant
  real<lower=1> MIN_NU; real<lower=1> MAX_NU;
}
transformed data {
}
parameters {
  // global trend parameters
  real gb; // global slope
  
  //level smoothing parameter
  real<lower=0,upper=1> l_sm; 
  //seasonality smoothing parameter
  real<lower=0,upper=1-l_sm> s_sm;
  real<lower=0,upper=l_sm> b_sm;

  // initial seasonality
  vector<lower=-1,upper=1>[SEASONALITY - 1] init_s;
  
  real<lower=1e-3,upper=RESPONSE_SD> obs_sigma;
  real<lower=MIN_NU,upper=MAX_NU> nu;
}

transformed parameters {
  vector[NUM_OF_OBS] err;
  vector[NUM_OF_OBS] yhat;
  vector[NUM_OF_OBS] l;
  vector[NUM_OF_OBS] b; // local slope
  vector[NUM_OF_OBS + SEASONALITY] s;
  vector[NUM_OF_OBS] gt_sum; // sum of global trend

  real sum_init_sea;
  vector[SEASONALITY] s_tilde;
  sum_init_sea = 0;
  for(i in 1:(SEASONALITY - 1)){
      s[i] = init_s[i];
      sum_init_sea += init_s[i];
  }

  // making sure the first cycle components sum up to zero
  s[SEASONALITY] = -1 * sum_init_sea;
  s[SEASONALITY + 1] = init_s[1];
  
  l[1] = RESPONSE[1] - s[1];
  b[1] = 0;
  gt_sum[1] = 0;
  err[1] = 0;
  yhat[1] = RESPONSE[1];

    
  for(i in 1:SEASONALITY) {
    s_tilde[i] = s[i];
  }

  for (t in 2:NUM_OF_OBS) {
    int m;
    real s_norm_factor;
    m = t % SEASONALITY;
    if (m == 0) {
      m = SEASONALITY;
    }

    gt_sum[t] = (t-1) * gb * TIME_DELTA;

    // forecast and observe process
    yhat[t] = gt_sum[t] + (l[t-1] + s[t] + DAMPED_FACTOR * b[t-1]);
    err[t] = RESPONSE[t] - yhat[t];

    // update process (l)
    l[t] = l[t-1] + l_sm * err[t];
    // update process (b)
    b[t] = DAMPED_FACTOR * b[t-1] + b_sm * err[t];
    // update process (s)
    // this normalization following Roberts (1982) and McKenzie (1986);
    // also see Hyndman (2008) Chpater 8.1.1
    s_norm_factor = s_sm/SEASONALITY * err[t];
    // print(s_norm_factor)
    // print(s[t])
    if (NORM_SEAS > 0) {
      s_tilde[m] = s_tilde[m] + s_sm * err[t] - s_norm_factor;
      s[t + SEASONALITY] = s_tilde[m];
      // for (i in 2:SEASONALITY) {
        // s_tilde[i] = s_tilde[i -1] - s_norm_factor;
      // }
      for (i in 1:SEASONALITY) {
        if (i != m) {
          s_tilde[i] -= s_norm_factor;
        }
      }
    } else {
      s[t + SEASONALITY] = s[t] + s_sm * err[t];
    }
  }
}
model {
  // gl ~ normal(0, 0.1);
  gb ~ normal(0, 1);
  //  subtracting previous seasonality to get initial level
  // init_l ~ normal(INIT_LEV - s[SEASONALITY], INIT_LEV_SD);
  obs_sigma ~ cauchy(1e-5, CAUCHY_SD) T[1e-5, RESPONSE_SD];
  for (t in 2:NUM_OF_OBS) 
    // err[t] ~ normal(0, obs_sigma);
    RESPONSE[t] ~ student_t(nu, yhat[t], obs_sigma);
    // err[t] ~ student_t(nu, 0, obs_sigma);
  
  // prior for seasonality
  for (i in 1:(SEASONALITY - 1))
    init_s[i] ~ normal(0, 0.33); // 33% lift is with 1 sd prob.
}
generated quantities {
}
