data {
  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  int NUM_OF_REG; // number of regressors
  
  vector[NUM_OF_OBS] RESPONSE;
  real<lower=0> RESPONSE_SD;

  // regressor values matrix
  matrix[NUM_OF_OBS, NUM_OF_REG] REG_MAT;
  
  // Residuals Tuning Hyper-Params
  // real<lower=0> CAUCHY_SD; // derived by MAX(RESPONSE)/constant
  real<lower=1> MIN_NU; real<lower=1> MAX_NU;
}
transformed data {
}
parameters {
  // global trend parameters
  real gb; // global slope
  
  vector[NUM_OF_REG] init_beta; //init reg coef
  
  //level smoothing parameter
  real<lower=0,upper=1> l_sm;
  vector<lower=0,upper=0.1>[NUM_OF_REG] b_sm;

  real<lower=1e-3,upper=RESPONSE_SD> obs_sigma;
  real<lower=MIN_NU,upper=MAX_NU> nu;
}

transformed parameters {
  vector[NUM_OF_OBS] err;
  vector[NUM_OF_OBS] yhat;
  vector[NUM_OF_OBS] l;
  vector[NUM_OF_OBS] reg;
  matrix[NUM_OF_OBS, NUM_OF_REG] r; // total regression
  matrix[NUM_OF_OBS, NUM_OF_REG] b; // reg coefs
  vector[NUM_OF_OBS] gt_sum;

  if (NUM_OF_REG >= 1) {
    for (i in 1:NUM_OF_REG) {
      b[1, i] = init_beta[i];
      r[1,i] = b[1,i] * REG_MAT[1,i];
    }
  }
  
  reg[1] = sum(r[1,:]);
  l[1] = RESPONSE[1] - reg[1];
  gt_sum[1] = 0;
  err[1] = 0;
  yhat[1] = RESPONSE[1];

  for (t in 2:NUM_OF_OBS) {
    gt_sum[t] = (t-1) * gb;
    // forecast and observe process
    // derive regression
    if (NUM_OF_REG >= 1) {
      for (i in 1:NUM_OF_REG) {
        // real delta;
        // delta = REG_MAT[t, i] - REG_MAT[t-1,i];
        // r[t,i] = b[t-1,i] * delta;
        r[t,i] = b[t-1,i] * REG_MAT[t, i];
      }
      // reg[t] = sum(r[t,:]) + reg[t-1];
      reg[t] = sum(r[t,:]);
    } else {
      reg[t] = 0;
    }
    // derive yhat and error
    yhat[t] = gt_sum[t] + l[t-1] + reg[t];
    err[t] = RESPONSE[t] - yhat[t];

    // update process (l)
    l[t] = l[t-1] + l_sm * err[t];
    
    // update coefficients and regression (r)
    if (NUM_OF_REG >= 1) {
      for (i in 1:NUM_OF_REG) {
        real delta;
        delta = REG_MAT[t, i] - REG_MAT[t-1,i];
        if (delta > 1e-5) {
          b[t,i] = b[t-1,i] + b_sm[i] * err[t]/delta;
        } else {
          b[t,i] = b[t-1,i];
        }
      }
    }
  }
}
model {
  gb ~ normal(0, 1);
  obs_sigma ~ cauchy(0.1 * RESPONSE_SD, 0.3 * RESPONSE_SD) T[1e-5, RESPONSE_SD];
  for (t in 2:NUM_OF_OBS) 
    RESPONSE[t] ~ student_t(nu, yhat[t], obs_sigma);
  if (NUM_OF_REG > 0) {
    init_beta ~ normal(0, 1);
  }  
}
generated quantities {
}
