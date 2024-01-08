data {
  // The sampling tempature t_star; this is currently not used.
  real<lower=0> T_STAR;
  // response related
  int<lower=0> NUM_OF_OBS;
  int<lower=0> N_VALID_RES;
  real<lower=0> RESPONSE_SD;
  real MEAN_Y;
  int<lower=0> DOF;
  vector[NUM_OF_OBS] RESPONSE;
  array[N_VALID_RES] int WHICH_VALID_RES;
  // trend related
  int<lower=0> N_KNOTS_LEV;
  matrix[NUM_OF_OBS, N_KNOTS_LEV] K_LEV;
  real<lower=0> LEV_KNOT_SCALE;
  // number of predictors
  int<lower=0> P;
  vector<lower=0>[P] COEF_INIT_KNOT_SCALE;
  vector<lower=0>[P] COEF_KNOT_SCALE;
  matrix[NUM_OF_OBS, P] REGRESSORS;
  
  // kernel
  int<lower=0> N_KNOTS_COEF;
  matrix[NUM_OF_OBS, N_KNOTS_COEF] K_COEF;
}
transformed data {
  vector[NUM_OF_OBS] RESPONSE_TRAN;
  // convert numpy index to stan
  array[N_VALID_RES] int WHICH_VALID_RES2;
  real t_star_inv;
  for (n in 1 : N_VALID_RES) {
    WHICH_VALID_RES2[n] = WHICH_VALID_RES[n] + 1;
  }
  RESPONSE_TRAN = RESPONSE - MEAN_Y;
  t_star_inv = 1.0 / T_STAR;
}
parameters {
  vector[N_KNOTS_LEV] lev_knot_drift;
  matrix[N_KNOTS_COEF, P] coef_knot_drift;
  real<lower=0, upper=RESPONSE_SD> obs_scale;
}
transformed parameters {
  // parameters after transformation by mean subtraction
  vector[NUM_OF_OBS] lev_tran;
  vector[N_KNOTS_LEV] lev_knot_tran;
  matrix[N_KNOTS_COEF, P] coef_knot_tran;
  matrix[NUM_OF_OBS, P] coef;
  vector[NUM_OF_OBS] regression;
  vector[NUM_OF_OBS] yhat;
  
  // Tempature based sampling
  // log probability of each observation
  matrix[NUM_OF_OBS, 1] loglk;
  loglk[ : , 1] = rep_vector(0, NUM_OF_OBS);
  
  // levels
  if (N_KNOTS_LEV > 1) {
    lev_knot_tran = cumulative_sum(lev_knot_drift);
  } else {
    lev_knot_tran = lev_knot_drift;
  }
  lev_tran = K_LEV * lev_knot_tran;
  
  // regression
  if (N_KNOTS_COEF > 1) {
    for (p in 1 : P) {
      coef_knot_tran[ : , p] = cumulative_sum(coef_knot_drift[ : , p]);
    }
  } else {
    coef_knot_tran = coef_knot_drift;
  }
  coef = rep_matrix(0, NUM_OF_OBS, P);
  if (P > 0) 
    coef = K_COEF * coef_knot_tran;
  
  if (P > 0) {
    for (n in 1 : NUM_OF_OBS) {
      regression[n] = sum(REGRESSORS[n,  : ] .* coef[n,  : ]);
    }
  } else {
    regression = rep_vector(0, NUM_OF_OBS);
  }
  
  yhat = lev_tran + regression;
  
  for (t in WHICH_VALID_RES2) {
    // the log probs of each overservation for WBIC
    loglk[t, 1] = student_t_lpdf(RESPONSE[t] | DOF, yhat[t], obs_scale);
  }
}
model {
  lev_knot_drift ~ double_exponential(0, LEV_KNOT_SCALE);
  if (N_KNOTS_COEF > 1) {
    for (p in 1 : P) {
      coef_knot_drift[1, p] ~ double_exponential(0, COEF_INIT_KNOT_SCALE[p]);
      coef_knot_drift[2 : N_KNOTS_COEF, p] ~ double_exponential(0,
                                                                COEF_KNOT_SCALE[p]);
    }
  } else {
    for (p in 1 : P) {
      coef_knot_drift[1, p] ~ double_exponential(0, COEF_INIT_KNOT_SCALE[p]);
    }
  }
  
  obs_scale ~ cauchy(0, RESPONSE_SD) T[0, RESPONSE_SD];
  // old way
  RESPONSE_TRAN[WHICH_VALID_RES2] ~ student_t(DOF, yhat[WHICH_VALID_RES2],
                                              obs_scale);
  // new way, with unit test issue
  //for  (t in WHICH_VALID_RES2) {
  //    target +=  t_star_inv*log_prob[t];
  //}
}
generated quantities {
  matrix[P, N_KNOTS_COEF] coef_knot;
  vector[N_KNOTS_LEV] lev_knot;
  vector[NUM_OF_OBS] lev;
  lev_knot = lev_knot_tran + MEAN_Y;
  lev = lev_tran + MEAN_Y;
  coef_knot = coef_knot_tran';
}

