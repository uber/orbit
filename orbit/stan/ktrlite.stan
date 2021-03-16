data {
  // response related
  int<lower=0> N_OBS;
  int<lower=0> N_VALID_RES;
  real<lower=0> SDY;
  real MEAN_Y;
  int<lower=0> DOF;
  vector[N_OBS] RESPONSE;
  int WHICH_VALID_RES[N_VALID_RES];
  // trend related
  int<lower=0> N_KNOTS_LEV;
  matrix[N_OBS, N_KNOTS_LEV] K_LEV;
  real<lower=0> LEV_KNOT_SCALE;
  // number of predictors
  int<lower=0> P;
  vector[P] COEF_KNOT_POOL_LOC;
  vector<lower=0>[P] COEF_KNOT_POOL_SCALE;
  vector<lower=0>[P] COEF_KNOT_SCALE;
  matrix[N_OBS, P] REGRESSORS;

  // kernel
  int<lower=0> N_KNOTS_COEF;
  matrix[N_OBS, N_KNOTS_COEF] K_COEF;
}
transformed data {
  vector[N_OBS] RESPONSE_TRAN;
  // convert numpy index to stan
  int WHICH_VALID_RES2[N_VALID_RES];
  for (n in 1:N_VALID_RES) {
    WHICH_VALID_RES2[n] = WHICH_VALID_RES[n] + 1;
  }
  RESPONSE_TRAN = RESPONSE - MEAN_Y;
}

parameters {
  vector[N_KNOTS_LEV] lev_knot_drift;
  // vector[P] coef_knot_loc;
  matrix[N_KNOTS_COEF, P] coef_knot_drift;
  real<lower=0, upper=SDY> obs_scale;
}
transformed parameters {
  vector[N_OBS] lev_tran;
  vector[N_OBS] regression;
  vector[N_OBS] yhat;
  vector[N_KNOTS_LEV] lev_knot_tran;
  matrix[N_KNOTS_COEF, P] coef_knot_tran;
  matrix[N_OBS, P] coef;

  lev_knot_tran = cumulative_sum(lev_knot_drift);
  for (p in 1:P) {
    coef_knot_tran[:, p] = cumulative_sum(coef_knot_drift[:, p]);
  }
  lev_tran = K_LEV * lev_knot_tran;

  coef = rep_matrix(0, N_OBS, P);
  if (P > 0) coef = K_COEF * coef_knot_tran;

  if (P > 0) {
    for (n in 1:N_OBS) {
      regression[n] = sum(REGRESSORS[n, :] .* coef[n, :]);
    }
  } else {
    regression = rep_vector(0, N_OBS);
  }
  yhat = lev_tran + regression;
}


model {
  lev_knot_drift ~ double_exponential(0, LEV_KNOT_SCALE);
  for (p in  1:P)  {
    coef_knot_drift[1, p] ~ double_exponential(0, COEF_KNOT_POOL_SCALE[p]);
    coef_knot_drift[2:N_KNOTS_COEF, p] ~ double_exponential(0, COEF_KNOT_SCALE[p]);
  }
  // coef_knot_loc ~ normal(COEF_KNOT_POOL_LOC, COEF_KNOT_POOL_SCALE);
  // for (n in 1:N_KNOTS_COEF) {
  //   coef_knot_tran[n,:] ~ normal(coef_knot_loc, COEF_KNOT_SCALE);
  // }
  obs_scale ~ cauchy(0, SDY)T[0, SDY];
  RESPONSE_TRAN[WHICH_VALID_RES2] ~ student_t(DOF, yhat[WHICH_VALID_RES2], obs_scale);
}

generated quantities {
  matrix[P, N_KNOTS_COEF] coef_knot;
  vector[N_KNOTS_LEV] lev_knot;
  vector[N_OBS] lev;
  lev_knot = lev_knot_tran + MEAN_Y;
  lev = lev_tran + MEAN_Y;
  coef_knot = coef_knot_tran';
  // if (P > 0) {
  //   coef_knot = append_col(rr_knot, pr_knot)';
  // } else {
  //   coef_knot = rep_matrix(0, N_RR, N_KNOTS_COEF);
  // }
}
