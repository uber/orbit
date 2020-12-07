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
  // regrression related
  int<lower=0> N_PR;
  int<lower=0> N_RR;
  matrix[N_OBS, N_PR] PR;
  matrix[N_OBS, N_RR] RR;
  int<lower=0> N_KNOTS_COEF;
  matrix[N_OBS, N_KNOTS_COEF] K_COEF;
  vector[N_PR] PR_KNOT_POOL_LOC;
  vector<lower=0>[N_PR] PR_KNOT_POOL_SCALE;
  vector<lower=0>[N_PR] PR_KNOT_SCALE;
  vector[N_RR] RR_KNOT_POOL_LOC;
  vector<lower=0>[N_RR] RR_KNOT_POOL_SCALE;
  vector<lower=0>[N_RR] RR_KNOT_SCALE;
}
transformed data {
  matrix[N_OBS, N_PR + N_RR] REGRESSORS;
  vector[N_OBS] RESPONSE_TRAN;
  // convert numpy index to stan
  int WHICH_VALID_RES2[N_VALID_RES];
  for (n in 1:N_VALID_RES) {
    WHICH_VALID_RES2[n] = WHICH_VALID_RES[n] + 1;
  }
  REGRESSORS = append_col(PR, RR);
  RESPONSE_TRAN = RESPONSE - MEAN_Y;
}

parameters {
  // vector[N_KNOTS_LEV] lev_knot;
  vector[N_KNOTS_LEV] lev_knot_drift;
  vector<lower=0>[N_PR] pr_knot_loc;
  vector[N_RR] rr_knot_loc;
  matrix<lower=0>[N_KNOTS_COEF, N_PR] pr_knot;
  matrix[N_KNOTS_COEF, N_RR] rr_knot;
  real<lower=0, upper=SDY> obs_scale;
}
transformed parameters {
  vector[N_OBS] lev;
  vector[N_OBS] regression;
  vector[N_OBS] yhat;
  matrix<lower=0>[N_OBS, N_PR] pr_coef;
  matrix[N_OBS, N_RR] rr_coef;
  matrix[N_OBS, N_PR + N_RR] coef;
  vector[N_KNOTS_LEV] lev_knot_tran;
  
  lev_knot_tran = cumulative_sum(lev_knot_drift);
  lev = K_LEV * lev_knot_tran;
  pr_coef = rep_matrix(0, N_OBS, N_PR);
  rr_coef = rep_matrix(0, N_OBS, N_RR);
  if (N_PR > 0) pr_coef = K_COEF * pr_knot;
  if (N_RR > 0) rr_coef = K_COEF * rr_knot;
  coef = append_col(pr_coef, rr_coef);
  if (N_PR + N_RR > 0) {
    for (n in 1:N_OBS) {
      regression[n] = sum(REGRESSORS[n, :] .* coef[n, :]);
    }
  } else {
    regression = rep_vector(0, N_OBS);
  }
  yhat = lev + regression;
}


model {
  // lev_knot ~ double_exponential(0, LEV_KNOT_SCALE);
  lev_knot_drift ~ double_exponential(0, LEV_KNOT_SCALE);
  pr_knot_loc ~ normal(PR_KNOT_POOL_LOC, PR_KNOT_POOL_SCALE);
  for (n in 1:N_KNOTS_COEF) {
    pr_knot[n,:] ~ normal(pr_knot_loc, PR_KNOT_SCALE);
  }
  rr_knot_loc ~ normal(RR_KNOT_POOL_LOC, RR_KNOT_POOL_SCALE);
  for (n in 1:N_KNOTS_COEF) {
    rr_knot[n,:] ~ normal(rr_knot_loc, RR_KNOT_SCALE);
  }
  obs_scale ~ cauchy(0, SDY)T[0, SDY];
  RESPONSE_TRAN[WHICH_VALID_RES2] ~ student_t(DOF, yhat[WHICH_VALID_RES2], obs_scale);
}

generated quantities {
  matrix[N_PR + N_RR, N_KNOTS_COEF] coef_knot;
  vector[N_KNOTS_LEV] lev_knot;
  lev_knot = lev_knot_tran + MEAN_Y;
  if (N_PR + N_RR > 0) {
    coef_knot = append_col(pr_knot, rr_knot)';
  } else {
    coef_knot = rep_matrix(0, N_PR + N_RR, N_KNOTS_COEF);
  }
}
