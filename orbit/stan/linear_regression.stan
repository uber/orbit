// Bayesian Linear Regression with Regularization

data {
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector[NUM_OF_OBS] RESPONSE; // response column

  // Regression Data
  int<lower=0> NUM_OF_PR; // number of positive regressors
  matrix[NUM_OF_OBS, NUM_OF_PR] PR_MAT; // positive coef regressors, less volatile range
  vector<lower=0>[NUM_OF_PR] PR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_PR] PR_SIGMA_PRIOR;
  int<lower=0> NUM_OF_RR; // number of regular regressors
  matrix[NUM_OF_OBS, NUM_OF_RR] RR_MAT; // regular coef regressors, more volatile range
  vector[NUM_OF_RR] RR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_RR] RR_SIGMA_PRIOR;

  // Regression Hyper Params
  // 0 As Fixed Ridge Penalty, 1 As Lasso, 2 As Auto-Ridge
  int <lower=0,upper=2> REG_PENALTY_TYPE;
  real<lower=0> AUTO_RIDGE_SCALE;
  real<lower=0> LASSO_SCALE;
}
transformed data {
  int USE_VARY_SIGMA;
  int NU;

  if (REG_PENALTY_TYPE == 2) USE_VARY_SIGMA = 1;

  NU = NUM_OF_OBS - (NUM_OF_PR + NUM_OF_RR) - 1;
}
parameters {
  // regression parameters
  real<lower=0> pr_sigma[NUM_OF_PR * (USE_VARY_SIGMA)];
  real<lower=0> rr_sigma[NUM_OF_RR * (USE_VARY_SIGMA)];
  vector<lower=0>[NUM_OF_PR] pr_beta;
  vector[NUM_OF_RR] rr_beta;
  real<lower=0> obs_sigma; // error scale
}
transformed parameters {
  vector[NUM_OF_OBS] pr; //positive regression component
  vector[NUM_OF_OBS] rr; //regular regression component
  vector[NUM_OF_OBS] y_predicted; // response prediction

  // compute regression
  if (NUM_OF_PR > 0)
    pr = PR_MAT * pr_beta;
  else
    pr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_RR>0)
    rr = RR_MAT * rr_beta;
  else
    rr = rep_vector(0, NUM_OF_OBS);
  y_predicted = pr + rr;
}
model {
  // likelihood
  RESPONSE ~ student_t(NU, y_predicted, obs_sigma);

  // regression prior
  // see these references for details
  // 1. https://jrnold.github.io/bayesian_notes/shrinkage-and-regularized-regression.html
  // 2. https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#33_wide_weakly_informative_prior
  if (NUM_OF_PR > 0) {
    if (REG_PENALTY_TYPE== 0) {
      // fixed penalty ridge
      pr_beta ~ normal(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      pr_beta ~ double_exponential(PR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      //weak prior for sigma
      for(i in 1:NUM_OF_PR) {
        pr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0,];
      }
      //weak prior for betas
      pr_beta ~ normal(PR_BETA_PRIOR, pr_sigma);
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
      //weak prior for sigma
      for(i in 1:NUM_OF_RR) {
        rr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0,];
      }
      //weak prior for betas
      rr_beta ~ normal(RR_BETA_PRIOR, rr_sigma);
    }
  }
}