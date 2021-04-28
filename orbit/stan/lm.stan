data {
  int<lower=0> N; // Number of observations/responses
  int<lower=0> P; // Number of covariates/regressors
  vector[N] y; // Vector of observations/responses
  matrix[N, P] X; // Design matrix / regressor matrix
  
}
parameters {
  real alpha; // Intercept
  vector[P] beta; // coefficients
  real<lower=0> sigma; // standard deviation of error
}
model {
  y ~ normal(alpha + X * beta, sigma);
}
