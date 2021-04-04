data {
  int<lower=0> N;
  int<lower=0> P;
  vector[N] y;
  matrix[N, P] X;
  
}
parameters {
  real alpha;
  vector[P] beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + X * beta, sigma);
}



