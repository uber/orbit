data {
  int <lower=2> N_OBS;
  vector[N_OBS] RESPONSE;
  int <lower=1> HORIZON;
  real <lower=0> SDY;
}
parameters {
  real<lower=1e-4, upper=1-1e-4>lev_sm;
  real<lower=0,upper=SDY> sigma;
}
transformed parameters {
  vector[N_OBS - HORIZON] lev;
  // matrix[N_OBS - 1, HORIZON] yhat;
  lev[1] = RESPONSE[1];
  for (t in 2:N_OBS - HORIZON) {
    lev[t] = (1 - lev_sm) * lev[t-1] + lev_sm * (RESPONSE[t]);
  }
}
model {
  for (t in 2:N_OBS - HORIZON) {
    RESPONSE[t + HORIZON] ~ normal(lev[t-1], sigma);
  }
}
