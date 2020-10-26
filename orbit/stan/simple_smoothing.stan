data {
  int <lower=2> N_OBS;
  vector[N_OBS] RESPONSE;
  int <lower=1> HORIZON;
  real <lower=0> SDY;
  int <lower=1> SEASONALITY;
}
parameters {
  real<lower=1e-4, upper=1-1e-4>lev_sm;
  real<lower=0,upper=SDY> sigma;
  vector[SEASONALITY - 1] init_s;
}
transformed parameters {
  vector[N_OBS - HORIZON] lev;
  vector[N_OBS - HORIZON] yhat;
  vector[N_OBS - HORIZON + SEASONALITY] sea;
  real init_s_sum;

  init_s_sum = 0;
  if (SEASONALITY > 1) {
    for (t in 1:SEASONALITY - 1) {
      sea[t] = init_s[t];
      init_s_sum += init_s[t];
    }
    sea[SEASONALITY] = 1 - init_s_sum;
    sea[SEASONALITY + 1] = sea[1];
    lev[1] = RESPONSE[1] - sea[1];
  } else {
    lev[1] = RESPONSE[1];
    sea = rep_vector(0, N_OBS - HORIZON + SEASONALITY);
  }
  
  for (t in 2:N_OBS - HORIZON) {
    yhat[t] = lev[t-1] + sea[t];
    sea[t + SEASONALITY] = sea[t];
    lev[t] = (1 - lev_sm) * lev[t-1] + lev_sm * (RESPONSE[t]);
  }
}
model {
  for (t in 2:N_OBS - HORIZON) {
    RESPONSE[t + HORIZON] ~ normal(yhat[t], sigma);
  }
  sea ~ normal(0, SDY * 0.3);
}
