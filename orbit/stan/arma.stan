
// ------   ARMA mode  -------
// this is an ARMA model which can handle aditional linear predictors 
// these predictors can be estimated either before or with the ARMA paramters.
// Note ARMA ~= ARIMA no differencing is done here 


// --- WBIC related work ---
// Conduct MCMC sampling at temperature t_star; pi(m)L(m)^{1/t_star}
// return the log probability (log_prob) of each observation
// this will be done in the .stan code

// make the common labels  
 data {
      int<lower=0> P;
      int<lower=0> Q;
      int<lower=0> NUM_OF_OBS;
      real<lower=0> T_STAR;
      real RESPONSE[NUM_OF_OBS]; 
      // to avoid having to do every lag 
      int LAG_AR[P]; 
      int LAG_MA[Q];
      // Do you want to do the LM first; i.e., use the residuals for the ARMA or run ARMA on y directly?
      int LM_FIRST; 

    }
    
    transformed data {
    real watanabe_beta;
    int N;
    real y[NUM_OF_OBS];

    watanabe_beta = 1.0/T_STAR; // the sampling temp  
    y = RESPONSE;
    N = NUM_OF_OBS;
    
    }
    
    parameters {
      real mu;
      real<lower=-1,upper=1> rho[P];
      real<lower=-1,upper=1> theta[Q];
      real<lower=0> obs_sigma;
    }
    
    transformed parameters {
        vector[N] yhat;  // full prediction
        vector[N] log_prob; // log prob of each observation
        vector[N] err;   // error for time t for the MA model
        vector[N] resid; // the residual of the linear model
        vector[N] level_hat;  // the prediction of the linear model
        vector[N] arhat; // the prediction of the AR model 
        vector[N] mahat; // the prediction of the MA  
        for (i in 1:N) {
            // initialize all the contributions 
            arhat[i] = 0; 
            mahat[i] = 0; 
            level_hat[i] = mu; // add the constant
            // lm_first = 1 means resid = y - X beta
            // lm_first = 0 means resid = y
            resid[i] = y[i] - LM_FIRST*level_hat[i]; // get the residuals from the linear model
            
            for (p in 1:P) {  // add in the ar terms 
              if ( i > LAG_AR[p] ) {
                  arhat[i] += rho[p] * resid[i-LAG_AR[p]];
                  }
              }
            for (q in 1:Q) { // add in the MA terms 
              if ( i > LAG_MA[q] ) {
                  mahat[i] += theta[q]*err[i-LAG_MA[q]] ;
                  }
              }
            // 
            
            // lm_first = 1 means error = 0 - ar - ma (1-lm_first = 0)
            // lm_first = 0 means resid = y - yhat = y - lm - ar - ma (1-lm_first = 1)
            err[i] = (1-LM_FIRST)*y[i] - (1-LM_FIRST)*level_hat[i] - arhat[i] - mahat[i]; // get the error 
            
            yhat[i] = level_hat[i] + arhat[i] + mahat[i]; // the full prediction

            log_prob[i] = normal_lpdf(y[i]|yhat[i], obs_sigma); // the log probs 
            }

    }
    
    
    model {
      
      for (i in 1:N) 
        target += watanabe_beta*log_prob[i];

    
    }