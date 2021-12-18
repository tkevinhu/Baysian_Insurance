/* Stan Logistic Regression - Model 2 */

data {
  int<lower=1> N;      // number of observations
  int<lower=1> J;      // number of plan groups
  vector[N] ldur;      // log duration     
  vector[N] lage;      // log age
  vector[N] lage2;     // log age squared
  int<lower=1,upper=J> plan[N]; // plan number  
  int<lower=0, upper=1> y [N]; //boolean 1=claim, 0=no
}
parameters {
  // coef on the each record
  vector[J] beta_0; //pooled intercept
  real beta_1; //log age coeff
  real beta_2; //log age squared coeff
  real beta_3; //log dur coeff
  //pooled coefficient
  real mu_beta_0;
  real<lower=0> sigma_beta_0;
}
model {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = beta_0[plan[i]] + beta_1 * lage[i] + beta_2 * lage2[i] + beta_3 * ldur[i];
  
  //priors, all N(0,1)
  beta_1 ~ normal(0,1);
  beta_2 ~ normal(0,1);
  beta_3 ~ normal(0,1);
  mu_beta_0 ~ normal(0,1);
  sigma_beta_0 ~ normal(0,1);
  
  //pooled intercepts
  beta_0 ~ normal(mu_beta_0, sigma_beta_0);
  
  //likelihood
  y ~ bernoulli_logit(y_hat);
}
generated quantities {
  vector[N] y_rep; // replications from posterior predictive dist

  for (n in 1:N) {
    real y_hat_n = beta_0[plan[n]] + beta_1 * lage[n] + beta_2 * lage2[n] + beta_3 * ldur[n];
    y_rep[n] = inv_logit(y_hat_n);
  }
}
