/* Stan Logistic Regression - Model 1 */

data {
  int<lower=1> N;      // number of observations
  vector[N] ldur;      //log duration     
  vector[N] lage;      //log age
  vector[N] lage2;     //log age squared
  int<lower=0, upper=1> y [N]; //boolean 1=claim, 0=no
}
parameters {
  // coef on the each record
  real beta_0; //intercept
  real beta_1; //log age coeff
  real beta_2; //log age squared coeff
  real beta_3; //log dur coeff
}
model {
  //priors, all N(0,1)
  beta_0 ~ normal(0,1);
  beta_1 ~ normal(0,1);
  beta_2 ~ normal(0,1);
  beta_3 ~ normal(0,1);
  
  //likelihood
  y ~ bernoulli_logit(beta_0 + beta_1 * lage + beta_2 * lage2 + beta_3 * ldur);
}
