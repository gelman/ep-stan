data {
  int N;  // number of observations (1000)
  int J;  // number of groups       (100)
  int K;  // number of predictors   (5)
  
  int<lower=1,upper=J> jj[N];
  matrix[N,K] X;
  int<lower=0, upper=1> y[N];
}
parameters {
  vector[J] eta;
  vector[K+1] phi;
}
transformed parameters {
  vector[J] a;
  real<lower=0> sigma_a;
  
  sigma_a <- exp(phi[K+1]);
  a <- eta * sigma_a;
}
model {
  vector[N] lin_pred;
  vector[K] b;
  b <- head(phi, K);
  
  eta ~ normal(0, 1);
  b ~ normal(0, 1);
  phi[K+1] ~ normal(0, log(5));

  for (n in 1:N) {
    lin_pred[n] <- a[jj[n]] + X[n] * b;
  }
  y ~ bernoulli_logit(lin_pred);
}