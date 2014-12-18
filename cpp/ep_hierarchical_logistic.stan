data {
  int N;
  int J;
  int K;
  int<lower=1,upper=J> jj[N];
  matrix[N,K] X;
  int<lower=0,upper=1> y[N];
  vector[K+1] mu_cavity;
  matrix[K+1,K+1] Sigma_cavity;
}
parameters {
  vector[J] eta;
  vector[K+1] phi;
}
transformed parameters {
  vector[J] a;
  real<lower=0> sigma_a;
  sigma_a <- exp(phi[K+1]);
  a <- eta*sigma_a;
}
model {
  vector[N] lin_pred;
  vector[K] b;
  eta ~ normal (0, 1);
  phi ~ multi_normal (mu_cavity, Sigma_cavity);
  b <- head (phi, K);
  for (n in 1:N){
    lin_pred[n] <- a[jj[n]] + X[n]*b;
  }
  y ~ bernoulli_logit (lin_pred);
}
