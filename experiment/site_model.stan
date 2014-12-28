
data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    int<lower=0,upper=1> y[N];
    vector[K+1] mu_cavity;
    cov_matrix[K+1] Sigma_cavity;
}
parameters {
    vector[K+1] phi;
    real eta;
}
transformed parameters {
    real alpha;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
}
model {
    eta ~ normal(0, 1);
    phi ~ multi_normal(mu_cavity, Sigma_cavity);
    y ~ bernoulli_logit(alpha + X * tail(phi, K));
}

