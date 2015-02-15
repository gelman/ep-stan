
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 2b with single group

data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    vector[2] mu_phi;
    matrix[2,2] Omega_phi;
}
parameters {
    vector[2] phi;
    real eta;
    vector[D] etb;
}
transformed parameters {
    real alpha;
    vector[D] beta;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    sigma_a <- exp(phi[1]);
    sigma_b <- exp(phi[2]);
    alpha <- eta * sigma_a;
    beta <- etb * sigma_b;
}
model {
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    etb ~ normal(0, 1);
    y ~ bernoulli_logit(alpha + X * beta);
}

