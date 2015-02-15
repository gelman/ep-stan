
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 3a single group

data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    real y[N];
    vector[D+2] mu_phi;
    matrix[D+2,D+2] Omega_phi;
}
parameters {
    vector[D+2] phi;
    real eta;
    vector[D] etb;
}
transformed parameters {
    real<lower=0> sigma;
    real alpha;
    real<lower=0> sigma_a;
    vector[D] beta;
    vector<lower=0>[D] sigma_b;
    sigma <- exp(phi[1]);
    sigma_a <- exp(phi[2]);
    alpha <- eta * sigma_a;
    sigma_b <- exp(tail(phi, D));
    beta <- etb .* sigma_b;
}
model {
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    etb ~ normal(0, 1);
    y ~ normal(alpha + X * beta, sigma);
}

