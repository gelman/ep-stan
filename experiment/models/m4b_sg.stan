
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 4b single group

data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    vector[2*D+2] mu_phi;
    matrix[2*D+2,2*D+2] Omega_phi;
}
parameters {
    vector[2*D+2] phi;
    real eta;
    vector[D] etb;
}
transformed parameters {
    real alpha;
    real mu_a;
    real<lower=0> sigma_a;
    vector[D] beta;
    vector[D] mu_b;
    vector<lower=0>[D] sigma_b;
    sigma_a <- exp(phi[2]);
    mu_a <- phi[1];
    alpha <- mu_a + eta * sigma_a;
    mu_b <- segment(phi, 3, D);
    sigma_b <- exp(tail(phi, D));
    beta <- mu_b + etb .* sigma_b;
}
model {
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    etb ~ normal(0, 1);
    y ~ bernoulli_logit(alpha + X * beta);
}

