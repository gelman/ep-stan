
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 1b with single group

data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    vector[D+1] mu_phi;
    matrix[D+1,D+1] Omega_phi;
}
parameters {
    vector[D+1] phi;
    real eta;
}
transformed parameters {
    real alpha;
    vector[D] beta;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
    beta <- tail(phi, D);
}
model {
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    y ~ bernoulli_logit(alpha + X * beta);
}

