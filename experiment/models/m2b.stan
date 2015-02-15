
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 2b

data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> J;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    int<lower=1,upper=J> j_ind[N];
    vector[2] mu_phi;
    matrix[2,2] Omega_phi;
}
parameters {
    vector[2] phi;
    vector[J] eta;
    vector[D] etb;
}
transformed parameters {
    vector[J] alpha;
    vector[D] beta;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    sigma_a <- exp(phi[1]);
    sigma_b <- exp(phi[2]);
    alpha <- eta * sigma_a;
    beta <- etb * sigma_b;
}
model {
    vector[N] f;
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    etb ~ normal(0, 1);
    f <- X * beta;
    for (n in 1:N){
        f[n] <- alpha[j_ind[n]] + f[n];
    }
    y ~ bernoulli_logit(f);
}

