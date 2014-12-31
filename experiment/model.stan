
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

data {
    int<lower=0> N;
    int<lower=0> D;
    int<lower=0> J;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    int<lower=1,upper=J> j_ind[N];
    vector[D+1] mu_phi;
    cov_matrix[D+1] Sigma_phi;
}
parameters {
    vector[D+1] phi;
    vector[J] eta;
}
transformed parameters {
    vector[J] alpha;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
}
model {
    vector[N] f;
    eta ~ normal(0, 1);
    phi ~ multi_normal(mu_phi, Sigma_phi);
    f <- X * tail(phi, D);
    for (n in 1:N){
        f[n] <- alpha[j_ind[n]] + f[n];
    }
    y ~ bernoulli_logit(f);
}

