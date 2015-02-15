
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 3b

data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> J;
    matrix[N,D] X;
    int<lower=0,upper=1> y[N];
    int<lower=1,upper=J> j_ind[N];
    vector[D+1] mu_phi;
    matrix[D+1,D+1] Omega_phi;
}
parameters {
    vector[D+1] phi;
    vector[J] eta;
    vector[D] etb[J];
}
transformed parameters {
    vector[J] alpha;
    real<lower=0> sigma_a;
    vector[D] beta[J];
    vector<lower=0>[D] sigma_b;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
    sigma_b <- exp(tail(phi, D));
    for (j in 1:J){
        beta[j] <- etb[j] .* sigma_b;
    }
}
model {
    vector[N] f;
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    for (j in 1:J){
        etb[j] ~ normal(0, 1);
    }
    for (n in 1:N){
        f[n] <- alpha[j_ind[n]] + X[n] * beta[j_ind[n]];
    }
    y ~ bernoulli_logit(f);
}

