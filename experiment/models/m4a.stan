
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 4a

data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> J;
    matrix[N,D] X;
    real y[N];
    int<lower=1,upper=J> j_ind[N];
    vector[2*D+3] mu_phi;
    matrix[2*D+3,2*D+3] Omega_phi;
}
parameters {
    vector[2*D+3] phi;
    vector[J] eta;
    vector[D] etb[J];
}
transformed parameters {
    real<lower=0> sigma;
    vector[J] alpha;
    real mu_a;
    real<lower=0> sigma_a;
    vector[D] beta[J];
    vector[D] mu_b;
    vector<lower=0>[D] sigma_b;
    sigma <- exp(phi[1]);
    mu_a <- phi[2];
    sigma_a <- exp(phi[3]);
    alpha <- mu_a + eta * sigma_a;
    mu_b <- segment(phi, 4, D);
    sigma_b <- exp(tail(phi, D));
    for (j in 1:J){
        beta[j] <- mu_b + etb[j] .* sigma_b;
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
    y ~ normal(f, sigma);
}

