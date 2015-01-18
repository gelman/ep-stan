
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

data {
    int<lower=1> N;
    int<lower=1> J;
    vector<lower=0>[N] X;
    real<lower=0> y[N];
    int<lower=1,upper=J> j_ind[N];
    vector[4] mu_phi;
    matrix[4,4] Omega_phi;
}
parameters {
    vector[4] phi;
    vector[J] eta;
}
transformed parameters {
    vector[J] alpha;
    real<lower=0> mu;
    real<lower=0> tau;
    real<lower=0> beta;
    real<lower=0> sigma;
    mu <- exp(phi[1]);
    tau <- exp(phi[2]);
    beta <- exp(phi[3]);
    sigma <- exp(phi[4]);
    alpha <- mu + eta*tau;
}
model {
    vector[N] f;
    eta ~ normal(0, 1);
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    for (n in 1:N){
        f[n] <- alpha[j_ind[n]] + X[n]*beta;
    }
    y ~ normal(f, sigma);
}

