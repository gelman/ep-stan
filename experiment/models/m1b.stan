
# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

# Model 1b

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
}
transformed parameters {
    vector[J] alpha;
    vector[D] beta;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
    beta <- tail(phi, D);
}
model {
    vector[N] f;
    phi ~ multi_normal_prec(mu_phi, Omega_phi);
    eta ~ normal(0, 1);
    f <- X * beta;
    for (n in 1:N){
        f[n] <- alpha[j_ind[n]] + f[n];
    }
    y ~ bernoulli_logit(f);
}

