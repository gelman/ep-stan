
from pystan import StanModel
import pickle

# Released under licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.
#
# The most recent version of the code can be found on GitHub:
# https://github.com/gelman/ep-stan

# -----------------------------------------
# Compile stan model for the tilted moments
# -----------------------------------------
model_name = 'hier_log'
model_code = """
data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    int<lower=0,upper=1> y[N];
    vector[K+1] mu_cavity;
    cov_matrix[K+1] Sigma_cavity;
}
parameters {
    vector[K+1] phi;
    real eta;
}
transformed parameters {
    real alpha;
    real<lower=0> sigma_a;
    sigma_a <- exp(phi[1]);
    alpha <- eta * sigma_a;
}
model {
    eta ~ normal(0, 1);
    phi ~ multi_normal(mu_cavity, Sigma_cavity);
    y ~ bernoulli_logit(alpha + X * tail(phi, K));
}
"""
# Build the model
sm = StanModel(model_code=model_code, model_name=model_name)
# Save it
with open(model_name+'.pkl', 'wb') as f:
    pickle.dump(sm, f)
