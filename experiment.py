"""An experiment for distributed EP algorithm described in an article
"Expectation propagation as a way of life" (arXiv:1412.4869).

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from distributed_ep import DistributedEP
from util import compare_plot

# Use seed = None for random seed
RAND = np.random.RandomState(seed=0)


# ---------------
#  Simulate Data
# ---------------

# Parameters
J = 50                               # number of groups
Nj = 50*np.ones(J)                   # number of observations per group
N = np.sum(Nj)                       # number of observations
K = 50                               # number of inputs
# Observation index limits for J groups
iiJ = np.concatenate(([0], np.cumsum(Nj)))

# Model definition (j = 1 ... J):
# y_j ~ bernoulli_logit(alpha_j + x_j * beta)
# Local parameter alpha_j ~ N(0, sigma2_a)
# Shared parameter beta ~ N(0,sigma2_b)
# Hyperparameter sigma2_a ~ log-N(0,sigma2_aH)
# Fixed sigma2_b, sigma2_aH
# phi = [log(sqrt(sigma2_a)), beta]

# Assign fixed parameters
sigma2_a = 2**2
beta = RAND.randn(K)
phi_true = np.append(0.5*np.log(sigma2_a), beta)
dphi = K+1  # Number of shared parameters

# Simulate
alpha_j = RAND.randn(J)*np.sqrt(sigma2_a)
X = RAND.randn(N,K)
y = X.dot(beta)
for j in range(J):
    y[iiJ[j]:iiJ[j+1]] += alpha_j[j]
y = 1/(1+np.exp(-y))
y = (RAND.rand(N) < y).astype(int)


# ---------------
#     Prior
# ---------------
# Priors for sigma2_a
m0_a = np.log(1)
V0_a = np.log(5)**2
# Prior for beta
m0_b = 0
V0_b = 1**2
# Natural parameters of the prior
Q0 = np.diag(np.append(1./V0_a, np.ones(K)/V0_b)).T
r0 = np.append(m0_a/V0_a, np.ones(K)*(m0_b/V0_b))
prior = {'Q':Q0, 'r':r0}

# ---------------
#     EP-STAN
# ---------------

# Options for the ep-algorithm see documentation of DistributedEP
options = {
    'seed'      : RAND,
    'init_prev' : True,
    'chains'    : 4,
    'iter'      : 400,
    'warmup'    : 100,
    'thin'      : 2
}

# Temp fix for the RandomState seed problem with pystan in 32bit Python
# Uncomment the following line with 32bit Python:
# options['tmp_fix_32bit'] = True

# Create the model instance
model = DistributedEP('hier_log.pkl', X, y, group_sizes=Nj,
                      prior=prior, **options)

# Run the algorithm for 6 iterations
niter = 6
m_phi, var_phi = model.run(niter)
S_mix, m_mix = model.mix_samples()
var_mix = np.diag(S_mix)

# --------------
#      Plot
# --------------

# Mean and variance as a function of the iteration
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)
axs[0].plot(np.arange(niter+1), np.vstack((m_phi, m_mix)))
axs[0].set_ylabel('Mean of params')
axs[1].plot(np.arange(niter+1), np.sqrt(np.vstack((var_phi, var_mix))))
axs[1].set_ylabel('Std of params')
axs[1].set_xlabel('Iteration')

# Estimates vs true values
compare_plot(phi_true, m_mix, b_err=3*np.sqrt(var_mix),
             a_label='True values',
             b_label='Estimated values ($\pm 3 \sigma$)')

plt.show()



