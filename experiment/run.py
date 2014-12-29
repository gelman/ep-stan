"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent dir to sys.path if not present already. This is only done because
# of easy importing of the package dep. Adding the parent directory into the
# PYTHONPATH works as well.
parent_dir = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir))
# Double check that the package is in the parent directory
if os.path.exists(os.path.join(parent_dir, 'dep')):
    if parent_dir not in os.sys.path:
        os.sys.path.insert(0, parent_dir)

from dep.serial import Master
from dep.util import compare_plot, load_stan, suppress_stdout

# Use seed = None for random seed
seed = 1
rand_state = np.random.RandomState(seed=seed)


# ------------------------------------------------------------------------------
#     Simulate data
# ------------------------------------------------------------------------------

# Model definition (j = 1 ... J):
# y_j ~ bernoulli_logit(alpha_j + x_j * beta)
# Local parameter alpha_j ~ N(0, sigma2_a)
# Shared parameter beta ~ N(0,sigma2_b)
# Hyperparameter sigma2_a ~ log-N(0,sigma2_aH)
# Fixed sigma2_b, sigma2_aH
# phi = [log(sqrt(sigma2_a)), beta]

# Parameters
J = 50                               # number of groups
Nj = 50*np.ones(J, dtype=np.int64)   # number of observations per group
N = np.sum(Nj)                       # number of observations
K = 50                               # number of inputs
# Observation index limits for J groups
jj_lim = np.concatenate(([0], np.cumsum(Nj)))
# Group indices for each sample
jj = np.empty(N, dtype=np.int64)
for j in xrange(J):
    jj[jj_lim[j]:jj_lim[j+1]] = j

# Assign fixed parameters
sigma2_a = 2**2
beta = rand_state.randn(K)
phi_true = np.append(0.5*np.log(sigma2_a), beta)
dphi = K+1  # Number of shared parameters

# Simulate
alpha_j = rand_state.randn(J)*np.sqrt(sigma2_a)
X = rand_state.randn(N,K)
y = X.dot(beta)
for j in range(J):
    y[jj_lim[j]:jj_lim[j+1]] += alpha_j[j]
y = 1/(1+np.exp(-y))
y = (rand_state.rand(N) < y).astype(int)


# ------------------------------------------------------------------------------
#     Prior
# ------------------------------------------------------------------------------

# Priors for sigma2_a
m0_a = np.log(1)
V0_a = np.log(5)**2
# Prior for beta
m0_b = 0
V0_b = 1**2
# Moment parameters of the prior (transposed in order to get F-contiguous)
S0 = np.diag(np.append(V0_a, np.ones(K)*V0_b)).T
r0 = np.append(m0_a, np.ones(K)*m0_b)
# Natural parameters of the prior
Q0 = np.diag(np.append(1./V0_a, np.ones(K)/V0_b)).T
r0 = np.append(m0_a/V0_a, np.ones(K)*(m0_b/V0_b))
prior = {'Q':Q0, 'r':r0}


# ------------------------------------------------------------------------------
#     Distributed EP
# ------------------------------------------------------------------------------

print "Distributed model."

# Options for the ep-algorithm see documentation of dep.serial.Master
options = {
    'seed'      : rand_state,
    'init_prev' : True,
    'chains'    : 4,
    'iter'      : 400,
    'warmup'    : 100,
    'thin'      : 2
}

# Temp fix for the RandomState seed problem with pystan in 32bit Python
# Uncomment the following line with 32bit Python:
#options['tmp_fix_32bit'] = True

# Create the Master instance
dep_master = Master('site_model', X, y, group_sizes=Nj,
                    prior=prior, **options)

# Run the algorithm for `niter` iterations
niter = 6
print "Run distributed EP algorithm for {} iterations.".format(niter)
m_phi, var_phi = dep_master.run(niter)
print "Form the final approximation by mixing the samples from all the sites."
S_mix, m_mix = dep_master.mix_samples()
var_mix = np.diag(S_mix)

print "Distributed model sampled."


# ------------------------------------------------------------------------------
#     Full model
# ------------------------------------------------------------------------------

print "\nFull model."

full_model = load_stan('full_model')
# In the following S0 is transposed in order to get C-contiguous
data = dict(N=N, K=K, J=J, X=X, y=y, jj=jj+1, mu_prior=r0, Sigma_prior=S0.T)
with suppress_stdout():
    fit = full_model.sampling(data=data, seed=rand_state.randint(2**31-1),
                              chains=4, iter=800, warmup=400, thin=2)
samp = fit.extract(pars='phi')['phi']
m_phi_full = samp.mean(axis=0)
var_phi_full = samp.var(axis=0, ddof=1)

print "Full model sampled."


# ------------------------------------------------------------------------------
#     Save results
# ------------------------------------------------------------------------------

if True:
    np.savez('res.npz',
        seed=seed,
        J=J,
        Nj=Nj,
        N=N,
        K=K,
        dphi=dphi,
        niter=niter,
        m0_a=m0_a,
        V0_a=V0_a,
        m0_b=m0_b,
        V0_b=V0_b,
        phi_true=phi_true,
        m_phi=m_phi,
        var_phi=var_phi,
        m_mix=m_mix,
        var_mix=var_mix,
        m_phi_full=m_phi_full,
        var_phi_full=var_phi_full
    )


# ------------------------------------------------------------------------------
#     Plot
# ------------------------------------------------------------------------------

if False:
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

    # Full vs distributed
    compare_plot(m_phi_full, m_mix,
                 a_err=1.96*np.sqrt(var_phi_full), b_err=1.96*np.sqrt(var_mix),
                 a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
                 b_label='Estimased from the dep model ($\pm 1.96 \sigma$)')

    plt.show()



