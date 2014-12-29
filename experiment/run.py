"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Model definition (j = 1 ... J):
y_j ~ bernoulli_logit(alpha_j + x_j * beta)
Local parameter alpha_j ~ N(0, sigma2_a)
Shared parameter beta ~ N(0,sigma2_b)
Hyperparameter sigma2_a ~ log-N(0,sigma2_aH)
Fixed sigma2_b, sigma2_aH
phi = [log(sqrt(sigma2_a)), beta]

Execute with:
    $ python run.py <filename>
where <filename> is the desired name of the result '.npz' file. If <filename> is
omitted, the default filename 'res.npz' is used. After running this skript,
the script plot_res.py can be used to plot the results.

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
from dep.util import load_stan, suppress_stdout


# ------------------------------------------------------------------------------
#               Configurations start
# ------------------------------------------------------------------------------

# ====== Seed ==================================================================
SEED = 1            # Use SEED = None for random seed

# ====== Data size =============================================================
J = 50              # Number of groups
K = 50              # Number of inputs
NPG = 50            # Number of observations per group

# ====== Set parameters ========================================================
# If SIGMA2_A is None, it is sampled from N(0,SIGMA2_AH)
SIGMA2_A = 2**2
SIGMA2_AH = None
# If BETA is None, it is sampled from N(0,SIGMA2_B)
BETA = None
SIGMA2_B = 1**2

# ====== Prior =================================================================
# Priors for sigma2_a
M0_A = 0
V0_A = 2**2
# Prior for beta
M0_B = 0
V0_B = 1**2

# ====== Sampling parameters ===================================================
CHAINS = 4
ITER = 800
WARMUP = 400
THIN = 2

# ====== 32bit Python ? ========================================================
# Temp fix for the RandomState seed problem with pystan in 32bit Python. Set
# the following to True if using 32bit Python.
TMP_FIX_32BIT = False

# ------------------------------------------------------------------------------
#               Configurations end
# ------------------------------------------------------------------------------


def main(filename='res.npz'):
    
    # Set seed
    rand_state = np.random.RandomState(seed=SEED)

    # ------------------------------------------------------
    #     Simulate data
    # ------------------------------------------------------
    
    # Parameters
    Nj = NPG*np.ones(J, dtype=np.int64)  # Number of observations for each group
    N = np.sum(Nj)                       # Number of observations
    # Observation index limits for J groups
    jj_lim = np.concatenate(([0], np.cumsum(Nj)))
    # Group indices for each sample
    jj = np.empty(N, dtype=np.int64)
    for j in xrange(J):
        jj[jj_lim[j]:jj_lim[j+1]] = j

    # Assign fixed parameters
    if SIGMA2_A is None:
        sigma2_a = rand_state.randn()*np.sqrt(SIGMA2_AH)
    else:
        sigma2_a = SIGMA2_A
    if BETA is None:
        beta = rand_state.randn(K)*np.sqrt(SIGMA2_B)
    else:
        beta = BETA
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

    # ------------------------------------------------------
    #     Prior
    # ------------------------------------------------------
    
    # Moment parameters of the prior (transposed in order to get F-contiguous)
    S0 = np.diag(np.append(V0_A, np.ones(K)*V0_B)).T
    r0 = np.append(M0_A, np.ones(K)*M0_B)
    # Natural parameters of the prior
    Q0 = np.diag(np.append(1./V0_A, np.ones(K)/V0_B)).T
    r0 = np.append(M0_A/V0_A, np.ones(K)*(M0_B/V0_B))
    prior = {'Q':Q0, 'r':r0}

    # ------------------------------------------------------
    #     Distributed EP
    # ------------------------------------------------------

    print "Distributed model."

    # Options for the ep-algorithm see documentation of dep.serial.Master
    options = {
        'seed'      : rand_state,
        'init_prev' : True,
        'chains'    : CHAINS,
        'iter'      : ITER,
        'warmup'    : WARMUP,
        'thin'      : THIN
    }

    # Temp fix for the RandomState seed problem with pystan in 32bit Python
    options['tmp_fix_32bit'] = TMP_FIX_32BIT

    # Create the Master instance
    dep_master = Master('site_model', X, y, group_sizes=Nj,
                        prior=prior, **options)

    # Run the algorithm for `niter` iterations
    niter = 6
    print "Run distributed EP algorithm for {} iterations.".format(niter)
    m_phi, var_phi = dep_master.run(niter)
    print "Form the final approximation " \
          "by mixing the samples from all the sites."
    S_mix, m_mix = dep_master.mix_samples()
    var_mix = np.diag(S_mix)

    print "Distributed model sampled."

    # ------------------------------------------------------
    #     Full model
    # ------------------------------------------------------

    print "\nFull model."

    full_model = load_stan('full_model')
    # In the following S0 is transposed in order to get C-contiguous
    data = dict(N=N, K=K, J=J, X=X, y=y, jj=jj+1, mu_prior=r0, Sigma_prior=S0.T)
    with suppress_stdout():
        fit = full_model.sampling(data=data, seed=rand_state.randint(2**31-1),
                                  chains=4, iter=1000, warmup=500, thin=2)
    samp = fit.extract(pars='phi')['phi']
    m_phi_full = samp.mean(axis=0)
    var_phi_full = samp.var(axis=0, ddof=1)

    print "Full model sampled."

    # ------------------------------------------------------
    #     Save results
    # ------------------------------------------------------

    np.savez(filename,
        seed=SEED,
        J=J,
        Nj=Nj,
        N=N,
        K=K,
        dphi=dphi,
        niter=niter,
        m0_a=M0_A,
        V0_a=V0_A,
        m0_b=M0_B,
        V0_b=V0_B,
        phi_true=phi_true,
        m_phi=m_phi,
        var_phi=var_phi,
        m_mix=m_mix,
        var_mix=var_mix,
        m_phi_full=m_phi_full,
        var_phi_full=var_phi_full
    )


if __name__ == '__main__':
    if len(os.sys.argv) == 2:
        main(os.sys.argv[1])
    else:
        main()



