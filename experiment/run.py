"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Model definition (j = 1 ... J):
y_j ~ bernoulli_logit(alpha_j + x_j * beta)
Local parameter alpha_j ~ N(0,sigma_a)
Shared parameter beta ~ N(0,sigma_b)
Hyperparameter sigma_a ~ log-N(0,sigma_aH)
Fixed sigma_b, sigma_aH
phi = [log(sigma_a), beta]

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
J = 50              # Number of hierarchical groups
D = 50              # Number of inputs
NPG = 50            # Number of observations per group
K = 60              # Number of sites

# ====== Set parameters ========================================================
# If SIGMA_A is None, it is sampled from log-N(0,SIGMA_AH)
SIGMA_A = 2
SIGMA_AH = None
# If BETA is None, it is sampled from N(0,SIGMA_B)
BETA = None
SIGMA_B = 1

# ====== Prior =================================================================
# Prior for log(sigma_a)
M0_A = 0
V0_A = 1**2
# Prior for beta
M0_B = 0
V0_B = 1**2

# ====== Sampling parameters ===================================================
CHAINS = 4
ITER = 800
WARMUP = 400
THIN = 2

# ====== Number of EP iterations ===============================================
EP_ITER = 6

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
    #Nj = NPG*np.ones(J, dtype=np.int64)  # Number of observations for each group
    Nj = rand_state.randint(NPG-10, NPG+11, size=J)
    N = np.sum(Nj)                       # Number of observations
    # Observation index limits for J groups
    j_lim = np.concatenate(([0], np.cumsum(Nj)))
    # Group indices for each sample
    j_ind = np.empty(N, dtype=np.int64)
    for j in xrange(J):
        j_ind[j_lim[j]:j_lim[j+1]] = j
    
    # Assign fixed parameters
    if SIGMA_A is None:
        sigma_a = np.exp(rand_state.randn()*SIGMA_AH)
    else:
        sigma_a = SIGMA_A
    if BETA is None:
        beta = rand_state.randn(D)*SIGMA_B
    else:
        beta = BETA
    phi_true = np.append(np.log(sigma_a), beta)
    dphi = D+1  # Number of shared parameters
    
    # Simulate
    alpha_j = rand_state.randn(J)*sigma_a
    X = rand_state.randn(N,D)
    y = X.dot(beta)
    for j in range(J):
        y[j_lim[j]:j_lim[j+1]] += alpha_j[j]
    y = 1/(1+np.exp(-y))
    y = (rand_state.rand(N) < y).astype(int)
    
    # ------------------------------------------------------
    #     Prior
    # ------------------------------------------------------
    
    # Moment parameters of the prior (transposed in order to get F-contiguous)
    S0 = np.diag(np.append(V0_A, np.ones(D)*V0_B)).T
    r0 = np.append(M0_A, np.ones(D)*M0_B)
    # Natural parameters of the prior
    Q0 = np.diag(np.append(1./V0_A, np.ones(D)/V0_B)).T
    r0 = np.append(M0_A/V0_A, np.ones(D)*(M0_B/V0_B))
    prior = {'Q':Q0, 'r':r0}
    
    # ------------------------------------------------------
    #     Distributed EP
    # ------------------------------------------------------
    
    print "Distributed model..."
    
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
    
    model = None
    if K < 2:
        raise ValueError("K should be at least 2.")
    elif K < J:
        # ---- Many groups per site ----
        # Distribute and combine J groups to K sites
        # Indexes of the groups for each site
        js_k = tuple(
            np.arange((J//K+1)*k, (J//K+1)*(k+1))
            if k < J%K else
            np.arange(J//K*k, J//K*(k+1)) + J%K
            for k in xrange(K)
        )
        Nj_k = np.empty(K, dtype=np.int64)  # Number of groups for each site
        Ns_k = np.empty(K, dtype=np.int64)  # Number of samples for each site
        j_ind_k = j_ind.copy()              # Within site group index
        for k in xrange(K):
            js = js_k[k]
            Nj_k[k] = len(js)
            Ns_k[k] = Nj[js[0]:js[-1]+1].sum()
            j_ind_k[j_lim[js[0]]:j_lim[js[-1]+1]] -= js[0]
        # Create the Master instance
        model = load_stan('model')
        dep_master = Master(
            model,
            X,
            y,
            A_k={'J':Nj_k},
            A_n={'j_ind':j_ind_k+1},
            site_sizes=Ns_k,
            prior=prior,
            **options
        )
    elif K == J:
        # ---- One group per site ----
        # Create the Master instance
        model_single_group = load_stan('model_single_group')
        dep_master = Master(
            model_single_group,
            X,
            y,
            site_sizes=Nj,
            prior=prior,
            **options
        )
    elif K <= N:
        # ---- Multiple sites per group ----
        # Distribute K sites to J groups and split the samples in each group
        # accordingly. Works surely only if K%J < Nj for all groups.
        Ns_k = np.empty(K, dtype=np.int64)  # Number of samples per site
        for j in xrange(J):
            if j < K%J:
                for ki in xrange(K//J+1):
                    k = ki + (K//J+1)*j
                    if ki < Nj[j]%(K//J+1):
                        Ns_k[k] = Nj[j]//(K//J+1)+1
                    else:
                        Ns_k[k] = Nj[j]//(K//J+1)
            else:
                for ki in xrange(K//J):
                    k = ki + (K//J)*j + K%J
                    if ki < Nj[j]%(K//J):
                        Ns_k[k] = Nj[j]//(K//J)+1
                    else:
                        Ns_k[k] = Nj[j]//(K//J)
        # Create the Master instance
        model_single_group = load_stan('model_single_group')
        dep_master = Master(
            model_single_group,
            X,
            y,
            site_sizes=Ns_k,
            prior=prior,
            **options
        )
    else:
        raise ValueError("K cant be greater than J*NPG")
    
    # Run the algorithm for `EP_ITER` iterations
    print "Run distributed EP algorithm for {} iterations.".format(EP_ITER)
    m_phi, var_phi = dep_master.run(EP_ITER)
    print "Form the final approximation " \
          "by mixing the samples from all the sites."
    S_mix, m_mix = dep_master.mix_samples()
    var_mix = np.diag(S_mix)
    
    print "Distributed model sampled."
    
    # ------------------------------------------------------
    #     Full model
    # ------------------------------------------------------
    
    print "Full model..."
    
    data = dict(
        N=N,
        D=D,
        J=J,
        X=X,
        y=y,
        j_ind=j_ind+1,
        mu_phi=r0,
        Sigma_phi=S0.T    # S0 transposed in order to get C-contiguous
    )
    if model is None:
        model = load_stan('model')
    with suppress_stdout():
        fit = model.sampling(
            data=data,
            seed=(rand_state.randint(2**31-1) if TMP_FIX_32BIT else rand_state),
            chains=4,
            iter=1000,
            warmup=500,
            thin=2
        )
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
        D=D,
        dphi=dphi,
        niter=EP_ITER,
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



