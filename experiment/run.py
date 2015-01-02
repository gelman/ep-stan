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
# >>>>>>>>>>>>> Configurations start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------------------------------------------------------------

# ====== Seed ==================================================================
# Use SEED = None for random seed
SEED_DATA = 0       # Seed for simulating the data
SEED_MCMC = 0       # Seed for the inference algorithms

# ====== Data size =============================================================
J = 50              # Number of hierarchical groups
D = 50              # Number of inputs
K = 22              # Number of sites
NPG = [40,60]       # Number of observations per group (constant or [min, max])

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
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


def main(filename='res.npz'):
    
    # ------------------------------------------------------
    #     Simulate data
    # ------------------------------------------------------
    
    # Set seed
    rnd_data = np.random.RandomState(seed=SEED_DATA)
    
    # Parameters
    # Number of observations for each group
    if hasattr(NPG, '__getitem__') and len(NPG) == 2:
        Nj = rnd_data.randint(NPG[0],NPG[1]+1, size=J)
    else:
        Nj = NPG*np.ones(J, dtype=np.int64)
    # Total number of observations
    N = np.sum(Nj)
    # Observation index limits for J groups
    j_lim = np.concatenate(([0], np.cumsum(Nj)))
    # Group indices for each sample
    j_ind = np.empty(N, dtype=np.int64)
    for j in xrange(J):
        j_ind[j_lim[j]:j_lim[j+1]] = j
    
    # Assign parameters
    if SIGMA_A is None:
        sigma_a = np.exp(rnd_data.randn()*SIGMA_AH)
    else:
        sigma_a = SIGMA_A
    if BETA is None:
        beta = rnd_data.randn(D)*SIGMA_B
    else:
        beta = BETA
    alpha_j = rnd_data.randn(J)*sigma_a
    phi_true = np.append(np.log(sigma_a), beta)
    dphi = D+1  # Number of shared parameters
    
    # Simulate data
    X = rnd_data.randn(N,D)
    y = alpha_j[j_ind] + X.dot(beta)
    y = 1/(1+np.exp(-y))
    y = (rnd_data.rand(N) < y).astype(int)
    
    # ------------------------------------------------------
    #     Prior
    # ------------------------------------------------------
    
    # Moment parameters of the prior (transposed in order to get F-contiguous)
    S0 = np.diag(np.append(V0_A, np.ones(D)*V0_B)).T
    m0 = np.append(M0_A, np.ones(D)*M0_B)
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
        'seed'      : SEED_MCMC,
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
        # Combine smallest pairs of consecutive groups until K has been reached
        Nk = Nj.tolist()
        Njd = (Nj[:-1]+Nj[1:]).tolist()
        Nj_k = [1]*J
        for _ in xrange(J-K):
            ind = Njd.index(min(Njd))
            if ind+1 < len(Njd):
                Njd[ind+1] += Nk[ind]
            if ind > 0:
                Njd[ind-1] += Nk[ind+1]
            Nk[ind] = Njd[ind]
            Nk.pop(ind+1)
            Njd.pop(ind)
            Nj_k[ind] += Nj_k[ind+1]
            Nj_k.pop(ind+1)
        Nk = np.array(Nk)                       # Number of samples per site
        Nj_k = np.array(Nj_k)                   # Number of groups per site
        j_ind_k = np.empty(N, dtype=np.int32)   # Within site group index
        k_lim = np.concatenate(([0], np.cumsum(Nj_k)))
        for k in xrange(K):
            for ji in xrange(Nj_k[k]):
                ki = ji + k_lim[k]
                j_ind_k[j_lim[ki]:j_lim[ki+1]] = ji        
        # Create the Master instance
        model = load_stan('model')
        dep_master = Master(
            model,
            X,
            y,
            A_k={'J':Nj_k},
            A_n={'j_ind':j_ind_k+1},
            site_sizes=Nk,
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
        # Split biggest groups until enough sites are formed
        ppg = np.ones(J, dtype=np.int64)    # Parts per group
        Nj2 = Nj.astype(np.float)
        for _ in xrange(K-J):
            cur_max = Nj2.argmax()
            ppg[cur_max] += 1
            Nj2[cur_max] = Nj[cur_max]/ppg[cur_max]
        Nj2 = Nj//ppg
        rem = Nj%ppg
        # Form the number of samples for each site
        Nk = np.empty(K, dtype=np.int64)
        k = 0
        for j in xrange(J):
            for kj in xrange(ppg[j]):
                if kj < rem[j]:
                    Nk[k] = Nj2[j] + 1
                else:
                    Nk[k] = Nj2[j]
                k += 1
        # Create the Master instance
        model_single_group = load_stan('model_single_group')
        dep_master = Master(
            model_single_group,
            X,
            y,
            site_sizes=Nk,
            prior=prior,
            **options
        )
    else:
        raise ValueError("K cant be greater than number of samples")
    
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
    
    # Set seed
    rnd_mcmc = np.random.RandomState(seed=SEED_MCMC)
    
    data = dict(
        N=N,
        D=D,
        J=J,
        X=X,
        y=y,
        j_ind=j_ind+1,
        mu_phi=m0,
        Sigma_phi=S0.T    # S0 transposed in order to get C-contiguous
    )
    if model is None:
        model = load_stan('model')
    
    # Sample and extract parameters
    with suppress_stdout():
        fit = model.sampling(
            data=data,
            seed=(rnd_mcmc.randint(2**31-1) if TMP_FIX_32BIT else rnd_mcmc),
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
        seed_data=SEED_DATA,
        seed_mcmc=SEED_MCMC,
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



