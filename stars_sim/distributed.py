"""Simulated stars-data experiment.

Execute with:
    $ python run.py <filename>
where <filename> is the desired name of the result '.npz' file. If <filename> is
omitted, the default filename 'res.npz' is used. After running this skript,
the script experimen/plot_res.py can be used to plot the results.

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
J = 3600            # Number of hierarchical groups
K = 60              # Number of sites
NPG = 20            # Number of observations per group (constant or [min, max])

# ====== Set parameters ========================================================
# Model parameters
MU = 270
TAU = 148
BETA = 250
SIGMA = 490
# Simulating parameters
X_MU = 200
X_STD = 100

# ====== Prior =================================================================
# Prior for phi = log(mu, tau, beta, sigma)
M0 = np.log([250, 250, 250, 250])
V0 = np.array([1.5,1.5,1.5,1.5])**2

# ====== Sampling parameters ===================================================
CHAINS = 4
ITER = 800
WARMUP = 400
THIN = 2

# ====== Number of EP iterations ===============================================
EP_ITER = 4

# ====== Tilted distribution precision estimate method =========================
# Available options are 'sample' and 'olse', see class serial.Master.
PREC_ESTIM = 'sample'

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
    alpha_j = MU + rnd_data.randn(J)*TAU
    phi_true = np.log([MU, TAU, BETA, SIGMA])
    dphi = 4  # Number of shared parameters
    
    # Simulate data
    # Truncated normal rejection sampling
    X = X_MU + rnd_data.randn(N)*X_STD
    xneg = X<0
    while np.any(xneg):
        X[xneg] = X_MU + rnd_data.randn(np.count_nonzero(xneg))*X_STD
        xneg = X<0
    f = alpha_j[j_ind] + X*BETA
    y = f + rnd_data.randn(N)*SIGMA
    yneg = y<0
    while np.any(yneg):
        y[yneg] = f[yneg] + rnd_data.randn(np.count_nonzero(yneg))*SIGMA
        yneg = y<0
    
    # ------------------------------------------------------
    #     Prior
    # ------------------------------------------------------
    
    # Moment parameters of the prior (transposed in order to get F-contiguous)
    S0 = np.diag(V0).T
    m0 = M0
    # Natural parameters of the prior
    Q0 = np.diag(np.ones(dphi)/V0).T
    r0 = M0/V0
    prior = {'Q':Q0, 'r':r0}
    
    # ------------------------------------------------------
    #     Distributed EP
    # ------------------------------------------------------
    
    print "Distributed model..."
    
    # Options for the ep-algorithm see documentation of dep.serial.Master
    options = {
        'seed'       : SEED_MCMC,
        'init_prev'  : True,
        'prec_estim' : PREC_ESTIM,
        'chains'     : CHAINS,
        'iter'       : ITER,
        'warmup'     : WARMUP,
        'thin'       : THIN
    }
    
    # Temp fix for the RandomState seed problem with pystan in 32bit Python
    options['tmp_fix_32bit'] = TMP_FIX_32BIT
    
    model = load_stan('model')
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
        dep_master = Master(
            model,
            X,
            y,
            A_k={'J': np.ones(K, dtype=np.int64)},
            A_n={'j_ind': np.ones(N, dtype=np.int64)},
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
        dep_master = Master(
            model,
            X,
            y,
            A_k={'J': np.ones(K, dtype=np.int64)},
            A_n={'j_ind': np.ones(N, dtype=np.int64)},
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
    
    print "Distributed model sampled:"
    print "    exp(phi) = {}".format(np.array2string(np.exp(m_mix), precision=1))
    print "True values:"
    print "    exp(phi) = {}".format([MU, TAU, BETA, SIGMA])
    
    # ------------------------------------------------------
    #     Save results
    # ------------------------------------------------------
    
    np.savez(filename,
        seed_data=SEED_DATA,
        seed_mcmc=SEED_MCMC,
        J=J,
        K=K,
        Nj=Nj,
        N=N,
        dphi=dphi,
        niter=EP_ITER,
        m0=M0,
        V0=V0,
        phi_true=phi_true,
        m_phi=m_phi,
        var_phi=var_phi,
        m_mix=m_mix,
        var_mix=var_mix
    )


if __name__ == '__main__':
    if len(os.sys.argv) == 2:
        main(os.sys.argv[1])
    else:
        main()



