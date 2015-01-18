"""Simulated stars-data experiment using full model.

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

# from dep.serial import Master
from dep.util import load_stan


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
ITER = 1600
WARMUP = 800
THIN = 2

# ====== 32bit Python ? ========================================================
# Temp fix for the RandomState seed problem with pystan in 32bit Python. Set
# the following to True if using 32bit Python.
TMP_FIX_32BIT = False

# ------------------------------------------------------------------------------
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


def main(filename='res_full.npz'):
    
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
    #     Full model
    # ------------------------------------------------------
    
    print "Full model..."
    
    # Set seed
    rnd_mcmc = np.random.RandomState(seed=SEED_MCMC)
    
    data = dict(
        N=N,
        J=J,
        X=X,
        y=y,
        j_ind=j_ind+1,
        mu_phi=m0,
        Omega_phi=Q0.T    # Q0 transposed in order to get C-contiguous
    )
    
    # Sample and extract parameters
    model = load_stan('model')
    fit = model.sampling(
        data=data,
        seed=(rnd_mcmc.randint(2**31-1) if TMP_FIX_32BIT else rnd_mcmc),
        chains=CHAINS,
        iter=ITER,
        warmup=WARMUP,
        thin=THIN
    )
    samp = fit.extract(pars='phi')['phi']
    m_phi_full = samp.mean(axis=0)
    var_phi_full = samp.var(axis=0, ddof=1)
    
    print "Full model sampled:"
    print "    exp(phi) = {}" \
          .format(np.array2string(np.exp(m_phi_full), precision=1))
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
        m0=M0,
        V0=V0,
        phi_true=phi_true,
        m_phi_full=m_phi_full,
        var_phi_full=var_phi_full
    )


if __name__ == '__main__':
    if len(os.sys.argv) == 2:
        main(os.sys.argv[1])
    else:
        main()



