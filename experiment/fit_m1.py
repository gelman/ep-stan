"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Group index j = 1 ... J
Model m1:
    y_j ~ bernoulli_logit(alpha_j + beta * x_j)
    alpha_j ~ N(0,sigma_a)
    beta ~ N(0,sigma_b)
    sigma_a ~ log-N(0,sigma_aH)
    Fixed sigma_aH, sigma_b
    phi = [log(sigma_a), beta]

Execute with:
    $ python fit_<model_name>.py [mtype]
where argument mtype can be either `full` or `distributed`. If type is omitted,
both models are fit. The results are saved into files res_f_<model_name>.npz and
res_d_<model_name>.npz into the folder results respectively.

After running this skript for both full and distributed, the script plot_res.py
can be used to plot the results.

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

from fit import fit_distributed, fit_full


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

# ====== Tilted distribution precision estimate method =========================
# Available options are 'sample' and 'olse', see class serial.Master.
PREC_ESTIM = 'olse'

# ====== 32bit Python ? ========================================================
# Temp fix for the RandomState seed problem with pystan in 32bit Python. Set
# the following to True if using 32bit Python.
TMP_FIX_32BIT = False

# ------------------------------------------------------------------------------
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


def main(mtype='both'):
    
    # Check mtype
    if mtype != 'both' and mtype != 'full' and mtype != 'distributed':
        raise ValueError("Invalid argument `mtype`")
    
    model_name = 'm1'
    
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
    #     Fit model(s)
    # ------------------------------------------------------
    
    if mtype == 'both' or mtype == 'distributed':
        
        # Options for the ep-algorithm see documentation of dep.serial.Master
        options = {
            'seed'       : SEED_MCMC,
            'init_prev'  : True,
            'prec_estim' : PREC_ESTIM,
            'chains'     : CHAINS,
            'iter'       : ITER,
            'warmup'     : WARMUP,
            'thin'       : THIN,
            'prior'      : prior
        }
        # Temp fix for the RandomState seed problem with pystan in 32bit Python
        options['tmp_fix_32bit'] = TMP_FIX_32BIT
        
        fit_distributed(model_name, EP_ITER, J, K, Nj, X, y, phi_true, options)
    
    if mtype == 'both' or mtype == 'full':
        
        seed = np.random.RandomState(seed=SEED_MCMC)
        
        # Temp fix for the RandomState seed problem with pystan in 32bit Python
        seed = seed.randint(2**31-1) if TMP_FIX_32BIT else seed
        
        fit_full(model_name, J, j_ind, X, y, phi_true, m0, Q0, seed)
    

if __name__ == '__main__':
    if len(os.sys.argv) == 2:
        main(os.sys.argv[1])
    else:
        main()



