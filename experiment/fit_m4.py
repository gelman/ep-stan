"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Group index j = 1 ... J
Model m4:
    y_j ~ bernoulli_logit(alpha_j + beta_j * x_j)
    alpha_j ~ N(mu_a,sigma_a)
    beta_j ~ N(mu_b,sigma_b)
        Cov([beta_j]_a, [beta_j]_b) = 0, a != b
    mu_a ~ N(0,sigma_ma)
    mu_b ~ N(0,sigma_mb)
    sigma_a ~ log-N(0,sigma_sa)
    sigma_b ~ log-N(0,sigma_sb)
    Fixed sigma_ma, sigma_mb, sigma_sa, sigma_sb
    phi = [mu_a, log(sigma_a), mu_b, log(sigma_b)]

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
J = 10              # Number of hierarchical groups
D = 10              # Number of inputs
K = 10              # Number of sites
NPG = [40,60]       # Number of observations per group (constant or [min, max])

# ====== Set parameters ========================================================
# If MU_A is None, it is sampled from N(0,SIGMA_MA)
MU_A = 0.1
SIGMA_MA = None
# If SIGMA_A is None, it is sampled from log-N(0,SIGMA_SA)
SIGMA_A = 1
SIGMA_SA = None
SIGMA_MB = 0
SIGMA_SB = 1

# ====== Prior =================================================================
# Prior for mu_a
M0_MA = 0
V0_MA = 1**2
# Prior for log(sigma_a)
M0_SA = 0
V0_SA = 1**2
# Prior for mu_b
M0_MB = 0
V0_MB = 1**2
# Prior for log(sigma_b)
M0_SB = 0
V0_SB = 1**2

# ====== Sampling parameters ===================================================
CHAINS = 4
ITER = 500
WARMUP = 200
THIN = 2

# ====== Number of EP iterations ===============================================
EP_ITER = 6

# ====== Tilted distribution precision estimate method =========================
# Available options are 'sample' and 'olse', see class serial.Master.
PREC_ESTIM = 'olse'

# ====== 32bit Python ? ========================================================
# Temp fix for the RandomState seed problem with pystan in 32bit Python. Set
# the following to True if using 32bit Python.
TMP_FIX_32BIT = True

# ------------------------------------------------------------------------------
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


def main(mtype='both'):
    
    # Check mtype
    if mtype != 'both' and mtype != 'full' and mtype != 'distributed':
        raise ValueError("Invalid argument `mtype`")
    
    model_name = 'm4'
    
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
        sigma_a = np.exp(rnd_data.randn()*SIGMA_SA)
    else:
        sigma_a = SIGMA_A
    if MU_A is None:
        mu_a = rnd_data.randn()*SIGMA_MA
    else:
        mu_a = MU_A
    sigma_b = np.exp(rnd_data.randn(D)*SIGMA_SB)
    mu_b = rnd_data.randn(D)*SIGMA_MB
    alpha_j = mu_a + rnd_data.randn(J)*sigma_a
    beta_j = mu_b + rnd_data.randn(J,D)*sigma_b
    dphi = 2*D+2  # Number of shared parameters
    phi_true = np.empty(dphi)
    phi_true[0] = mu_a
    phi_true[1] = np.log(sigma_a)
    phi_true[2:2+D] = mu_b
    phi_true[2+D:] = np.log(sigma_b)
    
    # Simulate data
    X = rnd_data.randn(N,D)
    y = np.empty(N)
    for n in xrange(N):
        y[n] = alpha_j[j_ind[n]] + X[n].dot(beta_j[j_ind[n]])
    y = 1/(1+np.exp(-y))
    y = (rnd_data.rand(N) < y).astype(int)
    
    # ------------------------------------------------------
    #     Prior
    # ------------------------------------------------------
    
    # Moment parameters of the prior (transposed in order to get F-contiguous)
    S0 = np.empty(dphi)
    S0[0] = V0_MA
    S0[1] = V0_SA
    S0[2:2+D] = V0_MB
    S0[2+D:] = V0_SB
    S0 = np.diag(S0).T
    m0 = np.empty(dphi)
    m0[0] = M0_MA
    m0[1] = M0_SA
    m0[2:2+D] = M0_MB
    m0[2+D:] = M0_SB
    # Natural parameters of the prior
    Q0 = np.diag(1/np.diag(S0)).T
    r0 = m0/np.diag(S0)
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



