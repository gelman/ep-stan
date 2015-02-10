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
both models are fit. The results are saved into files `res_f_<model_name>.npz`
and `res_d_<model_name>.npz` into the folder results respectively. The true
values are saved into the file `true_vals_<model_name>.npz`.

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
from dep.util import load_stan, distribute_groups, suppress_stdout


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
ITER = 500
WARMUP = 200
THIN = 2

# ====== Sampling parameters for the full model ================================
CHAINS_FULL = 4
ITER_FULL = 1000
WARMUP_FULL = 500
THIN_FULL = 2

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
    
    # Save true values
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savez('results/true_vals_{}.npz'.format(model_name),
        seed_data = SEED_DATA,
        phi       = phi_true,
        beta      = beta,
        alpha     = alpha_j
    )
    
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
    #     Fit distributed model
    # ------------------------------------------------------
    
    if mtype == 'both' or mtype == 'distributed':
        
        print "Distributed model {} ...".format(model_name)
        
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
        
        if K < 2:
            raise ValueError("K should be at least 2.")
        
        elif K < J:
            # ------ Many groups per site: combine groups ------
            Nk, Nj_k, j_ind_k = distribute_groups(J, K, Nj)
            # Create the Master instance
            model = load_stan('stan_files/'+model_name)
            dep_master = Master(
                model,
                X,
                y,
                A_k={'J':Nj_k},
                A_n={'j_ind':j_ind_k+1},
                site_sizes=Nk,
                **options
            )
            # Construct the map: which site contribute to which parameter
            shape_alpha = (J,)
            smap_alpha = []
            i = 0
            for k in xrange(K):
                smap_alpha.append(np.arange(i,i+Nj_k[k]))
                i += Nj_k[k]
            shape_beta = (D,)
            smap_beta = None
        
        elif K == J:
            # ------ One group per site ------
            # Create the Master instance
            model_single_group = load_stan('stan_files/'+model_name+'_sg')
            dep_master = Master(
                model_single_group,
                X,
                y,
                site_sizes=Nj,
                **options
            )
            # Construct the map: which site contribute to which parameter
            shape_alpha = (J,)
            smap_alpha = np.arange(K)
            shape_beta = (D,)
            smap_beta = None
        
        elif K <= N:
            # ------ Multiple sites per group: split groups ------
            Nk, Nk_j, _ = distribute_groups(J, K, Nj)
            # Create the Master instance
            model_single_group = load_stan('stan_files/'+model_name+'_sg')
            dep_master = Master(
                model_single_group,
                X,
                y,
                site_sizes=Nk,
                **options
            )
            # Construct the map: which site contribute to which parameter
            shape_alpha = (J,)
            smap_alpha = np.empty(K, dtype=np.int32)
            i = 0
            for j in xrange(J):
                for _ in xrange(Nk_j[j]):
                    smap_alpha[i] = j
                    i += 1
            shape_beta = (D,)
            smap_beta = None
            
            
        
        else:
            raise ValueError("K cant be greater than number of samples")
        
        # Run the algorithm for `EP_ITER` iterations
        print "Run distributed EP algorithm for {} iterations.".format(EP_ITER)
        m_phi, var_phi = dep_master.run(EP_ITER)
        print "Form the final approximation " \
              "by mixing the samples from all the sites."
        S_phi_mix, m_phi_mix = dep_master.mix_phi()
        var_phi_mix = np.diag(S_phi_mix)
        
        # Get mean and var of alpha and beta
        m_alpha, var_alpha = dep_master.mix_pred(
                'alpha', smap_alpha, shape_alpha)
        m_beta, var_beta = dep_master.mix_pred('beta', smap_beta, shape_beta)
        print "Distributed model sampled."
        
        # Save results
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savez('results/res_d_{}.npz'.format(model_name),
            seed_data   = SEED_DATA,
            seed_mcmc   = SEED_MCMC,
            m_phi       = m_phi,
            var_phi     = var_phi,
            m_phi_mix   = m_phi_mix,
            var_phi_mix = var_phi_mix,
            m_alpha     = m_alpha,
            var_alpha   = var_alpha,
            m_beta      = m_beta,
            var_beta    = var_beta
        )
        
    
    # ------------------------------------------------------
    #     Fit full model
    # ------------------------------------------------------
    
    if mtype == 'both' or mtype == 'full':
        
        print "Full model {} ...".format(model_name)
        
        seed = np.random.RandomState(seed=SEED_MCMC)
        # Temp fix for the RandomState seed problem with pystan in 32bit Python
        seed = seed.randint(2**31-1) if TMP_FIX_32BIT else seed
        
        data = dict(
            N=X.shape[0],
            D=X.shape[1],
            J=J,
            X=X,
            y=y,
            j_ind=j_ind+1,
            mu_phi=m0,
            Omega_phi=Q0.T    # Q0 transposed in order to get C-contiguous
        )
        model = load_stan('stan_files/'+model_name)
        
        # Sample and extract parameters
        with suppress_stdout():
            fit = model.sampling(
                data=data,
                seed=seed,
                chains=CHAINS_FULL,
                iter=ITER_FULL,
                warmup=WARMUP_FULL,
                thin=THIN_FULL
            )
        samp = fit.extract(pars='phi')['phi']
        m_phi_full = samp.mean(axis=0)
        var_phi_full = samp.var(axis=0, ddof=1)
        
        # Get mean and var of alpha and beta
        samp = fit.extract('alpha')['alpha']
        m_alpha_full = np.mean(samp, axis=0)
        var_alpha_full = np.var(samp, axis=0, ddof=1)
        samp = fit.extract('beta')['beta']
        m_beta_full = np.mean(samp, axis=0)
        var_beta_full = np.var(samp, axis=0, ddof=1)
        
        print "Full model sampled."
        
        # Save results
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savez('results/res_f_{}.npz'.format(model_name),
            seed_data      = SEED_DATA,
            seed_mcmc      = SEED_MCMC,
            m_phi_full     = m_phi_full,
            var_phi_full   = var_phi_full,
            m_alpha_full   = m_alpha_full,
            var_alpha_full = var_alpha_full,
            m_beta_full    = m_beta_full,
            var_beta_full  = var_beta_full
        )
    

if __name__ == '__main__':
    if len(os.sys.argv) == 2:
        main(os.sys.argv[1])
    else:
        main()



