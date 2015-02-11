"""A simple hierarchical logistic regression experiment for distributed EP
algorithm described in an article "Expectation propagation as a way of life"
(arXiv:1412.4869).

Execute with:
    $ python fit.py model_name [type [save_true]]
where:
    model_name - name of the model
    type       - which models are fit, available values are:
                 `both`, `full`, `distributed` and `none` (default `both`)
    save_true  - save the true values, available values are:
                 `true`, `false`, `1` and `0` (default `true`)

The results of full model are saved into file `res_f_<model_name>.npz`,
the results of distributed model are saved into file `res_d_<model_name>.npz`
and the true values are saved into the file `true_vals_<model_name>.npz`
into the folder results. Available models are in the folder models.

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
J   = 50            # Number of hierarchical groups
D   = 50            # Number of inputs
K   = 8            # Number of sites
NPG = [40,60]       # Number of observations per group (constant or [min, max])

# ====== Sampling parameters for the distributed model =========================
CHAINS = 4
ITER   = 500
WARMUP = 200
THIN   = 2

# ====== Sampling parameters for the full model ================================
CHAINS_FULL = 4
ITER_FULL   = 1000
WARMUP_FULL = 500
THIN_FULL   = 2

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


def main(model_name, mtype, save_true):
    
    # Import the model simulator module (import at runtime)
    model = getattr(__import__('models.'+model_name), model_name)
    
    # Simulate_data
    X, y, Nj, j_ind, true_vals = model.simulate_data(J, D, NPG, seed=SEED_DATA)
    
    # Save true values
    if save_true:
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savez('results/true_vals_{}.npz'.format(model_name),
                 seed_data = SEED_DATA, **true_vals)
        print "True values saved into results"
    
    # Get the prior
    S0, m0, Q0, r0 = model.get_prior(J, D)
    prior = {'Q':Q0, 'r':r0}
    
    # Get parameter information
    pnames, pshapes, phiers = model.get_param_definitions(J, D)
    
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
        
        stan_model = None
        
        if K < 2:
            raise ValueError("K should be at least 2.")
        
        elif K < J:
            # ------ Many groups per site: combine groups ------
            Nk, Nj_k, j_ind_k = distribute_groups(J, K, Nj)
            # Create the Master instance
            stan_model = load_stan('models/'+model_name)
            dep_master = Master(
                stan_model,
                X,
                y,
                A_k={'J':Nj_k},
                A_n={'j_ind':j_ind_k+1},
                site_sizes=Nk,
                **options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, Nj_k)
        
        elif K == J:
            # ------ One group per site ------
            # Create the Master instance
            stan_model_sg = load_stan('models/'+model_name+'_sg')
            dep_master = Master(
                stan_model_sg,
                X,
                y,
                site_sizes=Nj,
                **options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, None)
        
        elif K <= N:
            # ------ Multiple sites per group: split groups ------
            Nk, Nk_j, _ = distribute_groups(J, K, Nj)
            # Create the Master instance
            stan_model_sg = load_stan('models/'+model_name+'_sg')
            dep_master = Master(
                stan_model_sg,
                X,
                y,
                site_sizes=Nk,
                **options
            )
            # Construct the map: which site contribute to which parameter
            pmaps = _create_pmaps(phiers, Nk_j)
        
        else:
            raise ValueError("K cant be greater than number of samples")
        
        # Run the algorithm for `EP_ITER` iterations
        print "Run distributed EP algorithm for {} iterations.".format(EP_ITER)
        m_phi, var_phi = dep_master.run(EP_ITER)
        print "Form the final approximation " \
              "by mixing the samples from all the sites."
        S_phi_mix, m_phi_mix = dep_master.mix_phi()
        var_phi_mix = np.diag(S_phi_mix)
        
        # Get mean and var of inferred variables
        pms, pvars = dep_master.mix_pred(pnames, pmaps, pshapes)
        # Construct a dict of from these results
        presults = {}
        for i in xrange(len(pnames)):
            pname = pnames[i]
            presults['m_'+pname] = pms[i]
            presults['var_'+pname] = pvars[i]
        
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
            **presults
        )
        print "Distributed model results saved."
    
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
        # Load model if not loaded already
        if stan_model is None:
            stan_model = load_stan('models/'+model_name)
        
        # Sample and extract parameters
        with suppress_stdout():
            fit = stan_model.sampling(
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
        
        # Get mean and var of inferred variables
        presults = {}
        for i in xrange(len(pnames)):
            pname = pnames[i]
            samp = fit.extract(pname)[pname]
            presults['m_'+pname+'_full'] = np.mean(samp, axis=0)
            presults['var_'+pname+'_full'] = np.var(samp, axis=0, ddof=1)
        
        # Save results
        if not os.path.exists('results'):
            os.makedirs('results')
        np.savez('results/res_f_{}.npz'.format(model_name),
            seed_data      = SEED_DATA,
            seed_mcmc      = SEED_MCMC,
            m_phi_full     = m_phi_full,
            var_phi_full   = var_phi_full,
            **presults
        )
        print "Full model results saved."
    

def _create_pmaps(phiers, Ns):
    """Create the mappings for hierarhical parameters."""
    if K < 2:
        raise ValueError("K should be at least 2.")
    
    elif K < J:
        # ------ Many groups per site: combined groups ------
        pmaps = []
        for pi in xrange(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            else:
                pmap = []
                i = 0
                for k in xrange(K):
                    # Create indexings until the ih dimension, remaining
                    # dimension's slice(None) can be left out
                    if ih == 0:
                        pmap.append(slice(i, i+Ns[k]))
                    else:
                        pmap.append(
                            tuple(
                                slice(i, i+Ns[k])
                                if i2 == ih else slice(None)
                                for i2 in xrange(ih+1)
                            )
                        )
                    i += Ns[k]
                pmaps.append(pmap)
    
    elif K == J:
        # ------ One group per site ------
        pmaps = []
        for pi in xrange(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            elif ih == 0:
                # First dimensions can be mapped with one ndarray
                pmaps.append(np.arange(K))
            else:
                pmap = []
                for k in xrange(K):
                    # Create indexings until the ih dimension, remaining
                    # dimension's slice(None) can be left out
                    pmap.append(
                        tuple(
                            k if i2 == ih else slice(None)
                            for i2 in xrange(ih+1)
                        )
                    )
                pmaps.append(pmap)
    
    else:
        # ------ Multiple sites per group: split groups ------
        pmaps = []
        for pi in xrange(len(phiers)):
            ih = phiers[pi]
            if ih is None:
                pmaps.append(None)
            elif ih == 0:
                # First dimensions can be mapped with one ndarray
                pmap = np.empty(K, dtype=np.int32)
                i = 0
                for j in xrange(J):
                    for _ in xrange(Ns[j]):
                        pmap[i] = j
                        i += 1
                pmaps.append(pmap)
            else:
                pmap = []
                i = 0
                for j in xrange(J):
                    for _ in xrange(Ns[j]):
                        # Create indexings until the ih dimension, remaining
                        # dimension's slice(None) can be left out
                        pmap.append(
                            tuple(
                                j if i2 == ih else slice(None)
                                for i2 in xrange(ih+1)
                            )
                        )
                        i += 1
                pmaps.append(pmap)
    
    return pmaps


if __name__ == '__main__':
    
    # Parse arguments
    if len(os.sys.argv) > 4:
        raise TypeError("Wrong number of arguments")
    
    if len(os.sys.argv) > 3:
        save_true = os.sys.argv[3].lower()
        if (     save_true != 'true'
             and save_true != 'false'
             and save_true != '1'
             and save_true != '0'
           ):
            raise ValueError("Invalid argument `mtype`")
        save_true = save_true == 'true' or save_true == '1'
    else:
        save_true = True
    
    if len(os.sys.argv) > 2:
        mtype = os.sys.argv[2].lower()
        if (     mtype != 'both'
             and mtype != 'full'
             and mtype != 'distributed'
             and mtype != 'none'
           ):
            raise ValueError("Invalid argument `mtype`")
    else:
        mtype = 'both'
    
    if len(os.sys.argv) > 1:
        model_name = os.sys.argv[1]
    else:
        raise ValueError("Give the model name (e.g. `m1`) as argument")
    
    # Run
    main(model_name, mtype, save_true)



