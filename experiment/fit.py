"""Methods for fitting some expereiment models. Used by skripts
fit_<model_name>.py.

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


def fit_distributed(model_name, niter, J, K, Nj, X, y, phi_true, options):
    """Fit distributed model and save the results."""
    
    print "Distributed model {} ...".format(model_name)
    
    N = Nj.sum()
    
    if K < 2:
        raise ValueError("K should be at least 2.")
    elif K < J:
        # ---- Many groups per site ----
        # Combine smallest pairs of consecutive groups until K has been reached
        j_lim = np.concatenate(([0], np.cumsum(Nj)))
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
        model = load_stan(model_name)
        dep_master = Master(
            model,
            X,
            y,
            A_k={'J':Nj_k},
            A_n={'j_ind':j_ind_k+1},
            site_sizes=Nk,
            **options
        )
    elif K == J:
        # ---- One group per site ----
        # Create the Master instance
        model_single_group = load_stan(model_name+'_sg')
        dep_master = Master(
            model_single_group,
            X,
            y,
            site_sizes=Nj,
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
        model_single_group = load_stan(model_name+'_sg')
        dep_master = Master(
            model_single_group,
            X,
            y,
            site_sizes=Nk,
            **options
        )
    else:
        raise ValueError("K cant be greater than number of samples")
    
    # Run the algorithm for `niter` iterations
    print "Run distributed EP algorithm for {} iterations.".format(niter)
    m_phi, var_phi = dep_master.run(niter)
    print "Form the final approximation " \
          "by mixing the samples from all the sites."
    S_mix, m_mix = dep_master.mix_samples()
    var_mix = np.diag(S_mix)
    
    print "Distributed model sampled."
    
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savez('results/res_d_{}.npz'.format(model_name),
        phi_true=phi_true,
        m_phi=m_phi,
        var_phi=var_phi,
        m_mix=m_mix,
        var_mix=var_mix,
    )


def fit_full(model_name, J, j_ind, X, y, phi_true, m0, Q0, seed):
    """Fit full model and save the results."""
    
    print "Full model {} ...".format(model_name)
    
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
    model = load_stan(model_name)
    
    # Sample and extract parameters
    with suppress_stdout():
        fit = model.sampling(
            data=data,
            seed=seed,
            chains=4,
            iter=1000,
            warmup=500,
            thin=2
        )
    samp = fit.extract(pars='phi')['phi']
    m_phi_full = samp.mean(axis=0)
    var_phi_full = samp.var(axis=0, ddof=1)
    
    print "Full model sampled."
    
    
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savez('results/res_f_{}.npz'.format(model_name),
        phi_true=phi_true,
        m_phi_full=m_phi_full,
        var_phi_full=var_phi_full,
    )



