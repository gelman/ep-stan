"""A simulated experiment model used by the sckript fit.py

Model name: m4
Definition:
    group index j = 1 ... J
    input index d = 1 ... D
    explanatory variable x = [x_1 ... x_D]
    response variable y
    local parameter alpha = [alpha_1 ... alpha_J]
    local parameter beta = [[beta_11 ... beta_1D] ... [beta_J1 ... beta_JD]]
    shared parameter sigma_a
    shared parameter sigma_b = [sigma_b_1 ... sigma_b_D]
    shared parameter mu_a
    shared parameter mu_b = [mu_b_1 ... mu_b_D]
    y ~ bernoulli_logit(alpha_j + beta_j*' * x)
    alpha ~ N(mu_a, sigma_a)
    beta_*d ~ N(mu_b_d, sigma_b_d), for all d
    mu_a ~ N(0, sigma_maH)
    mu_b_d ~ N(0, sigma_mbH), for all d
    sigma_a ~ log-N(0, sigma_aH)
    sigma_b_d ~ log-N(0, sigma_bH), for all d
    phi = [mu_a, log(sigma_a), mu_b, log(sigma_b)]

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import numpy as np


# ------------------------------------------------------------------------------
# >>>>>>>>>>>>> Configurations start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------------------------------------------------------------

# ====== Model parameters ======================================================
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
V0_MA = 1.5**2
# Prior for log(sigma_a)
M0_SA = 0
V0_SA = 1.5**2
# Prior for mu_b
M0_MB = 0
V0_MB = 1.5**2
# Prior for log(sigma_b)
M0_SB = 0
V0_SB = 1.5**2

# ------------------------------------------------------------------------------
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


# Inferred parameters
PARAMS = ('alpha', 'beta')

def simulate_data(J, D, NPG, seed=None):
    """Simulate data from the model.
    
    Parameters
    ----------
    J : int
        Number of groups
    
    D : int
        Number of inputs
    
    NPG : {int, seq of ints}
        Number of observations per group (constant or [min, max])
    
    seed : int
        Seed for the presudo random generator
    
    Returns
    -------
    X : ndarray
        Explanatory variable
    
    y : ndarray
        Response variable data
    
    Nj : ndarray
        Number of observations in each group
    
    j_ind : ndarray
        The group index of each observation
    
    true_values : dict
        True values of `phi` and other inferred variables
    
    """
    
    # Set seed
    rnd_data = np.random.RandomState(seed=seed)
    
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
    
    return X, y, Nj, j_ind, {'phi':phi_true, 'alpha':alpha_j, 'beta':beta_j}


def get_prior(J, D):
    """Get prior for the model.
    
    Returns: S, m, Q, r
    """
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
    return S0, m0, Q0, r0


def get_param_definitions(J, D):
    """Return the definition of the inferred parameters.
    
    Returns
    -------
    names : seq of str
        Names of the parameters
    
    shapes : seq of tuples
        Shapes of the parameters
    
    hiers : seq of int 
        The indexes of the hierarchical dimension of the parameter or None if it
        does not have one.
    
    """    
    names = ('alpha', 'beta')
    shapes = ((J,), (J,D))
    hiers = (0, 0)
    return names, shapes, hiers


