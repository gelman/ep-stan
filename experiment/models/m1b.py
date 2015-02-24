"""A simulated experiment model used by the sckript fit.py

Model name: m1b
Definition:
    group index j = 1 ... J
    input index d = 1 ... D
    explanatory variable x = [x_1 ... x_D]
    response variable y
    local parameter alpha = [alpha_1 ... alpha_J]
    shared parameter beta = [beta_1 ... beta_D]
    shared parameter sigma_a
    y ~ bernoulli_logit(alpha_j + beta' * x)
    alpha ~ N(0, sigma_a)
    beta_d ~ N(0, sigma_b), for all d
    sigma_a ~ log-N(0, sigma_aH)
    phi = [log(sigma_a), beta]

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import numpy as np
from common import data, calc_input_param_classification


# ------------------------------------------------------------------------------
# >>>>>>>>>>>>> Configurations start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------------------------------------------------------------

# ====== Model parameters ======================================================
# If SIGMA_A is None, it is sampled from log-N(0,SIGMA_AH)
SIGMA_A = 1
SIGMA_AH = None
# If BETA is None, it is sampled from N(0,SIGMA_B)
BETA = None
SIGMA_B = 1

# ====== Prior =================================================================
# Prior for log(sigma_a)
M0_A = 0
V0_A = 1.5**2
# Prior for beta
M0_B = 0
V0_B = 1.5**2

# ------------------------------------------------------------------------------
# <<<<<<<<<<<<< Configurations end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ------------------------------------------------------------------------------


class model(object):
    """Model definition.
    
    Parameters
    ----------
    J : int
        Number of groups
    
    D : int
        Number of inputs
    
    npg : {int, seq of ints}
        Number of observations per group (constant or [min, max])
    
    """
    
    def __init__(self, J, D, npg):
        self.J = J
        self.D = D
        self.npg = npg
        self.dphi = D+1
    
    def simulate_data(self, seed=None):
        """Simulate data from the model.
        
        Returns models.common.data instance
        
        """
        # Localise params
        J = self.J
        D = self.D
        npg = self.npg
        
        # Set seed
        rnd_data = np.random.RandomState(seed=seed)
        
        # Parameters
        # Number of observations for each group
        if hasattr(npg, '__getitem__') and len(npg) == 2:
            Nj = rnd_data.randint(npg[0],npg[1]+1, size=J)
        else:
            Nj = npg*np.ones(J, dtype=np.int64)
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
        
        # Determine suitable mu_x and sigma_x
        mu_x_j, sigma_x_j = calc_input_param_classification(alpha_j, beta)
        
        # Simulate data
        # Different mu_x and sigma_x for every group
        X = np.empty((N,D))
        for j in xrange(J):
            X[j_lim[j]:j_lim[j+1],:] = \
                mu_x_j[j] + rnd_data.randn(Nj[j],D)*sigma_x_j[j]
        y = alpha_j[j_ind] + X.dot(beta)
        y = 1/(1+np.exp(-y))
        y_true = (0.5 < y).astype(int)
        y = (rnd_data.rand(N) < y).astype(int)
        
        return data(
            X, y, {'mu_x':mu_x_j, 'sigma_x':sigma_x_j}, y_true, Nj, j_lim, 
            j_ind, {'phi':phi_true, 'alpha':alpha_j, 'beta':beta}
        )
    
    def get_prior(self):
        """Get prior for the model.
        
        Returns: S, m, Q, r
        
        """
        D = self.D
        # Moment parameters of the prior (transposed in order to get
        # F-contiguous)
        S0 = np.diag(np.append(V0_A, np.ones(D)*V0_B)).T
        m0 = np.append(M0_A, np.ones(D)*M0_B)
        # Natural parameters of the prior
        Q0 = np.diag(np.append(1./V0_A, np.ones(D)/V0_B)).T
        r0 = np.append(M0_A/V0_A, np.ones(D)*(M0_B/V0_B))
        return S0, m0, Q0, r0
    
    def get_param_definitions(self):
        """Return the definition of the inferred parameters.
        
        Returns
        -------
        names : seq of str
            Names of the parameters
        
        shapes : seq of tuples
            Shapes of the parameters
        
        hiers : seq of int 
            The indexes of the hierarchical dimension of the parameter or None
            if it does not have one.
        
        """
        names = ('alpha', 'beta')
        shapes = ((self.J,), (self.D,))
        hiers = (0, None)
        return names, shapes, hiers


