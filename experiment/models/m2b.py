"""A simulated experiment model used by the sckript fit.py

Model name: m2b
Definition:
    group index j = 1 ... J
    input index d = 1 ... D
    explanatory variable x = [x_1 ... x_D]
    response variable y
    local parameter alpha = [alpha_1 ... alpha_J]
    shared parameter beta = [beta_1 ... beta_D]
    shared parameters sigma_a, sigma_b
    y ~ bernoulli_logit(alpha_j + beta' * x)
    alpha ~ N(0, sigma_a)
    beta_d ~ N(0, sigma_b), for all d
    sigma_a ~ log-N(0, sigma_aH)
    sigma_b ~ log-N(0, sigma_bH)
    phi = [log(sigma_a), log(sigma_b)]

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy.linalg import cholesky
from .common import data, calc_input_param_classification, rand_corr_vine


# ------------------------------------------------------------------------------
# >>>>>>>>>>>>> Configurations start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------------------------------------------------------------

# ====== Model parameters ======================================================
# If SIGMA_A is None, it is sampled from log-N(0,SIGMA_AH)
SIGMA_A = 1
SIGMA_AH = None
# If SIGMA_B is None, it is sampled from log-N(0,SIGMA_BH)
SIGMA_B = 1
SIGMA_BH = None

# ====== Prior =================================================================
# Prior for log(sigma_a)
M0_A = 0
V0_A = 1.5**2
# Prior for log(sigma_b)
M0_B = 0
V0_B = 1.5**2

# ====== Regulation ============================================================
# Min for abs(sum(beta))
B_ABS_MIN_SUM = 1e-4

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
        self.dphi = 2
    
    def simulate_data(self, Sigma_x=None, seed=None):
        """Simulate data from the model.
        
        Returns models.common.data instance
        
        Parameters
        ----------
        Sigma_x : {None, 'rand', ndarray}
            The covariance structure of the explanatory variable. This is 
            scaled to regulate the uncertainty. If not provided or None, 
            identity matrix is used. Providing string 'rand' uses method
            common.rand_corr_vine to randomise one.
        
        """
        # Localise params
        J = self.J
        D = self.D
        npg = self.npg
        
        # Set seed
        rnd_data = np.random.RandomState(seed=seed)
        # Draw random seed for input covariance for consistency in randomness
        # even if not needed
        seed_input_cov = rnd_data.randint(2**31-1)
        
        # Randomise input covariance structure if needed
        if Sigma_x == 'rand':
            Sigma_x = rand_corr_vine(D, seed=seed_input_cov)
        
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
        for j in range(J):
            j_ind[j_lim[j]:j_lim[j+1]] = j
        
        # Assign parameters
        if SIGMA_A is None:
            sigma_a = np.exp(rnd_data.randn()*SIGMA_AH)
        else:
            sigma_a = SIGMA_A
        if SIGMA_B is None:
            sigma_b = np.exp(rnd_data.randn()*SIGMA_BH)
        else:
            sigma_b = SIGMA_B
        alpha_j = rnd_data.randn(J)*sigma_a
        beta = rnd_data.randn(D)*sigma_b
        
        # Regulate beta
        beta_sum = np.sum(beta)
        while np.abs(beta_sum) < B_ABS_MIN_SUM:
            # Replace one random element in beta
            index = rnd_data.randint(D)
            beta_sum -= beta[index]
            beta[index] = rnd_data.randn()*sigma_b
            beta_sum += beta[index]
        
        phi_true = np.append(np.log(sigma_a), np.log(sigma_b))
        
        # Determine suitable mu_x and sigma_x
        mu_x_j, sigma_x_j = calc_input_param_classification(
            alpha_j, beta, Sigma_x
        )
        
        # Simulate data
        # Different mu_x and sigma_x for every group
        X = np.empty((N,D))
        if Sigma_x is None:
            for j in range(J):
                X[j_lim[j]:j_lim[j+1],:] = \
                    mu_x_j[j] + rnd_data.randn(Nj[j],D)*sigma_x_j[j]
        else:
            cho_x = cholesky(Sigma_x)
            for j in range(J):
                X[j_lim[j]:j_lim[j+1],:] = \
                    mu_x_j[j] + rnd_data.randn(Nj[j],D).dot(sigma_x_j[j]*cho_x)
        y = alpha_j[j_ind] + X.dot(beta)
        y = 1/(1+np.exp(-y))
        y_true = (0.5 < y).astype(int)
        y = (rnd_data.rand(N) < y).astype(int)
        
        return data(
            X, y, {'mu_x':mu_x_j, 'sigma_x':sigma_x_j, 'Sigma_x':Sigma_x}, 
            y_true, Nj, j_lim, j_ind, {'phi':phi_true, 'alpha':alpha_j, 
            'beta':beta}
        )
    
    def get_prior(self):
        """Get prior for the model.
        
        Returns: S, m, Q, r
        
        """
        D = self.D
        # Moment parameters of the prior (transposed in order to get
        # F-contiguous)
        S0 = np.diag(np.append(V0_A, V0_B)).T
        m0 = np.append(M0_A, M0_B)
        # Natural parameters of the prior
        Q0 = np.diag(np.append(1./V0_A, 1./V0_B)).T
        r0 = np.append(M0_A/V0_A, M0_B/V0_B)
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


