"""A simulated experiment model used by the sckript fit.py

Model name: m5a
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
    shared parameter sigma
    y ~ N(alpha_j + beta_j*' * x, sigma)
    alpha ~ Laplace(mu_a, sigma_a)
    beta_*d ~ Laplace(mu_b_d, sigma_b_d), for all d
    mu_a ~ Laplace(0, sigma_maH)
    mu_b_d ~ Laplace(0, sigma_mbH), for all d
    sigma ~ log-N(0, sigma_H)
    sigma_a ~ half-Cauchy(0, sigma_aH)
    sigma_b_d ~ half-Cauchy(0, sigma_bH), for all d
    phi = [log(sigma), mu_a, log(sigma_a), mu_b, log(sigma_b)]

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy.linalg import cholesky
from .common import data, calc_input_param_lin_reg, rand_corr_vine


# ------------------------------------------------------------------------------
# >>>>>>>>>>>>> Configurations start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------------------------------------------------------------

# ====== Model parameters ======================================================
# If SIGMA is None, it is sampled from log-N(0,SIGMA_H)
SIGMA = 1
SIGMA_H = None
# If MU_A is None, it is sampled from Laplace(0,SIGMA_MA)
MU_A = 0.1
SIGMA_MA = None
# If SIGMA_A is None, it is sampled from half-Cauchy(0,SIGMA_SA)
SIGMA_A = 1
SIGMA_SA = None
SIGMA_MB = 0
SIGMA_SB = 1

# ====== Prior =================================================================
# Prior for log(sigma)
M0_S = 0
V0_S = 1.5**2
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
        self.dphi = 2*D+3

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
        if SIGMA is None:
            sigma = np.exp(rnd_data.randn()*SIGMA_H)
        else:
            sigma = SIGMA
        if SIGMA_A is None:
            sigma_a = np.abs(rnd_data.standard_cauchy()*SIGMA_SA)
        else:
            sigma_a = SIGMA_A
        if MU_A is None:
            mu_a = rnd_data.laplace()*SIGMA_MA
        else:
            mu_a = MU_A
        sigma_b = np.abs(rnd_data.standard_cauchy(D)*SIGMA_SB)
        mu_b = rnd_data.laplace(size=D)*SIGMA_MB
        alpha_j = mu_a + rnd_data.laplace(size=J)*sigma_a
        beta_j = mu_b + rnd_data.laplace(size=(J,D))*sigma_b
        
        # Regulate beta
        for j in range(J):
            beta_sum = np.sum(beta_j[j])
            while np.abs(beta_sum) < B_ABS_MIN_SUM:
                # Replace one random element in beta
                index = rnd_data.randint(D)
                beta_sum -= beta_j[j,index]
                beta_j[j,index] = mu_b[index] + rnd_data.randn()*sigma_b[index]
                beta_sum += beta_j[j,index]
        
        phi_true = np.empty(self.dphi)
        phi_true[0] = np.log(sigma)
        phi_true[1] = mu_a
        phi_true[2] = np.log(sigma_a)
        phi_true[3:3+D] = mu_b
        phi_true[3+D:] = np.log(sigma_b)
        
        # Determine suitable sigma_x
        sigma_x_j = calc_input_param_lin_reg(beta_j, sigma, Sigma_x)
        
        # Simulate data
        # Different sigma_x for every group
        X = np.empty((N,D))
        if Sigma_x is None:
            for j in range(J):
                X[j_lim[j]:j_lim[j+1],:] = rnd_data.randn(Nj[j],D)*sigma_x_j[j]
        else:
            cho_x = cholesky(Sigma_x)
            for j in range(J):
                X[j_lim[j]:j_lim[j+1],:] = \
                    rnd_data.randn(Nj[j],D).dot(sigma_x_j[j]*cho_x)
        y_true = np.empty(N)
        for n in range(N):
            y_true[n] = alpha_j[j_ind[n]] + X[n].dot(beta_j[j_ind[n]])
        y = y_true + rnd_data.randn(N)*sigma
        
        return data(
            X, y, {'sigma_x':sigma_x_j, 'Sigma_x':Sigma_x}, y_true, Nj, j_lim, 
            j_ind, {'phi':phi_true, 'alpha':alpha_j, 'beta':beta_j, 
            'sigma':sigma}
        )
    
    def get_prior(self):
        """Get prior for the model.
        
        Returns: S, m, Q, r
        
        """
        D = self.D
        # Moment parameters of the prior (transposed in order to get
        # F-contiguous)
        S0 = np.empty(self.dphi)
        S0[0] = V0_S
        S0[1] = V0_MA
        S0[2] = V0_SA
        S0[3:3+D] = V0_MB
        S0[3+D:] = V0_SB
        S0 = np.diag(S0).T
        m0 = np.empty(self.dphi)
        m0[0] = M0_S
        m0[1] = M0_MA
        m0[2] = M0_SA
        m0[3:3+D] = M0_MB
        m0[3+D:] = M0_SB
        # Natural parameters of the prior
        Q0 = np.diag(1/np.diag(S0)).T
        r0 = m0/np.diag(S0)
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
        names = ('alpha', 'beta', 'sigma')
        shapes = ((self.J,), (self.J,self.D), ())
        hiers = (0, 0, None)
        return names, shapes, hiers


