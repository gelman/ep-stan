"""Common features for the models."""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import numpy as np
from scipy.special import erfinv, logit


# ====== Linear regression uncertainity parameters =============================
# The target coefficient of determination
R_SQUARED = 0.5

# ====== Classification uncertainity parameters ================================
# ------ Provided values -------------------------
# Classification percentage threshold
P_0 = 0.1
# Tail probability threshold
GAMMA_0 = 0.025
# Min linear predictor term standard deviation
SIGMA_F0 = 0.5
# ------ Precalculated values --------------------
ERFINVGAMMA0 = erfinv(2*GAMMA_0-1)
LOGITP0 = logit(P_0)
# Max deviation: |alpha + beta'*x| <= DELTA_MAX
DELTA_MAX = np.sqrt(2)*SIGMA_F0*ERFINVGAMMA0 - LOGITP0


def calc_input_param_lin_reg(beta, sigma):
    """Calculate suitable sigma_x for linear regression models.
    
    Parameters
    ----------
    beta : float or ndarray
        The explanatory variable coefficient of size (J,D), (D,) or (), where J 
        is the number of groups and D is the number of input dimensions.
    
    sigma : float
        The noise standard deviation.
    
    Returns
    -------
    sigma_x : float or ndarray
        If beta is two dimensional, sigma_x is calculated for each group. 
        Otherwise a single value is returned.
    
    """
    beta = np.asarray(beta)
    if beta.ndim == 0 or beta.shape[-1] == 1:
        # One dimensional input
        if beta.ndim == 2:
            beta = beta[:,0]
        elif beta.ndim == 1:
            beta = beta[0]
        out = np.asarray(np.abs(beta))
        np.divide(np.sqrt(R_SQUARED/(1-R_SQUARED))*sigma, out, out=out)
    else:
        # Multidimensional input
        out = np.asarray(np.sum(np.square(beta), axis=-1))
        out *= 1 - R_SQUARED
        np.divide(R_SQUARED, out, out=out)
        np.sqrt(out, out=out)
        out *= sigma
    return out[()]


def calc_input_param_classification(alpha, beta):
    """Calculate suitable mu_x and sigma_x for classification models.
    
    Parameters
    ----------
    alpha : float or ndarray
        The intercept coefficient of size (J) or (), where J is the number of 
        groups.
    
    beta : float or ndarray
        The explanatory variable coefficient of size (J,D), (D,) or (), where J 
        is the number of groups and D is the number of input dimensions.
    
    Returns
    -------
    mu_x, sigma_x : float or ndarray
        If beta is two dimensional and/or alpha is one dimensional, params are 
        calculated for each group. Otherwise single values are returned.
    
    """
    # Check arguments
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if alpha.ndim > 1 or beta.ndim > 2:
        raise ValueError("Dimension of input arguments is too big")
    if alpha.ndim == 1 and alpha.shape[0] != 1:
        J = alpha.shape[0]
    elif beta.ndim == 2 and beta.shape[0] != 1:
        J = beta.shape[0]
    else:
        J = 1
        if alpha.ndim == 0 and beta.ndim < 2:
            scalar_output = True
        else:
            scalar_output = False
    
    # Process
    if J == 1:
        # Single group
        alpha = np.squeeze(alpha)[()]
        beta = np.squeeze(beta)
        if np.abs(alpha) < DELTA_MAX:
            # No mean adjustment needed
            mu_x = 0
            if beta.ndim == 1:
                ssbeta = np.sqrt(2*np.sum(np.square(beta)))
            else:
                # Only one input dimension
                ssbeta = np.sqrt(2)*beta
            sigma_x = (LOGITP0 + np.abs(alpha))/(ERFINVGAMMA0*ssbeta)
        else:
            # Mean adjustment needed
            if alpha > 0:
                mu_x = (DELTA_MAX -alpha)/np.sum(beta)
            else:
                mu_x = (-DELTA_MAX -alpha)/np.sum(beta)
            if beta.ndim == 1:
                ssbeta = np.sqrt(np.sum(np.square(beta)))
            else:
                # Only one input dimension
                ssbeta = beta
            sigma_x = SIGMA_F0/ssbeta
        if not scalar_output:
            mu_x = np.asarray([mu_x])
            sigma_x = np.asarray([sigma_x])
    
    else:
        # Multiple groups
        alpha = np.squeeze(alpha)
        if beta.ndim == 2 and beta.shape[0] == 1:
            beta = beta[0]
        if beta.ndim == 0:
            beta = beta[np.newaxis]
        
        if alpha.ndim == 0:
            # Common alpha: beta.ndim == 2
            if np.abs(alpha) < DELTA_MAX:
                # No mean adjustment needed
                mu_x = np.zeros(J)
                if beta.shape[1] != 1:
                    sigma_x = np.sum(np.square(beta), axis=-1)
                    sigma_x *= 2
                    np.sqrt(sigma_x, out=sigma_x)
                else:
                    # Only one input dimension
                    sigma_x = np.sqrt(2)*beta[:,0]
                sigma_x *= ERFINVGAMMA0
                np.divide(LOGITP0 + np.abs(alpha), sigma_x, out=sigma_x)
            else:
                # Mean adjustment needed
                mu_x = np.sum(beta, axis=-1)
                if alpha > 0:
                    np.divide(DELTA_MAX -alpha, mu_x, out=mu_x)
                else:
                    np.divide(-DELTA_MAX -alpha, mu_x, out=mu_x)
                if beta.shape[1] != 1:
                    sigma_x = np.sum(np.square(beta), axis=-1)
                    np.sqrt(sigma_x, out=sigma_x)
                else:
                    # Only one input dimension
                    sigma_x = beta[:,0].copy()
                np.divide(SIGMA_F0, sigma_x, out=sigma_x)
        
        elif beta.ndim == 1:
            # Common beta: alpha.ndim == 1
            sbeta = np.sum(beta)
            if beta.shape[0] != 1:
                ssbeta = np.sqrt(np.sum(np.square(beta)))
            else:
                ssbeta = beta
            divisor = np.sqrt(2) * ERFINVGAMMA0 * ssbeta
            mu_x = np.zeros(J)
            sigma_x = np.empty(J)
            for j in xrange(J):
                if np.abs(alpha[j]) < DELTA_MAX:
                    # No mean adjustment needed
                    sigma_x[j] = (LOGITP0 + np.abs(alpha[j])) / divisor
                else:
                    # Mean adjustment needed
                    if alpha[j] > 0:
                        mu_x[j] = (DELTA_MAX -alpha[j])/sbeta
                    else:
                        mu_x[j] = (-DELTA_MAX -alpha[j])/sbeta
                    sigma_x[j] = SIGMA_F0/ssbeta
        
        else:
            # Multiple alpha and beta: alpha.ndim == 1 and beta.ndim == 2
            sbeta = np.sum(beta, axis=-1)
            if beta.shape[1] != 1:
                ssbeta = np.sqrt(np.sum(np.square(beta), axis=-1))
            else:
                ssbeta = beta[:,0]
            divisor = np.sqrt(2) * ERFINVGAMMA0 * ssbeta
            mu_x = np.zeros(J)
            sigma_x = np.empty(J)
            for j in xrange(J):
                if np.abs(alpha[j]) < DELTA_MAX:
                    # No mean adjustment needed
                    sigma_x[j] = (LOGITP0 + np.abs(alpha[j])) / divisor[j]
                else:
                    # Mean adjustment needed
                    if alpha[j] > 0:
                        mu_x[j] = (DELTA_MAX -alpha[j])/sbeta[j]
                    else:
                        mu_x[j] = (-DELTA_MAX -alpha[j])/sbeta[j]
                    sigma_x[j] = SIGMA_F0/ssbeta[j]
    
    return mu_x, sigma_x


class data(object):
    """Data simulated from the hierarchical models.
    
    Attributes
    ----------
    X : ndarray
        Explanatory variable
    
    y : ndarray
        Response variable data
    
    X_param : dict
        Parameters of the distribution of X.
    
    y_true : ndarray
        The true expected values of the response variable at X
    
    Nj : ndarray
        Number of observations in each group
    
    N : int
        Total number of observations
    
    J : int
        Number of hierarchical groups
    
    j_lim : ndarray
        Index limits of the partitions of the observations:
        y[j_lim[j]:j_lim[j+1]] belong to group j.
    
    j_ind : ndarray
        The group index of each observation
    
    true_values : dict
        True values of `phi` and other inferred variables
    
    """
    
    def __init__(self, X, y, X_param, y_true, Nj, j_lim, j_ind, true_values):
        self.X = X
        self.y = y
        self.y_true = y_true
        self.Nj = Nj
        self.N = np.sum(Nj)
        self.J = Nj.shape[0]
        self.j_lim = j_lim
        self.j_ind = j_ind
        self.true_values = true_values
        self.X_param = X_param
    
    def calc_uncertainty(self):
        """Calculate the uncertainty in the response variable.
        
        Returns: uncertainty_global, uncertainty_group
        
        """
        y = self.y
        y_true = self.y_true
        j_lim = self.j_lim
        Nj = self.Nj
        if issubclass(y.dtype.type, np.integer):
            # Categorial: percentage of wrong classes
            uncertainty_global = np.count_nonzero(y_true != y)/self.N
            uncertainty_group = np.empty(self.J)
            for j in xrange(self.J):
                uncertainty_group[j] = (
                    np.count_nonzero(
                        y_true[j_lim[j]:j_lim[j+1]] != y[j_lim[j]:j_lim[j+1]]
                    ) / Nj[j]
                )
        else:
            # Continuous: R squared
            sst = np.sum(np.square(y - np.mean(y)))
            sse = np.sum(np.square(y - y_true))
            uncertainty_global = 1 - sse/sst
            uncertainty_group = np.empty(self.J)
            for j in xrange(self.J):
                sst = np.sum(np.square(
                    y[j_lim[j]:j_lim[j+1]] - np.mean(y[j_lim[j]:j_lim[j+1]])
                ))
                sse = np.sum(np.square(
                    y[j_lim[j]:j_lim[j+1]] - y_true[j_lim[j]:j_lim[j+1]]
                ))
                uncertainty_group[j] = 1 - sse/sst
        return uncertainty_global, uncertainty_group


