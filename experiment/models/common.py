"""Common features for the models."""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import numpy as np


class data(object):
    """Data simulated from the hierarchical models.
    
    Attributes
    ----------
    X : ndarray
        Explanatory variable
    
    y : ndarray
        Response variable data
    
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
    
    def __init__(self, X, y, y_true, Nj, j_lim, j_ind, true_values):
        self.X = X
        self.y = y
        self.y_true = y_true
        self.Nj = Nj
        self.N = np.sum(Nj)
        self.J = Nj.shape[0]
        self.j_lim = j_lim
        self.j_ind = j_ind
        self.true_values = true_values
    
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
            # Continuous: r squared
            sst = np.sum(np.square(self.y-np.mean(self.y)))
            sse = np.sum(np.square(self.y-self.true_values))
            uncertainty_global = 1 - sse/sst
            uncertainty_group = np.empty(self.J)
            for j in xrange(self.J):
                sst = np.sum(np.square(
                    self.y[j_lim[j]:j_lim[j+1]]
                    -np.mean(self.y[j_lim[j]:j_lim[j+1]])
                ))
                sse = np.sum(np.square(
                    self.y[j_lim[j]:j_lim[j+1]]
                    -self.true_values[j_lim[j]:j_lim[j+1]]
                ))
                uncertainty_group[j] = 1 - sse/sst
        return uncertainty_global, uncertainty_group


