"""An implementation of a distributed EP algorithm described in an article
"Expectation propagation as a way of life" (arXiv:1412.4869).

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Released under licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import pickle
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from util import copy_triu_to_tril

# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')


def invert_spd(A, out=None):
    """Invert a symmetric positive-definite matrix.
    
    Parameters
    ----------
    A : ndarray
        A real symmetric positive-definite matrix to be inverted.
    
    out : {None, ndarray, 'in_place'}
        Spesifies where the output is calculate into; None (default) indicates
        that a new array is created, providing a string 'in_place' overwrites
        the input matrix `A`.
    
    Returns
    -------
    out : ndarray
        The output array (in F-order).
    
    Raises
    ------
    LinAlgError
        If the array is not positive definite.
    
    """
    # Process parameters
    if out == 'in_place':
        out = A
    elif not out:
        out = A.copy(order='F')
    else:
        np.copyto(out, A)
    if not out.flags.farray:
        # Convert from C-order to F-order by transposing (note symmetric)
        out = out.T
        if not out.flags.farray:
            raise ValueError('Provided array is inappropriate')
    # Invert
    linalg.cho_factor(out, overwrite_a=True) # TODO: Combine linalg.solve here
    _, info = dpotri_routine(out, overwrite_c=True)
    if info:
        # This should never occour because cho_factor was succesful ... I think
        raise linalg.LinAlgError(
                "dpotri LAPACK routine failed with error code {}".format(info))
    # Copy the upper triangular into the bottom
    copy_triu_to_tril(out)
    return out
    

class DistributedEP(object):
    """Manages the distributed EP algorithm.
    
    Parameters
    ----------
    site_model : StanModel or string
        Model for sampling from the tilted distribution of a site. Can be
        provided either directly as a PyStan model instance or as filename
        string pointing to a pickled model. The model has a restricted
        structure; see Notes.
    
    X : ndarray
        Explanatory variable data in an ndarray of shape (N,K), where N is the
        number of observations and K is the number of variables. X should be
        C contiguous (copy made if not).
    
    y : ndarray
        Response variable data in an ndarray of shape (N,), where N is the
        number of observations (same N as for X).
    
    prior : dict
        The parameters of the multivariate normal prior distribution for `phi`
        provided in a dict containing either:
            1)  mean parameters with keys 'm' and 'S'
            2)  natural parameters with keys 'r' and 'Q'.
        The matrix 'Q' should be F contiguous (copy made if not).
    
    """
    
    def __init__(self, site_model, X, y, prior=None):
        
        # Ensure C contiguous
        self.X = np.ascontiguousarray(X)
        
        # Initialise prior
        if not prior:
            # Use default prior
            pass
        else:
            # Use provided prior
            if not hasattr(a, 'has_key'):
                raise TypeError("Argument `prior` is of wrong type")
            if prior.has_key('Q') and prior.has_key('r'):
                # In a natural form already
                self.Q0 = np.asfortranarray(prior['Q'])
                self.r0 = prior['r']
            elif prior.has_key('S') and prior.has_key('m'):
                # Convert into natural format
                self.Q0 = invert_spd(prior['S'])
                self.r0 = np.dot(self.Q0, prior['m'])
            else:
                raise ValueError("Argument `prior` is not appropriate")
    
    










