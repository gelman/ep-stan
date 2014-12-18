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


def invert_normal_params(A, b, out_A=None, out_b=None):
    """Invert moment parameters into natural parameters or vice versa.
    
    Switch between moment parameters (S,m) and natural parameters (Q,r) of
    a multivariate normal distribution. Providing (S,m) yields (Q,r) and vice
    versa.
    
    Parameters
    ----------
    A : ndarray
        A symmetric positive-definite matrix to be inverted. Either the
        covariance matrix S or the precision matrix Q.
    
    b : ndarray
        Either the mean vector m or the natural parameter vector r.
    
    out_A, out_b : {None, ndarray, 'in_place'}
        Spesifies where the output is calculate into; None (default) indicates
        that a new array is created, providing a string 'in_place' overwrites
        the corresponding input array.
    
    Returns
    -------
    out_A, out_b : ndarray
        The corresponding output arrays (out_A in F-order).
    
    Raises
    ------
    LinAlgError
        If the provided array A is not positive definite.
    
    """
    # Process parameters
    if out_A == 'in_place':
        out_A = A
    elif out_A is None:
        out_A = A.copy(order='F')
    else:
        np.copyto(out_A, A)
    if not out_A.flags.farray:
        # Convert from C-order to F-order by transposing (note symmetric)
        out_A = out_A.T
        if not out_A.flags.farray:
            raise ValueError('Provided array A is inappropriate')
    if out_b == 'in_place':
        out_b = b
    elif out_b is None:
        out_b = b.copy()
    else:
        np.copyto(out_b, b)
    # Invert
    # N.B. The following two lines could also be done with linalg.solve but this
    # shows more clearly what is happening.
    cho = linalg.cho_factor(out_A, overwrite_a=True)
    linalg.cho_solve(cho, out_b, overwrite_b=True)
    _, info = dpotri_routine(out_A, overwrite_c=True)
    if info:
        # This should never occour because cho_factor was succesful ... I think
        raise linalg.LinAlgError(
                "dpotri LAPACK routine failed with error code {}".format(info))
    # Copy the upper triangular into the bottom
    copy_triu_to_tril(out_A)
    return out_A, out_b
    

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
        number of observations and K is the number of variables. `X` should be
        C contiguous (copy made if not). N.B. `X` can not be one dimensional
        because then it would not be possible to know, if the data has one
        variables and many observations or many variables and one observation,
        even though the latter is unreasonable.
    
    y : ndarray
        Response variable data in an ndarray of shape (N,), where N is the
        number of observations (same N as for X).
    
    prior : dict
        The parameters of the multivariate normal prior distribution for `phi`
        provided in a dict containing either:
            1)  moment parameters with keys 'm' and 'S'
            2)  natural parameters with keys 'r' and 'Q'.
        The matrix 'Q' should be F contiguous (copy made if not).
    
    """
    
    def __init__(self, site_model, X, y, prior=None):
        
        # Validate X
        if len(X.shape) != 2:
            raise ValueError("Argument `X` should be two dimensional")
        self.N = X.shape[0]
        self.K = X.shape[1]
        self.X = np.ascontiguousarray(X)
        
        # Validate y
        if len(y.shape) != 1:
            raise ValueError("Argument `y` should be one dimensional")
        if y.shape[0] != self.N
            raise ValueError("The shapes of `y` and `X` does not match")
        self.y = y
        
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
                self.Q0, self.r0 = invert_normal_params(prior['S'], prior['m'])
            else:
                raise ValueError("Argument `prior` is not appropriate")
    
    










