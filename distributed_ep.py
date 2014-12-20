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
    group_model : StanModel or string
        Model for sampling from the tilted distribution of a group. Can be
        provided either directly as a PyStan model instance or as filename
        string pointing to a pickled model. The model has a restricted
        structure (see Notes).
    
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
    
    group_ind, group_ind_ord, group_sizes : ndarray, optional
        Arrays indicating which sample belong to which group. Providing one of
        these keyword arguments is enough. If none of these are provided, a
        clustering is performed. Description of individual arguments:
            group_ind     : Array of length N containing the group number
                            (non-negative integer) of each point.
            group_ind_ord : Similary as `group_ind` but the groups are in order,
                            i.e. the samples are sorted.
            group_sizes   : Array of size J, where J is the number og groups,
                            indicating the number of samples in each group.
                            When this argument is provided, the samples are
                            assumed to be in order (similary as for argument
                            `group_ind_ord`).
        Providing `group_ind_ord` or `group_sizes` is preferable over
        `group_ind` because then the data arrays `X` and `y` does not have to be
        copied.
    
    dphi : int, optional
        Number of parameters for the group model, i.e. the length of phi
        (see Notes). Has to be given if prior is not provided.
    
    prior : dict, optional
        The parameters of the multivariate normal prior distribution for phi
        provided in a dict containing either:
            1)  moment parameters with keys 'm' and 'S'
            2)  natural parameters with keys 'r' and 'Q'.
        The matrix 'Q' should be F contiguous (copy made if not). Argument
        `dphi` can be ommited if a prior is given. If prior is not given, the
        standard normal distribution is used.
    
    init_prev : bool, optional
        Indicates if the last sample of each chain in the group mcmc sampling is
        used as the starting point for the next iteration sampling. Default is
        True.
    
    Other parameters
    ----------------
    nchains : int, optional
        The number of chains in the group_model mcmc sampling. Default is 4.
    
    nsamp : int, optional
        The number of samples in the group_model mcmc sampling. Default
        is 1000.
    
    warmup : int, optional
        The number of samples to be discarded from the begining of each chain
        in the group_model mcmc sampling. Default is nsamp//2.
    
    thin : int, optional
        Thinning parameter for the group_model mcmc sampling. Default is 2.
    
    seed : {int, RandomState}, optional
        The random seed used in the sampling. If not provided, a random seed is
        used.
    
    df0 : float or function, optional
        The initial damping factor for each iteration. Must be a number in the
        range (0,1]. If a number is given, a constant initial damping factor for
        each iteration is used. If a function is given, it must return the
        desired initial damping factor when called with the iteration number.
        If not provided, an exponentially decaying function from `df0_exp_start`
        to `df0_exp_end` with speed `df0_exp_speed` is used (see the respective
        parameters).
    
    df0_exp_start, df0_exp_end, df0_exp_speed : float, optional
        The parameters for the default exponentially decreasing initial damping
        factor (see `df0`).
    
    df_decay : float, optional
        The decay multiplier for the damping factor used if the resulting
        posterior covariance or cavity distributions are not positive definite.
        Default value is 0.9.
    
    df_treshold : float, optional
        The treshold value for the damping factor. If the damping factor decays
        below this value, the algorithm is stopped. Default is 1e-8.
    
    smooth : {None, array_like}, optional
        A portion of samples from previous iterations to be taken into account
        in current round. A list of arbitrary length consisting of positive
        weights so that smooth[0] is a weight for the previous tilted
        distribution, smooth[1] is a weight for the distribution two iterations
        ago, etc. Empty list or None indicates that no smoothing is done
        (default behaviour).
    
    smooth_ignore : int, optional
        If smoothing is applied, this non-negative integer indicates how many
        iterations are performed before the smoothing is started. Default is 1.
    
    Notes
    -----
    TODO: Describe the structure of the site model.
    
    """
    
    def __init__(self, group_model, X, y, **kwargs):
        """Constructor populates the instance with the following attributes:
                iter, group_model, dphi, X, y, N, K, J, Nj, jj, jj_lim, Q0, r0,
                stan_params, init_prev, rand, df0, df_decay, df_treshold,
                smooth [, smooth_ignore]
        
        """
        # Parse keyword arguments
        default_kwargs = {
            'group_ind'         : None,
            'group_ind_ord'     : None,
            'group_sizes'       : None,
            'dphi'              : None,
            'prior'             : None,
            'nchains'           : 4,
            'nsamp'             : 1000,
            'warmup'            : None,
            'thin'              : 2,
            'init_prev'         : True,
            'seed'              : None,
            'df0'               : None,
            'df0_exp_start'     : 0.6,
            'df0_exp_end'       : 0.1,
            'df0_exp_speed'     : 0.8,
            'df_decay'          : 0.9,
            'df_treshold'       : 1e-8,
            'smooth'            : False,
            'smooth_ignore'     : 1
        }
        # Check for given unrecognised options
        for kw in kwargs.iterkeys():
            if not default_kwargs.has_key(kw):
                raise TypeError("Unexpected keyword argument '{}'".format(kw))
        # Set missing options to defaults
        for (kw, default) in default_kwargs.iteritems():
            if not kwargs.has_key(kw):
                kwargs[kw] = default
        
        # Validate X
        if len(X.shape) != 2:
            raise ValueError("Argument `X` should be two dimensional")
        self.N = X.shape[0]
        self.K = X.shape[1]
        
        # Validate y
        if len(y.shape) != 1:
            raise ValueError("Argument `y` should be one dimensional")
        if y.shape[0] != self.N:
            raise ValueError("The shapes of `y` and `X` does not match")
        self.y = y
        
        # Process group indices
        # J      : number of groups
        # Nj     : number of samples per group
        # jj     : group index of each sample
        # jj_lim : sample index limits
        if kwargs['group_sizes']:
            # Size of each group provided
            self.Nj = kwargs['group_sizes']
            self.J = len(self.Nj)
            self.jj_lim = np.concatenate(([0], np.cumsum(self.Nj)))
            self.jj = np.empty(self.N, dtype=np.int64)
            for j in xrange(self.J):
                self.jj[self.jj_lim[j]:self.jj_lim[j+1]] = j
            # Ensure that X is C contiguous
            self.X = np.ascontiguousarray(X)
        elif kwargs['group_ind_ord']:
            # Sorted array of group indices provided
            self.jj = kwargs['group_ind_ord']
            self.Nj = np.bincount(self.jj)
            self.J = len(self.Nj)
            self.jj_lim = np.concatenate(([0], np.cumsum(self.Nj)))
            # Ensure that X is C contiguous X
            self.X = np.ascontiguousarray(X)
        elif kwargs['group_ind']:
            # Unsorted array of group indices provided
            jj = kwargs['group_ind']
            jj_sort = jj.argsort(kind='mergesort') # Stable sort
            self.jj = jj[jj_sort]
            self.Nj = np.bincount(self.jj)
            self.J = len(self.Nj)
            self.jj_lim = np.concatenate(([0], np.cumsum(self.Nj)))
            # Copy X to a new sorted array
            self.X = X[jj_sort]
        else:
            raise NotImplementedError('Auto clustering not yet implemented')
        if self.jj_lim[-1] != self.N:
            raise ValueError("Group definition does not match with `X`")
        
        # Initialise prior
        prior = kwargs['prior']
        self.dphi = kwargs['dphi']
        if prior is None:
            # Use default prior
            if self.dphi is None:
                raise ValueError("If arg. `prior` is not provided, "
                                 "arg. `dphi` has to be given")
            self.Q0 = np.eye(self.dphi).T  # Transposed for F contiguous
            self.r0 = np.zeros(self.dphi)
        else:
            # Use provided prior
            if not hasattr(prior, 'has_key'):
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
            if self.dphi is None:
                self.dphi = self.Q0.shape[0]
            if self.Q0.shape[0] != self.dphi or self.r0.shape[0] != self.dphi:
                raise ValueError("Arg. `dphi` does not match with `prior`")
        
        # Random State
        if isinstance(kwargs['df0'], np.random.RandomState):
            self.rand = kwargs['df0']
        else:
            self.rand = np.random.RandomState(seed=kwargs['df0'])
        
        # Damping factor
        self.df_decay = kwargs['df_decay']
        self.df_treshold = kwargs['df_treshold']
        if kwargs['df0'] is None:
            # Use default exponential decay function
            df0_speed = kwargs['df0_exp_speed']
            df0_start = kwargs['df0_exp_start']
            df0_end = kwargs['df0_exp_end']
            self.df0 = lambda i: \
                    np.exp(-df0_speed*i) * (df0_start - df0_end) + df0_end
        elif isinstance(kwargs['df0'], (float, int)):
            # Use constant initial damping factor
            if kwargs['df0'] <= 0 or kwargs['df0'] > 1:
                raise ValueError("Constant initial damping factor has to be "
                                 "between zero and one")
            self.df0 = lambda i: kwargs['df0']
        else:
            # Use provided initial damping factor function
            self.df0 = kwargs['df0']
        
        # Smoothing
        self.smooth = kwargs['smooth']
        if self.smooth:
            self.smooth_ignore = kwargs['smooth_ignore']
            if self.smooth_ignore < 0:
                raise ValueError("Arg. `smooth_ignore` has to be non-negative")
        
        # Extract stan model parameters
        stan_param_names = ('nchains', 'nsamp', 'warmup', 'thin')
        self.stan_params = dict((k,v) for (k,v) in kwargs.iteritems()
                                      if k in stan_param_names)
        if self.stan_params['warmup'] is None:
            self.stan_params['warmup'] = self.stan_params['nsamp']//2
        
        # Get Stan model
        if isinstance(group_model, basestring):
            # From file
            with open(group_model, 'rb') as f:
                self.group_model = pickle.load(f)
        else:
            self.group_model = group_model
        
        # Populate self with other parameters
        self.iter = 0







