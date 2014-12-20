"""An implementation of a distributed EP algorithm described in an article
"Expectation propagation as a way of life" (arXiv:1412.4869).

Currently the implementation works serially with shared memory between workers.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
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


def invert_normal_params(A, b=None, out_A=None, out_b=None):
    """Invert moment parameters into natural parameters or vice versa.
    
    Switch between moment parameters (S,m) and natural parameters (Q,r) of
    a multivariate normal distribution. Providing (S,m) yields (Q,r) and vice
    versa.
    
    Parameters
    ----------
    A : ndarray
        A symmetric positive-definite matrix to be inverted. Either the
        covariance matrix S or the precision matrix Q.
    
    b : {None, ndarray}, optional
        The mean vector m, the natural parameter vector r, or None (default)
        if `out_b` is not requested.
    
    out_A, out_b : {None, ndarray, 'in_place'}, optional
        Spesifies where the output is calculate into; None (default) indicates
        that a new array is created, providing a string 'in_place' overwrites
        the corresponding input array.
    
    Returns
    -------
    out_A, out_b : ndarray
        The corresponding output arrays (`out_A` in F-order). If `b` was not
        provided, `out_b` is None.
    
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
    if b:
        if out_b == 'in_place':
            out_b = b
        elif out_b is None:
            out_b = b.copy()
        else:
            np.copyto(out_b, b)
    else:
        out_b = None
    
    # Invert
    # N.B. The following two lines could also be done with linalg.solve but this
    # shows more clearly what is happening.
    cho = linalg.cho_factor(out_A, overwrite_a=True)
    if out_b:
        linalg.cho_solve(cho, out_b, overwrite_b=True)
    _, info = dpotri_routine(out_A, overwrite_c=True)
    if info:
        # This should never occour because cho_factor was succesful ... I think
        raise linalg.LinAlgError(
                "dpotri LAPACK routine failed with error code {}".format(info))
    # Copy the upper triangular into the bottom
    copy_triu_to_tril(out_A)
    return out_A, out_b


def get_last_sample(fit, out=None):
    """Extract the last sample from a PyStan fit object.
    
    Parameters
    ----------
    fit :  StanFit4<model_name>
        Instance containing the fitted results.
    out : list of dict, optional
        The list into which the output is placed. By default a new list is
        created. Must be of appropriate shape and content (see Returns).
	    
	Returns
	-------
	list of dict
		List of nchains dicts for which each parameter name yields an ndarray
        corresponding to the sample values (similary to the init argument for
        the method StanModel.sampling).
    
    """
    
    # The following works at least for pystan version 2.5.0.0
    if not out:
        # Initialise list of dicts
        out = [{fit.model_pars[i] : np.empty(fit.par_dims[i], order='F')
                for i in range(len(fit.model_pars))} 
               for _ in range(fit.sim['chains'])]
    # Extract the sample for each chain and parameter
    for c in range(fit.sim['chains']):         # For each chain
        for i in range(len(fit.model_pars)):   # For each parameter
            p = fit.model_pars[i]
            if not fit.par_dims[i]:
                # Zero dimensional (scalar) parameter
                out[c][p][()] = fit.sim['samples'][c]['chains'][p][-1]
            elif len(fit.par_dims[i]) == 1:
                # One dimensional (vector) parameter
                for d in xrange(fit.par_dims[i][0]):
                    out[c][p][d] = fit.sim['samples'][c]['chains'] \
                                   [u'{}[{}]'.format(p,d)][-1]
            else:
                # Multidimensional parameter
                namefield = p + u'[{}' + u',{}'*(len(fit.par_dims[i])-1) + u']'
                it = np.nditer(out[c][p], flags=['multi_index'],
                               op_flags=['writeonly'], order='F')
                while not it.finished:
                    it[0] = fit.sim['samples'][c]['chains'] \
                            [namefield.format(*it.multi_index)][-1]
                    it.iternext()
    return out


class Worker(object):
    """Worker responsible of calculations for each site.
    
    Parameters
    ----------
    X : ndarray
        The C contiguous part of the explanatory variable.
    
    y : ndarray
        Part of the response variable.
    
    dphi : int
        The length of the parameter vector phi.
    
    stan_model : StanModel
        The StanModel instance responsible for the mcmc sampling.
    
    rand_state : RandomState
        np.random.RandomState instance used for seeding the sampler.
    
    Other parameters
    ----------------
    See the class DistributedEP
    
    """
    
    DEFAULT_OPTIONS = {
        'init_prev'     : True,
        'smooth'        : None,
        'smooth_ignore' : 1
    }
    
    DEFAULT_STAN_PARAMS = {
        'nchains'       : 4,
        'nsamp'         : 1000,
        'warmup'        : None,
        'thin'          : 2,
        'init'          : 'random'
    }
    
    def __init__(self, X, y, dphi, stan_model, rand_state, **options):
        
        # Parse options
        self.stan_params = self.DEFAULT_STAN_PARAMS.copy()
        for (kw, val) in options.iteritems():
            if self.DEFAULT_STAN_PARAMS.has_key(kw):
                self.stan_params[kw] = val
            elif not self.DEFAULT_OPTIONS.has_key(kw):
                # Unrecognised option
                raise TypeError("Unexpected option '{}'".format(kw))
        # Set missing options to defaults
        for (kw, default) in self.DEFAULT_OPTIONS.iteritems():
            if not options.has_key(kw):
                options[kw] = default
        
        # Allocate space for calculations
        # N.B. these arrays are used for various variables
        self.M = np.empty((dphi,dphi), order='F')
        self.v = np.empty(dphi)
        self.v2 = np.empty(dphi)
        
        # Current iteration global approximations
        self.Q = None
        self.r = None
        
        # Data for stan model in method tilted
        self.data = dict(N=X.shape[0],
                         K=X.shape[1],
                         X=X,
                         y=y,
                         mu_cavity=self.v,
                         Sigma_cavity=self.M.T)
                         # M transposed in order to get C-order
        
        # Store other instance variables
        self.dphi = dphi
        self.stan_model = stan_model
        self.rand_state = rand_state
        self.iteration = 0
        self.init_prev = options['init_prev']
        
        if self.init_prev:
            # Store the original init method so that it can be reset, when
            # an iteration fails
            self.init_orig = self.stan_params['init']
            if not isinstance(self.init_orig, basestring):
                # If init_prev is used, init option has to be a string
                raise ValueError("Arg. `init` has to be a string if "
                                 "`init_prev` is True")
        
        self.smooth = options['smooth']
        if self.smooth:
            # Memorise previous tilted distributions
            self.smooth = np.asarray(options['smooth'])
            # Skip some first iterations
            if options['smooth_ignore'] < 0:
                raise ValueError("Arg. `smooth_ignore` has to be non-negative")
            self.prev_stored = -options['smooth_ignore']
            # Temporary array for calculations
            self.prev_M = np.empty((dphi,dphi), order='F')
            # Arrays from the previous iterations
            self.prev_St = [np.empty((dphi,dphi), order='F')
                            for _ in range(len(self.smooth))]
            self.prev_mt = [np.empty(dphi)
                            for _ in range(len(self.smooth))]
        
        # Final tilted samples
        self.samp = None
    
    
    def cavity(self, Q, Qi, r, ri):
        
        self.Q = Q
        self.r = r
        np.subtract(self.Q, Qi, out=self.M)
        np.subtract(self.r, ri, out=self.v2)
        
        # Convert to mean-cov parameters for Stan
        try:
            invert_normal_params(self.M, self.v2,
                                 out_A='in_place', out_b=self.v)
        except linalg.LinAlgError:
            # Not positive definite
            return False
        return True
        
        
    def tilted(self, dQi, dri, store_samples=False, S_phi=None, m_phi=None):
        
        # Sample from the model
        with suppress_stdout():
            samp = self.stan_model.sampling(
                    data=self.data,
                    seed=self.rand_state,
                    pars=('phi'),
                    **self.stan_params
            )
        
        if self.init_prev:
            # Store the last sample of each chain
            if isinstance(self.stan_params['init'], basestring):
                # No samples stored before ... initialise list of dicts
                self.stan_params['init'] = get_last_sample(samp)
            else:
                get_last_sample(samp, out=self.stan_params['init'])
        
        # TODO: Make a non-copying extract
        samp = samp.extract(pars='phi')['phi']
        if store_samples:
            self.samp = samp
        
        # Assign arrays
        St = self.M
        mt = self.v
        
        # Sample mean and covariance
        np.mean(samp, 0, out=mt)
        samp -= mt
        np.dot(samp.T, samp, out=St.T)
        
        if self.smooth:
            # Smoothen the distribution
            # Use dri and dQi as a temporary arrays
            St, mt = apply_smooth(samp.shape[0], dri, dQi)
        else:
            # No smoothing at all ... normalise St
            St /= (samp.shape[0] - 1)
        
        # Default return value
        ret = True
        
        # Calculate KL-divergence between the posterior and the tilted
        # N.B. Not optimal ... allocates memory and wastes resouces in general
        # Intended for debug purposes only
        if S_phi and m_phi:
            try:
                cho_St = linalg.cho_factor(St)
                cho_S_phi = linalg.cho_factor(S_phi)
                dif_m = mt - m_phi
                ret = 0.5 * (   np.trace(linalg.cho_solve(cho_St, S_phi))
                              + np.sum(linalg.cho_solve(cho_St, dif_m)*dif_m)
                              - self.dphi
                              - 2*np.sum(   np.log(np.diag(cho_S_phi[0]))
                                          - np.log(np.diag(cho_St[0]))
                                        )
                            )
            except linalg.LinAlgError:
                # Failed update
                ret = np.nan
        
        # Convert (St,mt) to natural parameters
        Qt = self.M  # Same as St
        rt = self.v2
        try:
            invert_normal_params(St, mt, out_A='in_place', out_b=rt)
        except linalg.LinAlgError:
            # Not positive definite
            ret = False
            dQi.fill(0)
            dri.fill(0)
            if self.smooth:
                # Reset tilted memory
                self.prev_stored = 0
            if self.init_prev:
                # Reset initialisation method
                self.init = self.init_orig
        else:
            # Positive definite
            # Unbiased natural parameter estimates
            unbias_k = (samp.shape[0]-self.dphi-2)/(samp.shape[0]-1)
            Qt *= unbias_k
            rt *= unbias_k
            # Calculate the difference into the output array
            np.subtract(Qt, self.Q, out=dQi)
            np.subtract(rt, self.r, out=dri)
        
        self.iteration += 1
        return ret
    
    
    def apply_smooth(self, nsamp, temp_v, temp_M):
        """Memorise and combine previous St and mt."""
        
        St = self.M
        mt = self.v
        
        if self.prev_stored < 0:
            # Skip some first iterations ... no smoothing yet
            self.prev_stored += 1
            # Normalise St
            St /= (nsamp - 1)
        
        elif self.prev_stored == 0:
            # Store the first St and mt ... no smoothing yet
            self.prev_stored += 1
            np.copyto(self.prev_mt[0], mt)
            np.copyto(self.prev_St[0], St)
            # Normalise St
            St /= (nsamp - 1)
        
        else:
            # Smooth
            pmt = self.prev_mt
            pSt = self.prev_St
            ps = self.prev_stored                
            mt_new = self.v2
            St_new = self.prev_M
            # Calc combined mean
            np.multiply(pmt[ps-1], self.smooth[ps-1], out=mt_new)
            for i in range(ps-2,-1,-1):
                np.multiply(pmt[i], self.smooth[i], out=temp_v)
                mt_new += temp_v
            mt_new += mt
            mt_new /= 1 + self.smooth[:ps].sum()
            # Calc combined covariance matrix
            np.subtract(pmt[ps-1], mt_new, out=temp_v)
            np.multiply(temp_v[:,np.newaxis], temp_v, out=St_new)
            St_new *= self.smooth[ps-1]
            for i in range(ps-2,-1,-1):
                np.subtract(pmt[i], mt_new, out=temp_v)
                np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M)
                temp_M *= self.smooth[i]
                St_new += temp_M
            np.subtract(mt, mt_new, out=temp_v)
            np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M)
            St_new += temp_M
            St_new *= nsamp
            for i in range(ps-1,-1,-1):
                np.multiply(pSt[i], self.smooth[i], out=temp_M)
                St_new += temp_M
            St_new += St
            # Normalise St_new
            St_new /= ((1 + self.smooth[:ps].sum())*nsamp - 1)
            
            # Rotate array pointers
            temp_M2 = pSt[-1]
            temp_v2 = pmt[-1]
            for i in range(len(self.smooth)-1,0,-1):
                pSt[i] = pSt[i-1]
                pmt[i] = pmt[i-1]
            pSt[0] = St
            pmt[0] = mt
            # Redirect other pointers in the object
            self.prev_M = temp_M2
            self.v2 = temp_v2
            self.M = St_new
            self.v = mt_new
            self.data['mu_cavity'] = self.v
            self.data['Sigma_cavity'] = self.M.T                
            
            if self.prev_stored < len(self.smooth):
                self.prev_stored += 1
            
            return St_new, mt_new


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
    
    init_prev : bool, optional
        Indicates if the last sample of each chain in the group mcmc sampling is
        used as the starting point for the next iteration sampling. Default is
        True.
    
    init : {'random', '0', 0, function returning dict, list of dict}, optional
        Specifies how the initialisation is performed for the sampler (see 
        StanModel.sampling). If `init_prev` is True, this parameter affects only
        the sampling on the first iteration, and strings 'random' and '0' are
        the only acceptable values for this argument.
    
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
    
    DEFAULT_KWARGS = {
        'group_ind'         : None,
        'group_ind_ord'     : None,
        'group_sizes'       : None,
        'dphi'              : None,
        'prior'             : None,
        'seed'              : None,
        'df0'               : None,
        'df0_exp_start'     : 0.6,
        'df0_exp_end'       : 0.1,
        'df0_exp_speed'     : 0.8,
        'df_decay'          : 0.9,
        'df_treshold'       : 1e-8
    }
    
    def __init__(self, group_model, X, y, **kwargs):
        """Constructor populates the instance with the following attributes:
                iter, group_model, dphi, X, y, N, K, J, Nj, jj, jj_lim, Q0, r0,
                stan_params, init_prev, rand, df0, df_decay, df_treshold,
                smooth [, smooth_ignore]
        
        """
        # Parse keyword arguments
        worker_options = {}
        for (kw, val) in kwargs.iteritems():
            if (    Worker.DEFAULT_OPTIONS.has_key(kw)
                 or Worker.DEFAULT_STAN_PARAMS.has_key(kw)
               ):
                worker_options[kw] = val
            elif not self.DEFAULT_KWARGS.has_key(kw):
                # Unrecognised keyword argument
                raise TypeError("Unexpected keyword argument '{}'".format(kw))
        # Set missing options to defaults
        for (kw, default) in self.DEFAULT_KWARGS.iteritems():
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
            self.rand_state = kwargs['df0']
        else:
            self.rand_state = np.random.RandomState(seed=kwargs['df0'])
        
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
        
        # Get Stan model
        if isinstance(group_model, basestring):
            # From file
            with open(group_model, 'rb') as f:
                self.group_model = pickle.load(f)
        else:
            self.group_model = group_model
        
        # Populate self with other parameters
        self.iteration = 0
        
        # Initialise the workers
        self.workers = tuple(
            Worker(
                X[self.jj_lim[ji]:self.jj_lim[ji+1],:],
                y[self.jj_lim[ji]:self.jj_lim[ji+1]],
                self.dphi,
                self.group_model,
                self.rand_state,
                **worker_options
            )
            for ji in range(J)
        )
        
    
    def run(self, iterations, plot_intermediate=False):
        """Run the distributed EP algorithm.
        
        Parameters
        ----------
        iterations : int
            Number of iterations to run.
        
        plot_intermediate : bool, optional
            If true, the diagnostic plot is drawn between each iteration.
            Default is False.
        
        """
        pass # TODO


# >>> Temp solution to suppres output from STAN model (remove when fixed)
# This part of the code is by jeremiahbuddha from:
# http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
import os
class suppress_stdout(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
# <<< Temp solution to suppres output from STAN model (remove when fixed)


