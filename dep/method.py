"""An implementation of a distributed EP algorithm described in an article
"Expectation propagation as a way of life" (arXiv:1412.4869).

This implementation works with parallel EP but the calculations are done
serially with shared memory between workers.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import sys
from timeit import default_timer as timer
import numpy as np
from scipy import linalg
from sklearn.covariance import GraphLassoCV

# LAPACK qr routine
dgeqrf_routine = linalg.get_lapack_funcs('geqrf')

from .util import (
    invert_normal_params,
    olse,
    get_last_fit_sample,
    suppress_stdout,
    load_stan,
    copy_fit_samples
)


class Worker(object):
    """Worker responsible of calculations for each site.
    
    Parameters
    ----------
    index : integer
        The index of this site
    
    stan_model : StanModel
        The StanModel instance responsible for the MCMC sampling.
    
    dphi : int
        The length of the parameter vector phi.
    
    X, y: ndarray
        The data included in this site.
    
    A : dict, optional
        Additional data included in this site.
    
    Other parameters
    ----------------
    See the class DistributedEP
    
    """
    
    DEFAULT_OPTIONS = {
        'init_prev'       : True,
        'prec_estim'      : 'sample',
        'prec_estim_skip' : 0,
        'verbose'         : True,
        'tmp_fix_32bit'   : False # FIXME: Temp fix for RandomState problem
    }
    
    DEFAULT_STAN_PARAMS = {
        'chains'          : 4,
        'iter'            : 1000,
        'warmup'          : None,
        'thin'            : 2,
        'init'            : 'random',
        'seed'            : None
    }
    
    # Available values for option `prec_estim`
    PREC_ESTIM_OPTIONS = ('sample', 'olse', 'glassocv')
    
    RESERVED_STAN_PARAMETER_NAMES = ['X', 'y', 'N', 'D', 'mu_phi', 'Omega_phi']
    
    def __init__(self, index, stan_model, dphi, X, y, A={}, **options):
        
        # Parse options
        # Set missing options to defaults
        for (kw, default) in self.DEFAULT_OPTIONS.items():
            if kw not in options:
                options[kw] = default
        for (kw, default) in self.DEFAULT_STAN_PARAMS.items():
            if kw not in options:
                options[kw] = default
        # Extranct stan parameters
        self.stan_params = {}
        for (kw, val) in options.items():
            if kw in self.DEFAULT_STAN_PARAMS:
                self.stan_params[kw] = val
            elif kw not in self.DEFAULT_OPTIONS:
                # Unrecognised option
                raise TypeError("Unexpected option '{}'".format(kw))
        
        # Allocate space for calculations
        # After calling the method cavity, self.Mat holds the precision matrix
        # and self.vec holds the mean of the cavity distribution. After calling
        # the method tilted, self.Mat holds the unnormalised covariance matrix
        # and self.vec holds the mean of the tilted distributions.
        self.Mat = np.empty((dphi,dphi), order='F')
        self.vec = np.empty(dphi)
        # The instance variable self.phase indicates if self.Mat and self.vec
        # contains the cavity or tilted distribution parameters:
        #     0: neither
        #     1: cavity
        #     2: tilted
        self.phase = 0
        # In the case of tilted distribution, the instance variable self.nsamp
        # indicates how many samples has contributed into the unnormalised
        # covariance matrix in self.Mat
        self.nsamp = None
        
        # Current iteration global approximations
        self.Q = None
        self.r = None
        
        # Temporary arrays for calculations
        self.temp_M = np.empty((dphi,dphi), order='F')
        self.temp_v = np.empty(dphi)
        
        # Data for stan model in method tilted
        self.data = dict(
            N=X.shape[0],
            X=X,
            y=y,
            mu_phi=self.vec,
            Omega_phi=self.Mat.T,  # Mat transposed in order to get C-order
            **A
        )
        # Add param `D` only if `X` is two dimensional
        if len(X.shape) == 2:
            self.data['D'] = X.shape[1]
        
        # Store other instance variables
        self.index = index
        self.stan_model = stan_model
        self.dphi = dphi
        self.iteration = 0
        
        # The last fit object
        self.fit = None
        # The last elapsed time
        self.last_time = None
        # The names of the shared parameters in this
        self.fit_pnames = list('phi[{}]'.format(i) for i in range(self.dphi))
        
        # Initialisation
        self.init_prev = options['init_prev']
        if self.init_prev:
            # Store the original init method so that it can be reset, when
            # an iteration fails
            self.init_orig = self.stan_params['init']
            if not isinstance(self.init_orig, str):
                # If init_prev is used, init option has to be a string
                raise ValueError("Arg. `init` has to be a string if "
                                 "`init_prev` is True")
        
        # Tilted precision estimate method
        self.prec_estim = options['prec_estim']
        if not self.prec_estim in self.PREC_ESTIM_OPTIONS:
            raise ValueError("Invalid value for option `prec_estim`")
        if self.prec_estim != 'sample':
            self.prec_estim_skip = options['prec_estim_skip']
        else:
            self.prec_estim_skip = 0
        if self.prec_estim == 'glassocv':
            self.glassocv = GraphLassoCV(assume_centered=True)
        
        # Verbose option
        self.verbose = options['verbose']
        
        # FIXME: Temp fix for RandomState problem in 32-bit Python
        if options['tmp_fix_32bit']:
            self.fix32bit = True
            self.rstate = self.stan_params['seed']
        else:
            self.fix32bit = False
        
    
    def cavity(self, Q, r, Qi, ri):
        """Form the cavity distribution and convert them to moment parameters.
        
        Parameters
        ----------
        Q, r : ndarray
            Natural parameters of the global approximation
        
        Qi, ri : ndarray
            Natural site parameters
        
        Returns
        -------
        pos_def
            True if the cavity distribution covariance matrix is positive
            definite. False otherwise.
        
        """
        
        self.Q = Q
        self.r = r
        np.subtract(self.Q, Qi, out=self.Mat)
        np.subtract(self.r, ri, out=self.vec)
        
        # Check if positive definite and solve the mean
        try:
            np.copyto(self.temp_M, self.Mat)
            cho = linalg.cho_factor(self.temp_M, overwrite_a=True)
            linalg.cho_solve(cho, self.vec, overwrite_b=True)
        except linalg.LinAlgError:
            # Not positive definite
            self.phase = 0
            return False
        else:
            self.phase = 1
            return True
        
        
    def tilted(self, dQi, dri, save_fit=False):
        """Estimate the tilted distribution parameters.
        
        This method estimates the tilted distribution parameters and calculates
        the resulting site parameter updates into the given arrays. The cavity
        distribution has to be calculated before this method is called, i.e. the
        method cavity has to be run before this.
        
        After calling this method the instance variables self.Mat and self.vec
        hold the tilted distribution moment parameters (note however that the
        covariance matrix is unnormalised and the number of samples contributing
        to this matrix is stored in the instance variable self.nsamp).
        
        Parameters
        ----------
        dQi, dri : ndarray
            Output arrays where the site parameter updates are placed.
        
        save_fit : bool, optional
            If True, the Stan fit-object is saved into the instance variable
            `fit` for later use. Default is False.
        
        Returns
        -------
        pos_def
            True if the estimated tilted distribution covariance matrix is
            positive definite. False otherwise.
        
        """
        
        if self.phase != 1:
            raise RuntimeError('Cavity has to be calculated before tilted.')
        
        # FIXME: Temp fix for RandomState problem in 32-bit Python
        if self.fix32bit:
            self.stan_params['seed'] = self.rstate.randint(2**31-1)
        
        # Sample from the model
        with suppress_stdout():
            time_start = timer()
            fit = self.stan_model.sampling(
                data=self.data,
                **self.stan_params
            )
            time_end = timer()
            self.last_time = (time_end - time_start)
        
        if self.verbose:
            # Mean stepsize
            steps = [np.mean(p['stepsize__'])
                     for p in fit.get_sampler_params()]
            print('\n    mean stepsize: {:.4}'.format(np.mean(steps)))
            # Max Rhat (from all but last row in the last column)
            print('    max Rhat: {:.4}'.format(
                np.max(fit.summary()['summary'][:-1,-1])
            ))
        
        if self.init_prev:
            # Store the last sample of each chain
            if isinstance(self.stan_params['init'], str):
                # No samples stored before ... initialise list of dicts
                self.stan_params['init'] = get_last_fit_sample(fit)
            else:
                get_last_fit_sample(fit, out=self.stan_params['init'])
        
        # Extract samples
        # TODO: preallocate space for samples
        samp = copy_fit_samples(fit, self.fit_pnames)
        self.nsamp = samp.shape[0]
        
        if save_fit:
            # Save fit
            self.fit = fit
        else:
            # Dereference fit here so that it can be garbage collected
            fit = None
        
        # Estimate precision matrix
        try:
            # Basic sample estimate
            if self.prec_estim == 'sample' or self.prec_estim_skip > 0:
                # Mean
                mt = np.mean(samp, axis=0, out=self.vec)
                # Center samples
                samp -= mt
                # Use QR-decomposition for obtaining Cholesky of the scatter
                # matrix (only R needed, Q-less algorithm would be nice)
                _, _, _, info = dgeqrf_routine(samp, overwrite_a=True)
                if info:
                    raise linalg.LinAlgError(
                        "dgeqrf LAPACK routine failed with error code {}"
                        .format(info)
                    )
                # Copy the relevant part of the array into contiguous memory
                np.copyto(self.Mat, samp[:self.dphi,:])
                invert_normal_params(
                    self.Mat, mt, out_A=dQi, out_b=dri,
                    cho_form=True
                )
                # Unbiased (for normal distr.) natural parameter estimates
                unbias_k = (self.nsamp - self.dphi - 2)
                dQi *= unbias_k
                dri *= unbias_k
                if self.prec_estim_skip > 0:
                    self.prec_estim_skip -= 1
            
            # Optimal linear shrinkage estimate
            elif self.prec_estim == 'olse':
                # Mean
                mt = np.mean(samp, axis=0, out=self.vec)
                # Center samples
                samp -= mt
                # Sample covariance
                np.dot(samp.T, samp, out=self.Mat.T)
                # Normalise self.Mat into dQi
                np.divide(self.Mat, self.nsamp, out=dQi)
                # Estimate
                olse(dQi, self.nsamp, P=self.Q, out='in-place')
                np.dot(dQi, mt, out=dri)
            
            # Graphical lasso with cross validation
            elif self.prec_estim == 'glassocv':
                # Mean
                mt = np.mean(samp, axis=0, out=self.vec)
                # Center samples
                samp -= mt
                # Fit
                self.glassocv.fit(samp)
                if self.verbose:
                    print('    glasso alpha: {:.4}'.format(self.glassocv.alpha_))
                np.copyto(dQi, self.glassocv.precision_.T)
                # Calculate corresponding r
                np.dot(dQi, mt, out=dri)
            
            else:
                raise ValueError("Invalid value for option `prec_estim`")
            
            # Calculate the difference into the output arrays
            np.subtract(dQi, self.Q, out=dQi)
            np.subtract(dri, self.r, out=dri)
            
        except linalg.LinAlgError:
            # Precision estimate failed
            pos_def = False
            self.phase = 0
            dQi.fill(0)
            dri.fill(0)
            if self.init_prev:
                # Reset initialisation method
                self.init = self.init_orig
        else:
            # Set return and phase flag
            pos_def = True
            self.phase = 2
        
        self.iteration += 1
        return pos_def


class Master(object):
    """Manages the distributed EP algorithm.
    
    Parameters
    ----------
    site_model : StanModel or string
        Model for sampling from the tilted distribution of a site. Can be
        provided either directly as a PyStan model instance or as filename
        string pointing to a pickled model or stan source code. The model has a
        restricted structure (see Notes).
    
    X : ndarray
        Explanatory variable data in an ndarray of shape (N,D), where N is the
        number of observations and D is the number of variables. `X` should be
        C-contiguous (copy made if not). N.B. One dimensional array of shape
        (N,) is also acceptable, in which case D is not provided to the stan
        model.
    
    y : ndarray
        Response variable data in an ndarray of shape (N,), where N is the
        number of observations (same N as for X).
    
    A : dict, optional
        Additional data for the site model. The keys in the dict are the names
        of the variables and the values are the coresponding objects e.g.
        integers or ndarrays. These arrays are distributed as a whole for each
        site. Example: {'var':[3,2]} distributes var=[3,2] to each site.
    
    A_k : dict, optional
        Additional data for the site model. The keys in the dict are the names
        of the variables and the values are array-likes of length K, where K is
        the number of sites. The first element of the array-likes are
        distributed to the first site etc. Example: {'var':[3,2]} distributes
        var=3 to the first site and var=2 to the second site.
    
    A_n : dict, optional
        Additional sliced data arrays provided for the site model. The keys in
        the dict are the names of the variables and the values are the
        coresponding ndarrays of size (N, ...). These arrays are sliced for each
        site (similary as `X` and `y`).
    
    site_ind, site_ind_ord, site_sizes : ndarray, optional
        Arrays indicating which sample belong to which site. Providing one of
        these keyword arguments is enough. If none of these are provided, a
        clustering is performed. Description of individual arguments:
            site_ind     : Array of length N containing the site number
                           (non-negative integer) of each point.
            site_ind_ord : Similary as `site_ind` but the sites are in order,
                           i.e. the samples are sorted.
            site_sizes   : Array of size K, where K is the number of sites,
                           indicating the number of samples in each site.
                           When this argument is provided, the samples are
                           assumed to be in order (similary as for argument
                           `site_ind_ord`).
        Providing `site_ind_ord` or `site_sizes` is preferable over
        `site_ind` because then the data arrays `X` and `y` does not have to be
        copied.
    
    dphi : int, optional
        Number of parameters for the site model, i.e. the length of phi
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
    seed : {None, int, RandomState}, optional
        The random seed used in the sampling. If not provided, a random seed is
        used.
    
    init_site : scalar or ndarray, optional
        The initial site precision matrix. If not provided, improper uniform
        N(0,inf I), i.e. Q is allzeroes, is used. If scalar, N(0,A^2/K I),
        where A = `init_site`, is used.
    
    overwrite_model : bool, optional
        If a string for `site_model` is provided, the model is compiled even
        if a precompiled model is found (see util.load_stan).
    
    chains : int, optional
        The number of chains in the site_model mcmc sampling. Default is 4.
    
    iter : int, optional
        The number of samples in the site_model mcmc sampling. Default
        is 1000.
    
    warmup : int, optional
        The number of samples to be discarded from the begining of each chain
        in the site_model mcmc sampling. Default is nsamp//2.
    
    thin : int, optional
        Thinning parameter for the site_model mcmc sampling. Default is 2.
    
    init_prev : bool, optional
        Indicates if the last sample of each chain in the site mcmc sampling is
        used as the starting point for the next iteration sampling. Default is
        True.
    
    init : {'random', '0', 0, function returning dict, list of dict}, optional
        Specifies how the initialisation is performed for the sampler (see 
        StanModel.sampling). If `init_prev` is True, this parameter affects only
        the sampling on the first iteration, and strings 'random' and '0' are
        the only acceptable values for this argument.
    
    prec_estim : {'sample', 'olse', 'glassocv'}
        Method for estimating the precision matrix from the tilted distribution
        samples. The available methods are:
            'sample'    : basic sample estimate
            'olse'      : optimal linear shrinkage estimate (see util.olse)
            'glassocv'  : graphical lasso estimate with cross validation
    
    prec_estim_skip : int
        Non-negative integer indicating on how many iterations from the begining
        the tilted distribution precision matrix is estimated using the default
        sample estimate instead of anything else.
    
    df0 : float or function, optional
        The initial damping factor for each iteration. Must be a number in the
        range (0,1]. If a number is given, a constant initial damping factor for
        each iteration is used. If a function is given, it must return the
        desired initial damping factor when called with the iteration number.
        If not provided, sinusoidal transition from `df0_start` to `df0_end` is
        used (see the respective parameters).
    
    df0_start, df0_end, df0_iter: float, optional
        The parameters for the default sinusoidally transitioning damping
        factor (see `df0`). Transitions from `df0_start` to `df0_end` in
        `df0_iter` iterations. Default None applies
            df0_start = 1/K,
            df0_end = 1/K + (1 - 1/K) / 2,
            df0_iter = 20.
    
    df_decay : float, optional
        The decay multiplier for the damping factor used if the resulting
        posterior covariance or cavity distributions are not positive definite.
        Default value is 0.8.
    
    df_treshold : float, optional
        The treshold value for the damping factor. If the damping factor decays
        below this value, the algorithm is stopped. Default is 1e-6.
    
    Notes
    -----
    TODO: Describe the structure of the site model.
    
    """
    
    # Return codes for method run
    INFO_OK = 0
    INFO_INVALID_PRIOR = 1
    INFO_DF_TRESHOLD_REACHED_GLOBAL = 2
    INFO_DF_TRESHOLD_REACHED_CAVITY = 3
    INFO_ALL_SITES_FAIL = 4
    
    # List of constructor default keyword arguments
    DEFAULT_KWARGS = {
        'A'                 : {},
        'A_n'               : {},
        'A_k'               : {},
        'site_ind'          : None,
        'site_ind_ord'      : None,
        'site_sizes'        : None,
        'dphi'              : None,
        'prior'             : None,
        'init_site'         : None,
        'df0'               : None,
        'df0_start'         : None,
        'df0_end'           : None,
        'df0_iter'          : 20,
        'df_decay'          : 0.8,
        'df_treshold'       : 1e-6,
        'overwrite_model'   : False
    }
    
    def __init__(self, site_model, X, y, **kwargs):
        
        # Parse keyword arguments
        self.worker_options = {}
        for (kw, val) in kwargs.items():
            if (    kw in Worker.DEFAULT_OPTIONS
                 or kw in Worker.DEFAULT_STAN_PARAMS
               ):
                self.worker_options[kw] = val
            elif kw not in self.DEFAULT_KWARGS:
                # Unrecognised keyword argument
                raise TypeError("Unexpected keyword argument '{}'".format(kw))
        # Set missing kwargs to defaults
        for (kw, default) in self.DEFAULT_KWARGS.items():
            if kw not in kwargs:
                kwargs[kw] = default
        # Set missing worker options to defaults
        for (kw, default) in Worker.DEFAULT_OPTIONS.items():
            if kw not in self.worker_options:
                self.worker_options[kw] = default
        for (kw, default) in Worker.DEFAULT_STAN_PARAMS.items():
            if kw not in self.worker_options:
                self.worker_options[kw] = default
        
        # Validate X
        self.N = X.shape[0]
        if len(X.shape) == 2:
            self.D = X.shape[1]
        elif len(X.shape) == 1:
            self.D = None
        else:
            raise ValueError("Argument `X` should be one or two dimensional")
        self.X = X
        
        # Validate y
        if len(y.shape) != 1:
            raise ValueError("Argument `y` should be one dimensional")
        if y.shape[0] != self.N:
            raise ValueError("The shapes of `y` and `X` does not match")
        self.y = y
        
        # Process site indices
        # K     : number of sites
        # Nk    : number of samples per site
        # k_ind : site index of each sample
        # k_lim : sample index limits
        if not kwargs['site_sizes'] is None:
            # Size of each site provided
            self.Nk = kwargs['site_sizes']
            self.K = len(self.Nk)
            self.k_lim = np.concatenate(([0], np.cumsum(self.Nk)))
            self.k_ind = np.empty(self.N, dtype=np.int64)
            for k in range(self.K):
                self.k_ind[self.k_lim[k]:self.k_lim[k+1]] = k
        elif not kwargs['site_ind_ord'] is None:
            # Sorted array of site indices provided
            self.k_ind = kwargs['site_ind_ord']
            self.Nk = np.bincount(self.k_ind)
            self.K = len(self.Nk)
            self.k_lim = np.concatenate(([0], np.cumsum(self.Nk)))
        elif not kwargs['site_ind'] is None:
            # Unsorted array of site indices provided
            k_ind = kwargs['site_ind']
            k_sort = k_ind.argsort(kind='mergesort') # Stable sort
            self.k_ind = k_ind[k_sort]
            self.Nk = np.bincount(self.k_ind)
            self.K = len(self.Nk)
            self.k_lim = np.concatenate(([0], np.cumsum(self.Nk)))
            # Copy X and y to a new sorted array
            self.X = self.X[k_sort]
            self.y = self.y[k_sort]
        else:
            raise NotImplementedError("Auto clustering not yet implemented")
        if self.k_lim[-1] != self.N:
            raise ValueError("Site definition does not match with `X`")
        if np.any(self.Nk == 0):
            raise ValueError("Empty sites: {}. Index the sites from 1 to K-1"
                             .format(np.nonzero(self.Nk==0)[0]))
        if self.K < 2:
            raise ValueError("Distributed EP should be run with at least "
                             "two sites.")
        
        # Ensure that X and y are C contiguous
        self.X = np.ascontiguousarray(self.X)
        self.y = np.ascontiguousarray(self.y)
        
        # Process A
        self.A = kwargs['A']
        # Check for name clashes
        for key in self.A.keys():
            if key in Worker.RESERVED_STAN_PARAMETER_NAMES:
                raise ValueError("Additional data name {} clashes.".format(key))
        # Process A_n
        self.A_n = kwargs['A_n'].copy()
        for (key, val) in kwargs['A_n'].items():
            if val.shape[0] != self.N:
                raise ValueError("The shapes of `A_n[{}]` and `X` does not "
                                 "match".format(repr(key)))
            # Check for name clashes
            if (    key in Worker.RESERVED_STAN_PARAMETER_NAMES
                 or key in self.A
               ):
                raise ValueError("Additional data name {} clashes.".format(key))
            # Ensure C-contiguous
            if not val.flags['CARRAY']:
                self.A_n[key] = np.ascontiguousarray(val)
        # Process A_k
        self.A_k = kwargs['A_k']
        for (key, val) in self.A_k.items():
            # Check for length
            if len(val) != self.K:
                raise ValueError("Array-like length mismatch in `A_k` "
                                 "(should be: {}, found: {})"
                                 .format(self.K, len(val)))
            # Check for name clashes
            if (    key in Worker.RESERVED_STAN_PARAMETER_NAMES
                 or key in self.A
                 or key in self.A_n
               ):
                raise ValueError("Additional data name {} clashes.".format(key))
        
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
            if 'Q' in prior and 'r' in prior:
                # In a natural form already
                self.Q0 = np.asfortranarray(prior['Q'])
                self.r0 = prior['r']
            elif 'S' in prior and 'm' in prior:
                # Convert into natural format
                self.Q0, self.r0 = invert_normal_params(prior['S'], prior['m'])
            else:
                raise ValueError("Argument `prior` is not appropriate")
            if self.dphi is None:
                self.dphi = self.Q0.shape[0]
            if self.Q0.shape[0] != self.dphi or self.r0.shape[0] != self.dphi:
                raise ValueError("Arg. `dphi` does not match with `prior`")
        
        # Damping factor
        self.df_decay = kwargs['df_decay']
        self.df_treshold = kwargs['df_treshold']
        if kwargs['df0'] is None:
            # Use default sinusoidal function
            df0_start = kwargs['df0_start']
            if df0_start is None:
                df0_start = 1.0 / self.K
            df0_end = kwargs['df0_end']
            if df0_end is None:
                df0_end = ((self.K - 1) * 0.5 + 1) / self.K
            df0_iter = kwargs['df0_iter']
            self.df0 = lambda i: (
                df0_start + (df0_end - df0_start) * 0.5 * (
                    1 + np.sin(
                        np.pi * (
                            max(0, min(i-2, df0_iter-1))
                            / (df0_iter - 1) - 0.5
                        )
                    )
                )
            )
        elif isinstance(kwargs['df0'], (float, int)):
            # Use constant initial damping factor
            if kwargs['df0'] <= 0 or kwargs['df0'] > 1:
                raise ValueError("Constant initial damping factor has to be "
                                 "in (0,1]")
            self.df0 = lambda i: kwargs['df0']
        else:
            # Use provided initial damping factor function
            self.df0 = kwargs['df0']
        
        # Get Stan model
        if isinstance(site_model, str):
            # From file
            self.site_model = load_stan(site_model,
                                         overwrite=kwargs['overwrite_model'])
        else:
            self.site_model = site_model
        
        # Process seed in worker options
        if not isinstance(self.worker_options['seed'], np.random.RandomState):
            self.worker_options['seed'] = \
                np.random.RandomState(seed=self.worker_options['seed'])
        
        # Initialise the workers
        self.workers = []
        for k in range(self.K):
            A = dict((key, val[self.k_lim[k]:self.k_lim[k+1]])
                     for (key, val) in self.A_n.items())
            A.update(self.A)
            for (key, val) in self.A_k.items():
                A[key] = val[k]
            self.workers.append(
                Worker(
                    k,
                    self.site_model,
                    self.dphi,
                    X[self.k_lim[k]:self.k_lim[k+1]],
                    y[self.k_lim[k]:self.k_lim[k+1]],
                    A=A,
                    **self.worker_options
                )
            )
        
        # Allocate space for calculations
        # Mean and cov of the approximation
        self.S = np.empty((self.dphi,self.dphi), order='F')
        self.m = np.empty(self.dphi)
        # Natural parameters of the approximation
        self.Q = self.Q0.copy(order='F')
        self.r = self.r0.copy()
        # Natural site parameters
        self.Qi = np.zeros((self.dphi,self.dphi,self.K), order='F')
        self.ri = np.zeros((self.dphi,self.K), order='F')
        # Natural site proposal parameters
        self.Qi2 = np.zeros((self.dphi,self.dphi,self.K), order='F')
        self.ri2 = np.zeros((self.dphi,self.K), order='F')
        # Site parameter updates
        self.dQi = np.zeros((self.dphi,self.dphi,self.K), order='F')
        self.dri = np.zeros((self.dphi,self.K), order='F')
        
        if not kwargs['init_site'] is None:
            # Config initial site distributions
            if isinstance(kwargs['init_site'], np.ndarray):
                for k in range(self.K):
                    np.copyto(self.Qi[:,:,k], kwargs['init_site'])
            else:
                diag_elem = self.K / (kwargs['init_site']**2)
                for k in range(self.K):
                    self.Qi[:,:,k].flat[::self.dphi+1] = diag_elem
        
        # Track iterations
        self.iter = 0
    
    
    def run(self, niter, calc_moments=True, save_last_fits=True, verbose=True):
        """Run the distributed EP algorithm.
        
        Parameters
        ----------
        niter : int
            Number of iterations to run.
        
        calc_moments : bool, optional
            If True, the moment parameters (mean and covariance) of the
            posterior approximation are calculated every iteration and returned.
            Default is True.
        
        save_last_fits : bool
            If True (default), the Stan fit-objects from the last iteration are saved for future use (mix_phi and mix_pred methods).
        
        verbose : bool, optional
            If true, some progress information is printed. Default is True.
        
        Returns
        -------
        m_phi, var_phi : ndarray
            Mean and variance of the posterior approximation at every iteration.
            Returned only if `calc_moments` is True.
        
        info : int
            Return code. Zero if all ok. See variables Master.INFO_*.
        
        """
        
        if niter < 1:
            if verbose:
                print("Nothing to do here as provided arg. `niter` is {}" \
                      .format(niter))
            if calc_moments:
                return None, None, self.INFO_OK
            else:
                return self.INFO_OK
        
        # Localise some instance variables
        # Mean and cov of the posterior approximation
        S = self.S
        m = self.m
        # Natural parameters of the approximation
        Q = self.Q
        r = self.r
        # Natural site parameters
        Qi = self.Qi
        ri = self.ri
        # Natural site proposal parameters
        Qi2 = self.Qi2
        ri2 = self.ri2
        # Site parameter updates
        dQi = self.dQi
        dri = self.dri
        
        # Array for positive definitness checking of each cavity distribution
        posdefs = np.empty(self.K, dtype=bool)
        
        if calc_moments:
            # Allocate memory for results
            m_phi_s = np.zeros((niter, self.dphi))
            cov_phi_s = np.zeros((niter, self.dphi, self.dphi))
        
        # Monitor sampling times
        stimes = np.zeros(niter)
        
        # Iterate niter rounds
        for cur_iter in range(niter):
            self.iter += 1
            # Initial dampig factor
            if self.iter > 1:
                df = self.df0(self.iter)
            else:
                # At the first round (rond zero) there is nothing to damp yet
                df = 1
            if verbose:
                print("Iter {}, starting df {:.3g}".format(self.iter, df))
                fail_printline_pos = False
                fail_printline_cov = False
            
            while True:
                # Try to update the global posterior approximation
                
                # These 4 lines could be run in parallel also
                np.add(Qi, np.multiply(df, dQi, out=Qi2), out=Qi2)
                np.add(ri, np.multiply(df, dri, out=ri2), out=ri2)
                np.add(Qi2.sum(2, out=Q), self.Q0, out=Q)
                np.add(ri2.sum(1, out=r), self.r0, out=r)
                # N.B. In the first iteration Q=Q0, r=r0 (if zero initialised)
                
                # Check for positive definiteness
                cho_Q = S
                np.copyto(cho_Q, Q)
                try:
                    linalg.cho_factor(cho_Q, overwrite_a=True)
                except linalg.LinAlgError:
                    # Not positive definite -> reduce damping factor
                    df *= self.df_decay
                    if verbose:
                        fail_printline_pos = True
                        sys.stdout.write(
                            "\rNon pos. def. posterior cov, " +
                            "reducing df to {:.3}".format(df) +
                            " "*5 + "\b"*5
                        )
                        sys.stdout.flush()
                    if self.iter == 1:
                        if verbose:
                            print("\nInvalid prior.")
                        if calc_moments:
                            return m_phi_s, cov_phi_s, self.INFO_INVALID_PRIOR
                        else:
                            return self.INFO_INVALID_PRIOR
                    if df < self.df_treshold:
                        if verbose:
                            print("\nDamping factor reached minimum.")
                        if calc_moments:
                            return m_phi_s, cov_phi_s, \
                                self.INFO_DF_TRESHOLD_REACHED_GLOBAL
                        else:
                            return self.INFO_DF_TRESHOLD_REACHED_GLOBAL
                    continue
                
                # Cavity distributions (parallelisable)
                # -------------------------------------
                # Check positive definitness for each cavity distribution
                for k in range(self.K):
                    posdefs[k] = \
                        self.workers[k].cavity(Q, r, Qi2[:,:,k], ri2[:,k])
                    # Early stopping criterion (when in serial)
                    if not posdefs[k]:
                        break
                
                if np.all(posdefs):
                    # All cavity distributions are positive definite.
                    # Accept step (switch Qi-Qi2 and ri-ri2)
                    temp = Qi
                    Qi = Qi2
                    Qi2 = temp
                    temp = ri
                    ri = ri2
                    ri2 = temp
                    self.Qi = Qi
                    self.Qi2 = Qi2
                    self.ri = ri
                    self.ri2 = ri2
                    break
                    
                else:
                    # Not all cavity distributions are positive definite ...
                    # reduce the damping factor
                    df *= self.df_decay
                    if verbose:
                        if fail_printline_pos:
                            fail_printline_pos = False
                            print()
                        fail_printline_cov = True
                        sys.stdout.write(
                            "\rNon pos. def. cavity, " +
                            "(first encountered in site {}), "
                            .format(np.nonzero(~posdefs)[0][0]) +
                            "reducing df to {:.3}".format(df) +
                            " "*5 + "\b"*5
                        )
                        sys.stdout.flush()
                    if df < self.df_treshold:
                        if verbose:
                            print("\nDamping factor reached minimum.")
                        if calc_moments:
                            return m_phi_s, cov_phi_s, \
                                self.INFO_DF_TRESHOLD_REACHED_CAVITY
                        else:
                            return self.INFO_DF_TRESHOLD_REACHED_CAVITY
            if verbose and (fail_printline_pos or fail_printline_cov):
                print()
            
            if calc_moments:
                # Invert Q (chol was already calculated)
                # N.B. The following inversion could be done while
                # parallel jobs are running, thus saving time.
                invert_normal_params(cho_Q, r, out_A='in-place', out_b=m,
                                     cho_form=True)
                # Store the approximation moments
                np.copyto(m_phi_s[cur_iter], m)
                np.copyto(cov_phi_s[cur_iter], S.T)
                if verbose:
                    print("Mean and std of phi[0]: {:.3}, {:.3}" \
                          .format(m_phi_s[cur_iter,0], 
                                  np.sqrt(cov_phi_s[cur_iter,0,0])))
            
            # Tilted distributions (parallelisable)
            # -------------------------------------
            if verbose:
                    print("Process tilted distributions")
            for k in range(self.K):
                if verbose:
                    sys.stdout.write("\r    site {}".format(k+1)+' '*10+'\b'*9)
                    # Force flush here as it is not done automatically
                    sys.stdout.flush()
                # Process the site
                posdefs[k] = self.workers[k].tilted(
                    dQi[:,:,k],
                    dri[:,k],
                    save_fit = (save_last_fits and cur_iter == niter-1)
                )
                if verbose and not posdefs[k]:
                    sys.stdout.write("fail\n")
            if verbose:
                if np.all(posdefs):
                    print("\rAll sites ok")
                elif np.any(posdefs):
                    print("\rSome sites failed and are not updated")
                else:
                    print("\rEvery site failed")
            if not np.any(posdefs):
                if calc_moments:
                    return m_phi_s, cov_phi_s, self.INFO_ALL_SITES_FAIL
                else:
                    return self.INFO_ALL_SITES_FAIL
            
            # Store max sampling time
            stimes[cur_iter] = max([w.last_time for w in self.workers])
            
            if verbose and calc_moments:
                print(("Iter {} done, max sampling time {}"
                      .format(self.iter, stimes[cur_iter])))
        
        if verbose:
            print(("{} iterations done\nTotal limiting sampling time: {}"
                  .format(niter, stimes.sum())))
        
        if calc_moments:
            return m_phi_s, cov_phi_s, self.INFO_OK
        else:
            return self.INFO_OK
    
    
    def mix_phi(self, out_S=None, out_m=None):
        """Form the posterior approximation of phi by mixing the last samples.
        
        Mixes the last obtained MCMC samples from the tilted distributions to
        obtain an approximation to the posterior.
        
        Parameters
        ----------
        out_S, out_m : ndarray, optional
            The output arrays into which the approximation covariance and mean
            are stored.
        
        Returns
        -------
        S, m : ndarray
            The combined covariance matrix and the mean vector.
        
        """
        if self.iter == 0:
            raise RuntimeError("Can not mix samples before at least one "
                               "iteration has been done.")
        if not out_S:
            out_S = np.zeros((self.dphi,self.dphi), order='F')
        if not out_m:
            out_m = np.zeros(self.dphi)
        temp_M = np.empty((self.dphi,self.dphi), order='F')
        temp_v = np.empty(self.dphi)
        
        # Combine from all the sites
        nsamp_tot = 0
        means = []
        nsamps = []
        for k in range(self.K):
            samp = self.workers[k].fit.extract(pars='phi')['phi']
            nsamp = samp.shape[0]
            nsamps.append(nsamp)
            nsamp_tot += nsamp
            mt = np.mean(samp, axis=0)
            means.append(mt)
            samp -= mt
            out_m += mt
            samp.T.dot(samp, out=temp_M.T)
            out_S += temp_M
        out_m /= self.K
        for k in range(self.K):
            np.subtract(means[k], out_m, out=temp_v)
            np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M.T)
            temp_M *= nsamps[k]
            out_S += temp_M
        out_S /= nsamp_tot - 1
        
        return out_S, out_m
    
    
    def mix_pred(self, params, smap=None, param_shapes=None):
        """Get mean and variance prediction of required parameters.
        
        Mixes the last obtained MCMC samples from the tilted distributions to
        obtain an approximation to the posterior of required parameters.
        
        TODO: It would be nice to be able to weight the contributions of the 
        sites e.g. according to the number of samples. For example if there is 
        two sites, a and b, which both have 200 samples of alpha. Samples in a 
        have been sampled using 80 observations but b has only 5  associated 
        observations. The moments of a should be weighed more maby?
        
        Parameters
        ----------
        params : str or list of str
            The required parameter names.
        
        smap : mapping or list of mapping
            Mapping from each sites indexes to the global parameter indexes:
            mapping[k] is an sequence of length of the shape of the site
            parameter in site k, e.g. mapping[k] = ((1,3), slice(None)) will
            map the whole row 0 from the site k to row 1 in the global parameter
            and row 1 to 3. mapping None corresponds to direct index mapping
            i.e. when site parameters are consistent with global parameters.
            Do not provide this for scalar parameters.
        
        param_shapes : seq or list o seq
            The shape of the global parameter. Must be given if smap is used.
        
        Returns
        -------
        mean, var : ndarray or list of ndarray
            The corresponding mean and variance of the required parameters.
        
        Examples
        --------
        Global param alpha[3]
        2 sites with first site sampling alpha[1] and alpha[2]
        
        """
        if self.iter == 0:
            raise RuntimeError("Can not mix samples before at least one "
                               "iteration has been done.")
        
        # Check if one or multiple parameters are requested
        if isinstance(params, str):
            only_one_param = True
            params = [params]
            smap = [smap]
            param_shapes = [param_shapes]
        else:
            only_one_param = False
        
        # Process each parameter
        mean = []
        var = []
        for ip in range(len(params)):
            
            # Gather moments from each worker
            par = params[ip]
            sit = smap[ip] if smap is not None else None
            
            if sit is None:
                # Every site contribute to the parmeter
                fit = self.workers[0].fit
                samp = fit.extract(pars=par)[par]
                # Ensure that one dimensional parameters with length 1 are not
                # scalarised
                if fit.par_dims[fit.model_pars.index(par)] == [1]:
                    samp = samp[:,np.newaxis]
                par_shape = list(samp.shape)
                par_shape[0] = len(self.workers)
                # Get the moments
                ns = np.empty(len(self.workers), dtype=np.int64)
                ms = np.empty(par_shape)
                vs = np.empty(par_shape)
                ns[0] = samp.shape[0]
                ms[0] = np.mean(samp, axis=0)
                vs[0] = np.sum(samp**2, axis=0) - ns[0]*(ms[0]**2)
                for iw in range(1,len(self.workers)):
                    fit = self.workers[iw].fit
                    samp = fit.extract(pars=par)[par]
                    # Ensure that one dimensional parameters with length 1 are
                    # not scalarised
                    if fit.par_dims[fit.model_pars.index(par)] == [1]:
                        samp = samp[:,np.newaxis]
                    # Moments of current site
                    ns[iw] = samp.shape[0]
                    ms[iw] = np.mean(samp, axis=0)
                    samp -= ms[iw]
                    np.square(samp, out=samp)
                    vs[iw] = np.sum(samp, axis=0)
                
                # Combine moments
                n = np.sum(ns)
                mc = np.sum((ms.T*ns).T, axis=0)
                mc /= n
                temp = ms-mc
                np.square(temp, out=temp)
                np.multiply(temp.T, ns, out=temp.T)
                temp += vs
                vc = np.sum(temp, axis=0)
                vc /= (n-1)
                mean.append(mc)
                var.append(vc)
            
            else:
                # Parameters not consistent among sites
                par_shape = param_shapes[ip]
                ns = np.empty(len(self.workers), dtype=np.int64)
                ms = []
                vs = []
                count = np.zeros(par_shape)
                for iw in range(len(self.workers)):
                    count[sit[iw]] += 1  # Check smap                    
                    fit = self.workers[iw].fit
                    samp = fit.extract(pars=par)[par]
                    # Ensure that one dimensional parameters with length 1 are
                    # not scalarised
                    if fit.par_dims[fit.model_pars.index(par)] == [1]:
                        samp = samp[:,np.newaxis]
                    # Moments of current site
                    ns[iw] = samp.shape[0]
                    ms.append(np.mean(samp, axis=0))
                    samp -= ms[iw]
                    np.square(samp, out=samp)
                    vs.append(np.sum(samp, axis=0))
                if np.count_nonzero(count) != count.size:
                    raise ValueError("Arg. `smap` does not fill the parameter")
                
                # Combine
                onecont = count == 1
                if np.all(onecont):
                    
                    # Every index has only one contribution
                    mc = np.zeros(par_shape)
                    vc = np.zeros(par_shape)
                    for iw in range(len(self.workers)):
                        mc[sit[iw]] = ms[iw]
                        vc[sit[iw]] = vs[iw]/(ns[iw]-1)
                    mean.append(mc)
                    var.append(vc)
                
                else:
                    # Combine every index
                    nc = np.zeros(par_shape, dtype=np.int64)
                    mc = np.zeros(par_shape)
                    vc = np.zeros(par_shape)
                    for iw in range(len(self.workers)):
                        nc[sit[iw]] += ns[iw]
                        mc[sit[iw]] += ns[iw]*ms[iw]
                    mc /= nc
                    for iw in range(len(self.workers)):
                        temp = np.asarray(ms[iw] - mc[sit[iw]])
                        np.square(temp, out=temp)
                        temp *= ns[iw]
                        temp += vs[iw]
                        vc[sit[iw]] += temp
                    vc /= (nc-1)
                    
                    if np.any(onecont):
                        # Some indexes have only one contribution
                        # Replace those with more precise values
                        for iw in range(len(self.workers)):
                            mc[sit[iw]] = ms[iw]
                            vc[sit[iw]] = vs[iw]/(ns[iw]-1)
                    
                    mean.append(mc)
                    var.append(vc)
        
        # Return
        if only_one_param:
            return mean[0], var[0]
        else:
            return mean, var


