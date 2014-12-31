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
import numpy as np
from scipy import linalg

from util import (
    invert_normal_params,
    get_last_sample,
    suppress_stdout,
    load_stan
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
        'init_prev'     : True,
        'smooth'        : None,
        'smooth_ignore' : 1,
        'tmp_fix_32bit' : False # FIXME: Temp fix for RandomState problem
    }
    
    DEFAULT_STAN_PARAMS = {
        'chains'        : 4,
        'iter'          : 1000,
        'warmup'        : None,
        'thin'          : 2,
        'init'          : 'random',
        'seed'          : None
    }
    
    def __init__(self, index, stan_model, dphi, X, y, A={}, **options):
        
        # Parse options
        # Set missing options to defaults
        for (kw, default) in self.DEFAULT_OPTIONS.iteritems():
            if not options.has_key(kw):
                options[kw] = default
        for (kw, default) in self.DEFAULT_STAN_PARAMS.iteritems():
            if not options.has_key(kw):
                options[kw] = default
        # Extranct stan parameters
        self.stan_params = {}
        for (kw, val) in options.iteritems():
            if self.DEFAULT_STAN_PARAMS.has_key(kw):
                self.stan_params[kw] = val
            elif not self.DEFAULT_OPTIONS.has_key(kw):
                # Unrecognised option
                raise TypeError("Unexpected option '{}'".format(kw))
        
        # Allocate space for calculations
        # After calling the method cavity, these arrays hold the moment
        # parameters of the cavity distribution, and after calling the method
        # tilted, these hold the moment parameters of the tilted distributions.
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
        
        # Data for stan model in method tilted
        self.data = dict(
            N=X.shape[0],
            D=X.shape[1],
            X=X,
            y=y,
            mu_phi=self.vec,
            Sigma_phi=self.Mat.T,  # Mat transposed in order to get C-order
            **A
        )
        
        # Store other instance variables
        self.index = index
        self.stan_model = stan_model
        self.dphi = dphi
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
        if not self.smooth is None and len(self.smooth) == 0:
            self.smooth = None
        if not self.smooth is None:
            # Memorise previous tilted distributions
            self.smooth = np.asarray(self.smooth)
            # Skip some first iterations
            if options['smooth_ignore'] < 0:
                raise ValueError("Arg. `smooth_ignore` has to be non-negative")
            self.prev_stored = -options['smooth_ignore']
            # Temporary array for calculations
            self.prev_M = np.empty((dphi,dphi), order='F')
            self.prev_v = np.empty(dphi)
            # Arrays from the previous iterations
            self.prev_St = [np.empty((dphi,dphi), order='F')
                            for _ in range(len(self.smooth))]
            self.prev_mt = [np.empty(dphi)
                            for _ in range(len(self.smooth))]
        
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
        
        # Convert to mean-cov parameters for Stan
        try:
            invert_normal_params(self.Mat, self.vec,
                                 out_A='in_place', out_b='in_place')
        except linalg.LinAlgError:
            # Not positive definite
            self.phase = 0
            return False
        else:
            self.phase = 1
            return True
        
        
    def tilted(self, dQi, dri):
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
            fit = self.stan_model.sampling(
                    data=self.data,
                    pars=('phi'),
                    **self.stan_params
            )
        
        if self.init_prev:
            # Store the last sample of each chain
            if isinstance(self.stan_params['init'], basestring):
                # No samples stored before ... initialise list of dicts
                self.stan_params['init'] = get_last_sample(fit)
            else:
                get_last_sample(fit, out=self.stan_params['init'])
        
        # TODO: Make a non-copying extract
        samp = fit.extract(pars='phi')['phi']
        nsamp = samp.shape[0]
        # After this the fit object can be deleted
        del fit
        
        # Assign arrays
        St = self.Mat
        mt = self.vec
        
        # Sample mean and covariance
        np.mean(samp, 0, out=mt)
        samp -= mt
        np.dot(samp.T, samp, out=St.T)
        
        if not self.smooth is None:
            # Smoothen the distribution
            # Use dri and dQi as a temporary arrays
            St, mt = self.apply_smooth(nsamp, dri, dQi)
        else:
            # No smoothing at all ... normalise St
            self.nsamp = nsamp
            np.divide(St, self.nsamp - 1, out=dQi)
        
        # Convert (St,mt) to natural parameters, St and mt are preserved
        # Make rest of the matrix calculations in place
        Qt = dQi
        rt = dri
        try:
            invert_normal_params(dQi, mt, out_A='in_place', out_b=rt)
        except linalg.LinAlgError:
            # Not positive definite
            pos_def = False
            self.phase = 0
            dQi.fill(0)
            dri.fill(0)
            if not self.smooth is None:
                # Reset tilted memory
                self.prev_stored = 0
            if self.init_prev:
                # Reset initialisation method
                self.init = self.init_orig
        else:
            # Positive definite
            pos_def = True
            self.phase = 2
            # Unbiased natural parameter estimates
            unbias_k = (samp.shape[0]-self.dphi-2)/(samp.shape[0]-1)
            Qt *= unbias_k
            rt *= unbias_k
            # Calculate the difference into the output array
            np.subtract(Qt, self.Q, out=dQi)
            np.subtract(rt, self.r, out=dri)
        
        self.iteration += 1
        return pos_def
    
    
    def apply_smooth(self, nsamp, temp_v, temp_M):
        """Memorise and combine previous St and mt."""
        
        St = self.Mat
        mt = self.vec
        
        if self.prev_stored < 0:
            # Skip some first iterations ... no smoothing yet
            self.prev_stored += 1
            # Normalise St
            self.nsamp = nsamp
            np.divide(St, self.nsamp - 1, out=temp_M)
            return St, mt
        
        elif self.prev_stored == 0:
            # Store the first St and mt ... no smoothing yet
            self.prev_stored += 1
            np.copyto(self.prev_mt[0], mt)
            np.copyto(self.prev_St[0], St)
            # Normalise St
            self.nsamp = nsamp
            np.divide(St, self.nsamp - 1, out=temp_M)
            return St, mt
            
        else:
            # Smooth
            pmt = self.prev_mt
            pSt = self.prev_St
            ps = self.prev_stored                
            mt_new = self.prev_v
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
            # N.B. This assumes that the same number of samples has been drawn
            # in each iteration
            St_new *= nsamp
            for i in range(ps-1,-1,-1):
                np.multiply(pSt[i], self.smooth[i], out=temp_M)
                St_new += temp_M
            St_new += St
            # Normalise St_new
            self.nsamp = (1 + self.smooth[:ps].sum())*nsamp
            np.divide(St_new, self.nsamp - 1, out=temp_M)
            
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
            self.prev_v = temp_v2
            self.Mat = St_new
            self.vec = mt_new
            self.data['mu_phi'] = self.vec
            self.data['Sigma_phi'] = self.Mat.T                
            
            if self.prev_stored < len(self.smooth):
                self.prev_stored += 1
            
            return St_new, mt_new


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
        C contiguous (copy made if not). N.B. `X` can not be one dimensional
        because then it would not be possible to know, if the data has one
        variables and many observations or many variables and one observation,
        even though the latter is unreasonable.
    
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
    overwrite_model : bool, optional
        If a string for `site_model` is provided, the model is compiled even
        if a precompiled model is found (see util.load_stan).
    
    nchains : int, optional
        The number of chains in the site_model mcmc sampling. Default is 4.
    
    nsamp : int, optional
        The number of samples in the site_model mcmc sampling. Default
        is 1000.
    
    warmup : int, optional
        The number of samples to be discarded from the begining of each chain
        in the site_model mcmc sampling. Default is nsamp//2.
    
    thin : int, optional
        Thinning parameter for the site_model mcmc sampling. Default is 2.
    
    seed : {None, int, RandomState}, optional
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
        Indicates if the last sample of each chain in the site mcmc sampling is
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
    
    # Return codes for method run
    INVALID_PRIOR = -1
    DF_TRESHOLD_REACHED_GLOBAL = -2
    DF_TRESHOLD_REACHED_CAVITY = -3
    
    # List of constructor default keyword arguments
    DEFAULT_KWARGS = {
        'A'                : {},
        'A_n'              : {},
        'A_k'              : {},
        'site_ind'         : None,
        'site_ind_ord'     : None,
        'site_sizes'       : None,
        'dphi'             : None,
        'prior'            : None,
        'df0'              : None,
        'df0_exp_start'    : 0.6,
        'df0_exp_end'      : 0.1,
        'df0_exp_speed'    : 0.8,
        'df_decay'         : 0.9,
        'df_treshold'      : 1e-8,
        'overwrite_model'  : False
    }
    
    def __init__(self, site_model, X, y, **kwargs):
        
        # Parse keyword arguments
        self.worker_options = {}
        for (kw, val) in kwargs.iteritems():
            if (    Worker.DEFAULT_OPTIONS.has_key(kw)
                 or Worker.DEFAULT_STAN_PARAMS.has_key(kw)
               ):
                self.worker_options[kw] = val
            elif not self.DEFAULT_KWARGS.has_key(kw):
                # Unrecognised keyword argument
                raise TypeError("Unexpected keyword argument '{}'".format(kw))
        # Set missing kwargs to defaults
        for (kw, default) in self.DEFAULT_KWARGS.iteritems():
            if not kwargs.has_key(kw):
                kwargs[kw] = default
        # Set missing worker options to defaults
        for (kw, default) in Worker.DEFAULT_OPTIONS.iteritems():
            if not self.worker_options.has_key(kw):
                self.worker_options[kw] = default
        for (kw, default) in Worker.DEFAULT_STAN_PARAMS.iteritems():
            if not self.worker_options.has_key(kw):
                self.worker_options[kw] = default
        
        # Validate X
        if len(X.shape) != 2:
            raise ValueError("Argument `X` should be two dimensional")
        self.N = X.shape[0]
        self.D = X.shape[1]
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
            for k in xrange(self.K):
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
        for key in self.A.iterkeys():
            if key in ['X', 'y', 'N', 'D', 'mu_phi', 'Sigma_phi']:
                raise ValueError("Additional data name {} clashes.".format(key))
        # Process A_n
        self.A_n = kwargs['A_n'].copy()
        for (key, val) in kwargs['A_n'].iteritems():
            if val.shape[0] != self.N:
                raise ValueError("The shapes of `A_n[{}]` and `X` does not "
                                 "match".format(repr(key)))
            # Check for name clashes
            if (    key in ['X', 'y', 'N', 'D', 'mu_phi', 'Sigma_phi']
                 or key in self.A
               ):
                raise ValueError("Additional data name {} clashes.".format(key))
            # Ensure C-contiguous
            if not val.flags['CARRAY']:
                self.A_n[key] = np.ascontiguousarray(val)
        # Process A_k
        self.A_k = kwargs['A_k']
        for (key, val) in self.A_k.iteritems():
            # Check for length
            if len(val) != self.K:
                raise ValueError("Array-like length mismatch in `A_k` "
                                 "(should be: {}, found: {})"
                                 .format(self.K, len(val)))
            # Check for name clashes
            if (    key in ['X', 'y', 'N', 'D', 'mu_phi', 'Sigma_phi']
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
        
        # Damping factor
        self.df_decay = kwargs['df_decay']
        self.df_treshold = kwargs['df_treshold']
        if kwargs['df0'] is None:
            # Use default exponential decay function
            df0_speed = kwargs['df0_exp_speed']
            df0_start = kwargs['df0_exp_start']
            df0_end = kwargs['df0_exp_end']
            self.df0 = lambda i: \
                    np.exp(-df0_speed*(i-2)) * (df0_start - df0_end) + df0_end
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
        if isinstance(site_model, basestring):
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
        for k in xrange(self.K):
            A = dict((key, val[self.k_lim[k]:self.k_lim[k+1]])
                     for (key, val) in self.A_n.iteritems())
            A.update(self.A)
            for (key, val) in self.A_k.iteritems():
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
        
        # Track iterations
        self.iter = 0
    
    
    def run(self, niter, calc_moments=True, verbose=True):
        """Run the distributed EP algorithm.
        
        Parameters
        ----------
        niter : int
            Number of iterations to run.
        
        calc_moments : bool, optional
            If True, the moment parameters (mean and covariance) of the
            posterior approximation are calculated every iteration and returned.
            Default is True.
        
        verbose : bool, optional
            If true, some progress information is printed. Default is True.
        
        Returns
        -------
        m_phi, var_phi : ndarray
            Mean and variance of the posterior approximation at every iteration.
            Returned only if `calc_moments` is True.
        
        """
        
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
            var_phi_s = np.zeros((niter, self.dphi))
        
        # Iterate niter rounds
        for cur_iter in xrange(niter):
            self.iter += 1
            # Initial dampig factor
            if self.iter > 1:
                df = self.df0(self.iter)
            else:
                # At the first round (rond zero) there is nothing to damp yet
                df = 1
            if verbose:
                print 'Iter {}, starting df {:.3g}.'.format(self.iter, df)
            
            while True:
                # Try to update the global posterior approximation
                
                # These 4 lines could be run in parallel also
                np.add(Qi, np.multiply(df, dQi, out=Qi2), out=Qi2)
                np.add(ri, np.multiply(df, dri, out=ri2), out=ri2)
                np.add(Qi2.sum(2, out=Q), self.Q0, out=Q)
                np.add(ri2.sum(1, out=r), self.r0, out=r)
                # N.B. In the first iteration Q=Q0 and r=r0
                
                # Check for positive definiteness
                cho_Q = S
                np.copyto(cho_Q, Q)
                try:
                    linalg.cho_factor(cho_Q, overwrite_a=True)
                except linalg.LinAlgError:
                    # Not positive definite -> reduce damping factor
                    df *= self.df_decay
                    if verbose:
                        print 'Neg def posterior cov,', \
                              'reducing df to {:.3}'.format(df)
                    if self.iter == 1:
                        if verbose:
                            print 'Invalid prior.'
                        return self.INVALID_PRIOR
                    if df < self.df_treshold:
                        if verbose:
                            print 'Damping factor reached minimum.'
                        return self.DF_TRESHOLD_REACHED_GLOBAL
                    continue
                
                # Cavity distributions (parallelisable)
                # -------------------------------
                # Check positive definitness for each cavity distribution
                for k in xrange(self.K):
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
                        print 'Neg.def. cavity', \
                              '(first encountered in site {}),' \
                              .format(np.nonzero(~posdefs)[0][0]), \
                              'reducing df to {:.3}.'.format(df)
                    if df < self.df_treshold:
                        if verbose:
                            print 'Damping factor reached minimum.'
                        return self.DF_TRESHOLD_REACHED_CAVITY
            
            if calc_moments:
                # Invert Q (chol was already calculated)
                # N.B. The following inversion could be done while
                # parallel jobs are running, thus saving time.
                invert_normal_params(cho_Q, r, out_A='in_place', out_b=m,
                                     cho_form=True)
                # Store the approximation moments
                np.copyto(m_phi_s[cur_iter], m)
                np.copyto(var_phi_s[cur_iter], np.diag(S))
            
            # Tilted distributions (parallelisable)
            # -------------------------------
            for k in xrange(self.K):
                posdefs[k] = self.workers[k].tilted(dQi[:,:,k], dri[:,k])
            if verbose and not np.all(posdefs):
                print 'Neg.def. tilted in site(s) {}.' \
                      .format(np.nonzero(~posdefs)[0])
            
            if verbose and calc_moments:
                print 'Iter {} done, max var in the posterior: {}' \
                      .format(self.iter, np.max(var_phi_s[cur_iter]))
            
            if cur_iter == 1:
                print "See the change e.g. in the second element of dQi[0,0,:]:"
                print dQi[0,0,:]
            
        if calc_moments:
            return m_phi_s, var_phi_s
    
    
    def mix_samples(self, out_S=None, out_m=None):
        """Form the posterior approximation by mixing the last samples.
        
        Mixes the last obtained mcmc samples from the tilted distributions to
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
            out_S = np.empty((self.dphi,self.dphi), order='F')
        if not out_m:
            out_m = np.empty(self.dphi)
        temp_M = np.empty((self.dphi,self.dphi), order='F')
        temp_v = np.empty(self.dphi)
        
        # Combine mt from every site
        np.copyto(out_m, self.workers[0].vec)
        for k in xrange(1,self.K):
            out_m += self.workers[k].vec
        out_m /= self.K
        
        # Combine St from every site
        np.subtract(self.workers[0].vec, out_m, out = temp_v)
        np.multiply(temp_v[:,np.newaxis], temp_v, out=out_S)
        out_S *= self.workers[0].nsamp
        for k in xrange(1,self.K):
            np.subtract(self.workers[k].vec, out_m, out = temp_v)
            np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M)
            temp_M *= self.workers[k].nsamp
            out_S += temp_M
        nsamp_tot = 0
        for k in xrange(self.K):
            out_S += self.workers[k].Mat
            nsamp_tot += self.workers[k].nsamp
        out_S /= nsamp_tot - 1
        
        return out_S, out_m



