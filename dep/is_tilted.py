"""An implementation of a distributed EP algorithm described in an article
"Expectation propagation as a way of life" (arXiv:1412.4869).

Here, the tilted distribution inference is done with importance sampling
without Stan.

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

from __future__ import division
import sys
import numpy as np
from scipy import linalg

from util import invert_normal_params


class Worker(object):
    """Worker responsible of calculations for each site.
    
    Parameters
    ----------
    index : integer
        The index of this site
    
    lik : function
        The site's likelihood function that takes nsamp samples of the shared
        parameters phi (ndarray of sizse nsamp x dphi) as input and outputs the
        corresponding likelihood. The local parameters has to be sampled for
        each sample of phi within this function. The function must also accept
        np.random.RandomState sampler as named argument `rng` and output array
        as named argument `out`.
    
    dphi : int
        The length of the parameter vector phi.
    
    nsamp : int, optional
        The number of samples used for estimating the tilted distribution
        moments. The default value is 1000.
    
    rng : {None, int, np.random.RandomState}, optional
        The seed for the sampling
    
    temp_v, temp_M : ndarray, optional
        Temporary arrays for internal calculations of shapes (nsamp,) and
        (dphi,dphi) respectively. If not provided, new arrays are allocated.
    
    Notes
    -----
    Numpy random sampler does not support sampling into preallocated memory.
    If this would be supported, memory allocation could be done only once for
    the whole process.
    
    """
    
    def __init__(self, index, lik, dphi, nsamp=1000, rng=None, temp_v=None,
                 temp_M=None):
        
        # Store some instance variables
        self.index = index
        self.lik = lik
        self.dphi = dphi
        self.nsamp = nsamp
        self.iteration = 0
        
        # Allocate space for moments
        # After calling the method cavity, self.Mat holds the precision matrix
        # and self.vec holds the mean of the cavity distribution. After calling
        # the method tilted, self.Mat holds the scatter matrix  estimate and
        # self.vec holds the mean of the tilted distributions.
        self.Mat = np.empty((dphi,dphi), order='F')
        self.vec = np.empty(dphi)
        # The instance variable self.phase indicates if self.Mat and self.vec
        # contains the cavity or tilted distribution parameters:
        #     0: neither
        #     1: cavity
        #     2: tilted
        self.phase = 0
        
        # Current iteration global approximations
        self.Q = None
        self.r = None
        
        # Temporary arrays for calculations
        if temp_M is None:
            self.temp_M = np.empty((dphi,dphi), order='F')
        else:
            # Check if F-contiguous
            if temp_M.flags.farray:
                self.temp_M = temp_M
            elif temp_M.T.flags.farray:
                self.temp_M = temp_M.T
            else:
                raise ValueError('Provided temporary matrix is not contiguous')
            if self.temp.M.shape != (dphi,dphi):
                raise ValueError('Provided temporary matrix shape mismatch')
        if temp_v is None:
            self.temp_v = np.empty(nsamp)
        else:
            if temp_v.flags.carray:
                self.temp_v = temp_v
            else:
                raise ValueError('Provided temporary vector is not contiguous')
            if self.temp.v.shape != (nsamp,):
                raise ValueError('Provided temporary vector shape mismatch')
        
        # Random seed
        if not isinstance(rng, np.random.RandomState):
            self.rng = np.random.RandomState(seed=rng)
        else:
            self.rng = rng
    
    
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
        
        # Assign arrays
        St = self.Mat
        mt = self.vec
        w = self.temp_v
        
        # Sample phi (F-contiguous)
        samp = self.rng.standard_normal(size=(self.dphi,self.nsamp)).T
        # Get importance weights
        self.lik(samp, rng=self.rng, out=w)
        w_sum = w.sum()
        
        # Sample mean
        np.einsum('ij,i->j', samp, w, out=mt)
        mt /= w_sum
        
        # Sample covariance
        samp -= mt
        np.einsum('ki,kj,k->ij',samp, samp, w, out=St.T)
        St /= w_sum
        
        try:
            # Convert moment params to natural params
            invert_normal_params(St, mt, out_A=dQi, out_b=dri)
        except linalg.LinAlgError:
            # Precision estimate failed
            pos_def = False
            self.phase = 0
            dQi.fill(0)
            dri.fill(0)
        else:
            # Scale St to the sample size
            St *= self.nsamp
            # Unbiased natural parameter estimates
            unbias_k = (self.nsamp - self.dphi - 2) / self.nsamp
            dQi *= unbias_k
            dri *= unbias_k
            # Calculate the difference into the output arrays
            np.subtract(dQi, self.Q, out=dQi)
            np.subtract(dri, self.r, out=dri)
            # Set return and phase flag
            pos_def = True
            self.phase = 2
        
        self.iteration += 1
        return pos_def


class Master(object):
    """Manages the distributed EP algorithm.
    
    Parameters
    ----------
    liks : list of functions
        The site's likelihood functions that takes nsamp samples of the shared
        parameters phi (ndarray of sizse nsamp x dphi) as input and outputs the
        corresponding likelihood. The local parameters has to be sampled for
        each sample of phi within this function. The function must also accept
        np.random.RandomState sampler as named argument `rng` and output array
        as named argument `out`.
    
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
    
    seed : {None, int, RandomState}, optional
        The random seed used in the sampling. If not provided, a random seed is
        used.
    
    nsamp : int, optional
        The number of samples used for estimating the tilted distribution
        moments. The default value is 1000.
    
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
        Default value is 0.8.
    
    df_treshold : float, optional
        The treshold value for the damping factor. If the damping factor decays
        below this value, the algorithm is stopped. Default is 1e-6.
    
    """
    
    # Return codes for method run
    INFO_OK = 0
    INFO_INVALID_PRIOR = 1
    INFO_DF_TRESHOLD_REACHED_GLOBAL = 2
    INFO_DF_TRESHOLD_REACHED_CAVITY = 3
    INFO_ALL_SITES_FAIL = 4
    
    def __init__(self, liks, dphi=None, prior=None, seed=None, nsamp=1000,
                 df0=None, df0_exp_start=1.0, df0_exp_end=0.0,
                 df0_exp_speed=0.18, df_decay=0.8, df_treshold=1e-6):
        
        # Store likelihood function
        self.liks = liks
        self.K = len(liks)
        
        # Initialise prior
        if prior is None:
            # Use default prior
            if dphi is None:
                raise ValueError("If arg. `prior` is not provided, "
                                 "arg. `dphi` has to be given")
            self.Q0 = np.eye(dphi).T  # Transposed for F contiguous
            self.r0 = np.zeros(dphi)
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
            if dphi is None:
                dphi = self.Q0.shape[0]
            if self.Q0.shape[0] != dphi or self.r0.shape[0] != dphi:
                raise ValueError("Arg. `dphi` does not match with `prior`")
        self.dphi = dphi
        
        # Process seed
        if isinstance(seed, np.random.RandomState):
            self.rng = seed
        else:
            self.rng = np.random.RandomState(seed=seed)
        
        # Damping factor
        self.df_decay = df_decay
        self.df_treshold = df_treshold
        if df0 is None:
            # Use default exponential decay function
            self.df0 = lambda i: (
                  np.exp(-df0_exp_speed*(i-2)) * (df0_exp_start - df0_exp_end)
                + df0_exp_end
            )
        elif isinstance(df0, (float, int)):
            # Use constant initial damping factor
            if df0 <= 0 or df0 > 1:
                raise ValueError("Constant initial damping factor has to be "
                                 "between zero and one")
            self.df0 = lambda i: df0
        else:
            # Use provided initial damping factor function
            self.df0 = df0
        
        # Allocate common temp arrays for the workers
        w_temp_v = np.empty(nsamp)
        w_temp_M = np.empty((dphi,dphi), order='F')
        
        # Initialise the workers
        self.workers = [
            Worker(k, liks[k], dphi, nsamp, self.rng, w_temp_v, w_temp_M)
            for k in range(self.K)
        ]
        
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
        
        info : int
            Return code. Zero if all ok. See variables Master.INFO_*.
        
        """
        
        if niter < 1:
            if verbose:
                print "Nothing to do here as provided arg. `niter` is {}" \
                      .format(niter)
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
                print "Iter {}, starting df {:.3g}".format(self.iter, df)
                fail_printline_pos = False
                fail_printline_cov = False
            
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
                        fail_printline_pos = True
                        sys.stdout.write(
                            "\rNon pos. def. posterior cov, " +
                            "reducing df to {:.3}".format(df) +
                            " "*5 + "\b"*5
                        )
                        sys.stdout.flush()
                    if self.iter == 1:
                        if verbose:
                            print "\nInvalid prior."
                        if calc_moments:
                            return m_phi_s, cov_phi_s, self.INFO_INVALID_PRIOR
                        else:
                            return self.INFO_INVALID_PRIOR
                    if df < self.df_treshold:
                        if verbose:
                            print "\nDamping factor reached minimum."
                        if calc_moments:
                            return m_phi_s, cov_phi_s, \
                                self.INFO_DF_TRESHOLD_REACHED_GLOBAL
                        else:
                            return self.INFO_DF_TRESHOLD_REACHED_GLOBAL
                    continue
                
                # Cavity distributions (parallelisable)
                # -------------------------------------
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
                        if fail_printline_pos:
                            fail_printline_pos = False
                            print
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
                            print "\nDamping factor reached minimum."
                        if calc_moments:
                            return m_phi_s, cov_phi_s, \
                                self.INFO_DF_TRESHOLD_REACHED_CAVITY
                        else:
                            return self.INFO_DF_TRESHOLD_REACHED_CAVITY
            if verbose and (fail_printline_pos or fail_printline_cov):
                print
            
            if calc_moments:
                # Invert Q (chol was already calculated)
                # N.B. The following inversion could be done while
                # parallel jobs are running, thus saving time.
                invert_normal_params(cho_Q, r, out_A='in_place', out_b=m,
                                     cho_form=True)
                # Store the approximation moments
                np.copyto(m_phi_s[cur_iter], m)
                np.copyto(cov_phi_s[cur_iter], S.T)
                if verbose:
                    print "Mean and std of phi[0]: {:.3}, {:.3}" \
                          .format(m_phi_s[cur_iter,0], 
                                  np.sqrt(cov_phi_s[cur_iter,0,0]))
            
            # Tilted distributions (parallelisable)
            # -------------------------------------
            if verbose:
                    print "Process tilted distributions"
            for k in xrange(self.K):
                if verbose:
                    sys.stdout.write("\r    site {}".format(k+1)+' '*10+'\b'*9)
                    # Force flush here as it is not done automatically
                    sys.stdout.flush()
                # Process the site
                posdefs[k] = self.workers[k].tilted(dQi[:,:,k], dri[:,k])
                if verbose and not posdefs[k]:
                    sys.stdout.write("fail\n")
            if verbose:
                if np.all(posdefs):
                    print "\rAll sites ok"
                elif np.any(posdefs):
                    print "\rSome sites failed and are not updated"
                else:
                    print "\rEvery site failed"
            if not np.any(posdefs):
                if calc_moments:
                    return m_phi_s, cov_phi_s, self.INFO_ALL_SITES_FAIL
                else:
                    return self.INFO_ALL_SITES_FAIL
            
            if verbose and calc_moments:
                print "Iter {} done".format(self.iter)
            
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


