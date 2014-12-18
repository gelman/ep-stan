"""An experiment for distributed EP algorithm described in an article
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


# ------------------------------------------------------------------------------
# ---------- Settings start ----------------------------------------------------
# ------------------------------------------------------------------------------

# ========== Default sampling parameters =======================================
NCHAINS = 2  # Requires NCHAINS cores
NSAMP = 330
WARMUP = 30
THIN = 2

# ========== Smoothing =========================================================
# A portion of samples from previous iterations to be taken into account in
# current round. Variable SMOOTH is a list of arbitrary length consisting of
# positive weights or empty list or None for no smoothing:
# SMOOTH[0] is a weight for the previous tilted distribution,
# SMOOTH[1] is a weight for the distribution two iterations ago, ...
SMOOTH = None
#SMOOTH = [0.05, 0.01]
SMOOTH_IGNORE = 1       # Ignore first SMOOTH_IGNORE iterations

# ========== Initialisation ====================================================
# Choose if the last sample of each iteration is remembered to use as an
# initialisation for the next iteration or if a random initialisation is used
# for each iteration.
INIT_PREV = True

# ========== Seed ==============================================================
# Use seed = None for random seed
RAND = np.random.RandomState(seed=0)

# ========== Damping factor ====================================================
DF0_START = 0.6      # Initial damping factor at the second iteration
DF0_END = 0.1        # Initial damping factor at the infinite iteration
DF0_SPEED = 0.8      # Speed of the exponential decay from DF0_START to DF0_END
DF_MULTIPLIER = 0.9  # Reducement multiplier for the damping factor
DFMIN = 1e-8         # The minimum acceptable damping factor

# ------------------------------------------------------------------------------
# ---------- Settings end ------------------------------------------------------
# ------------------------------------------------------------------------------


# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')


def main():
    
    # ----------------
    #       Data
    # ----------------
    
    J = 5                               # number of groups
    # Nj = RAND.randint(50,60,size=J)    # number of observations per group
    Nj = 50*np.ones(J)                   # number of observations per group
    N = np.sum(Nj)                       # number of observations
    K = 10                               # number of inputs
    
    M = 2                                # number of workers
    # Evenly distributed group indices for M parallel jobs
    iiM = tuple(np.arange((J//M+1)*m, (J//M+1)*(m+1))
                if m < J%M else
                np.arange(J//M*m, J//M*(m+1)) + J%M
                for m in range(M))
    
    # Observation index limits for J groups
    iiJ = np.concatenate(([0], np.cumsum(Nj)))
    
    # Model definition:
    # y_i ~ B(alpha_i + x_i * beta)
    # Local parameter alpha_i ~ N(0, sigma2_a)
    # Shared parameter phi = [log(sigma2_a), beta]
    # Hyperparameter sigma2_a
    # Fixed parameters beta
    
    # Simulate from the model
    sigma2_a = 2**2
    a_sim = RAND.randn(J)*np.sqrt(sigma2_a)
    b_sim = RAND.randn(K)
    phi_true = np.append(0.5*np.log(sigma2_a), b_sim)
    X = RAND.randn(N,K)
    y = X.dot(b_sim)
    for j in range(J):
        y[iiJ[j]:iiJ[j+1]] += a_sim[j]
    y = 1/(1+np.exp(-y))
    y = (RAND.rand(N) < y).astype(int)
    
    # ------------------------------------------------
    # Load pre-built stan model for the tilted moments
    # ------------------------------------------------
    model_name = 'hier_log'
    with open(model_name+'.pkl', 'rb') as f:
        sm = pickle.load(f)
    pars = ('phi', 'eta')  # Parameters of interest
    
    # -----------------
    # Initialise the EP
    # -----------------
    
    niter = 6
    dphi = K+1  # Number of shared parameters
    
    # Priors for the hyperparameter
    # sigma2_a = exp(phi[0]) ~ lognpdf(log(1),log(5))
    m0_a = np.log(1)
    V0_a = np.log(5)**2
    # Prior for coefs
    m0_b = 0
    V0_b = 1**2
    # Natural parameters of the prior
    Q0 = np.diag(np.append(1./V0_a, np.ones(K)/V0_b)).T
    r0 = np.append(m0_a/V0_a, np.ones(K)*(m0_b/V0_b))
    
    # Allocate space for calculations
    # Mean and cov of the approximation
    C_phi = np.empty((dphi,dphi), order='F')
    m_phi = np.empty(dphi)
    # Natural parameters of the approximation
    Q = Q0.copy(order='F')
    r = r0.copy(order='F')
    # Natural site parameters
    Qi = np.zeros((dphi,dphi,J), order='F')
    ri = np.zeros((dphi,J), order='F')
    # Natural site proposal parameters
    Qi2 = np.zeros((dphi,dphi,J), order='F')
    ri2 = np.zeros((dphi,J), order='F')
    # Site parameter updates
    dQi = np.zeros((dphi,dphi,J), order='F')
    dri = np.zeros((dphi,J), order='F')
    
    # Array for positive definitness checking of each cavity distribution
    posdefs = np.empty(J, dtype=np.bool_)
    
    # Exponentially decreasing initial damping factor for posterior
    # approximation updates
    df0 = np.hstack((1, np.exp(-DF0_SPEED*np.arange(niter-1))
                        *(DF0_START-DF0_END) + DF0_END))
    
    # Worker instances
    workers = [Worker(dphi,
                      X[iiJ[ji]:iiJ[ji+1],:],
                      y[iiJ[ji]:iiJ[ji+1]],
                      sm, pars=pars)
               for ji in range(J)]
    
    # Utility variable for copying the upper triangular to bottom
    upind = np.triu_indices(dphi,1)
    
    # Convergence analysis
    dm = np.zeros((niter,J,dphi))
    dV = np.zeros((niter,J,dphi))
    
    # Results
    m_phi_s = np.zeros((niter+1, dphi))
    var_phi_s = np.zeros((niter+1, dphi))
    err = np.zeros((niter+1, dphi))
    
    # Plotting setup
    plot_intermediate = False
    
    # ---------------------------------------------
    # Run the update algorithm for niter iterations
    # ---------------------------------------------
    
    for i1 in range(niter):
        
        df = df0[i1] # Initial damping factor
        print 'Iter {}, starting df {:.3g} ...'.format(i1+1, df)
        while True:
            
            # Try EP update
            # -------------
            
            # These 4 lines could be run in parallel also
            np.add(Qi, np.multiply(df, dQi, out=Qi2), out=Qi2)
            np.add(ri, np.multiply(df, dri, out=ri2), out=ri2)
            np.add(Qi2.sum(2, out=Q), Q0, out=Q)
            np.add(ri2.sum(1, out=r), r0, out=r)
            # N.B. In the first iteration Q=Q0 and r=r0
            
            # Check for positive definiteness
            np.copyto(C_phi, Q)
            try:
                linalg.cho_factor(C_phi, overwrite_a=True)
            except linalg.LinAlgError:
                # Not positive definite -> reduce damping factor
                df *= DF_MULTIPLIER
                print 'Neg def posterior cov,', \
                      'reducing df to {:.3}'.format(df)
                if i1 == 0:
                    print 'Invalid prior'
                    return False
                if df < DFMIN:
                    print 'Damping factor reached minimum'
                    return False
                continue
            
            # Cavity distributions (parallel)
            # -------------------------------
            # Check positive definitness for each cavity distribution
            for m in range(M):
                # Run jobs in parallel
                for j in range(len(iiM[m])):
                    ji = iiM[m][j] # Group to update
                    posdefs[ji] = workers[ji].cavity(Q, Qi2[:,:,ji],
                                                     r, ri2[:,ji])
            
            if np.all(posdefs):
                # All cavity distributions are positive definite.
                # Accept step (switch Qi-Qi2 and ri-ri2)
                temp = Qi
                Qi = Qi2
                Qi2 = temp
                temp = ri
                ri = ri2
                ri2 = temp
                
                # N.B. The following inversion could be done while parallel jobs
                # are running, thus saving time. Also this is only needed for
                # convergence analysis and final results.
                # Invert Q
                _, info = dpotri_routine(C_phi, overwrite_c=True)
                if info:
                    # Inversion failed
                    print "DPOTRI failed with error code {}".format(info)
                    return False
                # Copy the upper triangular into the bottom
                # This could be made more efficient ... cython?
                C_phi.T[upind] = C_phi[upind]
                # Calculate mean
                C_phi.dot(r, out=m_phi)
                
                break
                
            else:
                # Not all cavity distributions are positive definite ...
                # reduce the damping factor
                df *= DF_MULTIPLIER
                ndef_ind = np.nonzero(~posdefs)[0]
                print 'Neg def cavity cov in site(s) {},'.format(ndef_ind), \
                      'reducing df to {:.3}'.format(df)
                if i1 == 0:
                    print 'Invalid prior'
                    return False
                if df < DFMIN:
                    print 'Damping factor reached minimum'
                    return False
        
        # Check approximation
        # -------------------       
        m_phi_s[i1] = m_phi
        var_phi_s[i1] = np.diag(C_phi)
        np.subtract(m_phi, phi_true, out=err[i1])
        if plot_intermediate:
            pass # TODO: Plot the same plots as in the end
        
        # Tilted distributions (parallel)
        # -------------------------------
        for m in range(M):
            for j in range(len(iiM[m])):
                # Compute tilted moments for each group
                ji = iiM[m][j] # Group to update
                if i1 != niter-1:
                    dm[i1,ji,:], dV[i1,ji,:] = \
                        workers[ji].tilted(dQi[:,:,ji], dri[:,ji], C_phi, m_phi)
                else:
                    # Final iteration ... memorise samples
                    workers[ji].tilted_final()
                
        if i1 != niter-1:
            print 'Iter {} done, max diff in tilted mean {:.4}, in cov {:.4}' \
                  .format(i1+1, np.max(dm[i1]), np.max(np.sqrt(dV[i1])))
        else:
            print 'Iter {} done'.format(i1+1)
    
    # Form final approximation by mixing samples from all the sites
    print 'Form final distribution by mixing the samples from all the sites'
    # N.B. calculated variable by variable for memory reasons
    prcs_p = [2.5, 25, 50, 75, 97.5]
    prcs = np.empty((dphi,len(prcs_p)))
    for i in range(dphi):
        comb = np.concatenate([w.samp[:,i] for w in workers])
        m_phi_s[-1,i] = np.mean(comb)
        var_phi_s[-1,i] = np.var(comb, ddof=1)
        np.percentile(comb, prcs_p, overwrite_input=True, out=prcs[i])
    np.subtract(m_phi_s[-1,:], phi_true, out=err[-1])
    
    # ----------------
    #   Plot results
    # ----------------
    
    # Mean and variance as a function of the iteration
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    axs[0].plot(np.arange(niter+1), m_phi_s)
    axs[0].set_ylabel('Mean of params')
    axs[1].plot(np.arange(niter+1), np.sqrt(var_phi_s))
    axs[1].set_ylabel('Std of params')
    axs[1].set_xlabel('Iteration')
    
    # Estimates vs true values
    plt.figure()
    ax = plt.plot(m_phi_s[-1], phi_true, 'bo')[0].get_axes()
    limits = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
              max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.plot(np.vstack((m_phi_s[-1]+3*np.sqrt(var_phi_s[-1]),
                       m_phi_s[-1]-3*np.sqrt(var_phi_s[-1]))),
            np.tile(phi_true, (2,1)),
            'b-')
    ax.plot(limits, limits, 'r-')
    ax.set_ylabel('True values')
    ax.set_xlabel('Estimated values (+- 3 sigmas)')
    
    plt.show()


class Worker(object):
    """Worker responsible of calculations for each site."""
    
    def __init__(self, dphi, X, y, sm, pars=None, nchains=NCHAINS, nsamp=NSAMP,
                 warmup=WARMUP, thin=THIN):
        
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
        
        # Model and parameters
        self.sm = sm
        self.pars = pars
        self.nchains = nchains
        self.nsamp = nsamp
        self.warmup = warmup
        self.thin = thin
        # The iteration number
        self.iteration = 0
        # The initialisation method
        self.init = 'random'
        # Utilities
        self.upind = np.triu_indices(dphi,1)
        self.dphi = dphi
        
        if SMOOTH:
            # Memorise previous tilted distributions
            self.smooth = np.array(SMOOTH)
            self.prev_stored = -SMOOTH_IGNORE  # Skip some first iterations
            # Temporary array for calculations
            self.prev_M = np.empty((dphi,dphi), order='F')
            # Arrays from the previous iterations
            self.prev_St = [np.empty((dphi,dphi), order='F')
                            for _ in range(len(SMOOTH))]
            self.prev_mt = [np.empty(dphi) for _ in range(len(SMOOTH))]
        
        # Final tilted samples
        self.samp = None
    
    
    def cavity(self, Q, Qi, r, ri):
        
        # 
        self.Q = Q
        self.r = r
        np.subtract(self.Q, Qi, out=self.M)
        np.subtract(self.r, ri, out=self.v2)
        
        # Convert to mean-cov parameters for Stan
        try:
            linalg.cho_factor(self.M, overwrite_a=True)
        except linalg.LinAlgError:
            # Not positive definite
            return False
        
        # Positive definite
        # Invert self.M
        _, info = dpotri_routine(self.M, overwrite_c=True)
        if info:
            # Inversion failed
            print "DPOTRI failed with error code {}".format(info)
            return False
        # Copy the upper triangular into the bottom
        # This could be made faster and more memory efficient ... cython?
        self.M.T[self.upind] = self.M[self.upind]
        # Calculate mean and store it in self.v
        np.dot(self.M, self.v2, out=self.v)
        
        return True
        
        
    def tilted(self, dQi, dri, C_phi, m_phi):
        # This method uses shared memory with the master
        
        # Sample from the model
        with suppress_stdout():
            samp = self.sm.sampling(data=self.data, chains=self.nchains,
                                    iter=self.nsamp, warmup=self.warmup,
                                    thin=self.thin, seed=RAND, pars=self.pars,
                                    init=self.init)
        
        if INIT_PREV:
            # Store the last sample of each chain
            if self.iteration == 0:
                # First iteration ... initialise list of dicts
                self.init = get_last_sample(samp)
            else:
                get_last_sample(samp, out=self.init)
        
        # >>> This would be more efficient if samp.extract would not copy data
        # samp = samp.extract(permuted=False)[:,:,:self.dphi]
        # # Stack the chains
        # samp = np.lib.stride_tricks.as_strided(
        #            samp,
        #            (samp.shape[0]*samp.shape[1], samp.shape[2]),
        #            (samp.strides[0], samp.strides[2])
        #        )
        # <<< This would be more efficient if samp.extract would not copy data
        samp = samp.extract(pars='phi')['phi']
        # TODO: Make a non-copying extract
        
        # Reuse memory
        St = self.M
        mt = self.v
        
        # Sample mean and covariance
        np.mean(samp, 0, out=mt)
        samp -= mt
        np.dot(samp.T, samp, out=St.T)
        
        if SMOOTH:
            # Memorise and combine previous St and mt
            # TODO: Move to own method
            
            if self.prev_stored < 0:
                # Skip some first iterations ... no smoothing yet
                self.prev_stored += 1
                # Normalise St
                St /= (samp.shape[0] - 1)
            
            elif self.prev_stored == 0:
                # Store the first St and mt ... no smoothing yet
                self.prev_stored += 1
                np.copyto(self.prev_mt[0], mt)
                np.copyto(self.prev_St[0], St)
                # Normalise St
                St /= (samp.shape[0] - 1)
            
            else:
                # Smooth
                pmt = self.prev_mt
                pSt = self.prev_St
                ps = self.prev_stored                
                mt_new = self.v2
                St_new = self.prev_M
                # Use dri and dQi as a temporary arrays
                temp_v = dri
                temp_M = dQi
                # Calc combined mean
                np.multiply(pmt[ps-1], SMOOTH[ps-1], out=mt_new)
                for i in range(ps-2,-1,-1):
                    np.multiply(pmt[i], SMOOTH[i], out=temp_v)
                    mt_new += temp_v
                mt_new += mt
                mt_new /= 1 + self.smooth[:ps].sum()
                # Calc combined covariance matrix
                np.subtract(pmt[ps-1], mt_new, out=temp_v)
                np.multiply(temp_v[:,np.newaxis], temp_v, out=St_new)
                St_new *= SMOOTH[ps-1]
                for i in range(ps-2,-1,-1):
                    np.subtract(pmt[i], mt_new, out=temp_v)
                    np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M)
                    temp_M *= SMOOTH[i]
                    St_new += temp_M
                np.subtract(mt, mt_new, out=temp_v)
                np.multiply(temp_v[:,np.newaxis], temp_v, out=temp_M)
                St_new += temp_M
                St_new *= samp.shape[0]
                for i in range(ps-1,-1,-1):
                    np.multiply(pSt[i], SMOOTH[i], out=temp_M)
                    St_new += temp_M
                St_new += St
                # Normalise St_new
                St_new /= ((1 + self.smooth[:ps].sum())*samp.shape[0] - 1)
                
                # Rotate array pointers
                temp_M = pSt[-1]
                temp_v = pmt[-1]
                for i in range(len(SMOOTH)-1,0,-1):
                    pSt[i] = pSt[i-1]
                    pmt[i] = pmt[i-1]
                pSt[0] = St
                pmt[0] = mt
                St = St_new
                mt = mt_new
                # Redirect other pointers in the object
                self.prev_M = temp_M
                self.v2 = temp_v
                self.M = St_new
                self.v = mt_new
                self.data['mu_cavity'] = self.v
                self.data['Sigma_cavity'] = self.M.T                
                
                if self.prev_stored < len(SMOOTH):
                    self.prev_stored += 1
        else:
            # No smoothing at all ... normalise St
            St /= (samp.shape[0] - 1)
        
        # Calculate difference
        dm = np.abs(mt-m_phi)
        dV = np.abs(np.diag(St-C_phi))
        
        # Check if St is positive definite
        try:
            linalg.cho_factor(St, overwrite_a=True)
        except linalg.LinAlgError:
            # Not positive definite -> discard update
            print 'Neg def tilted covariance'
            dQi.fill(0)
            dri.fill(0)
            if SMOOTH:
                # Reset tilted memory
                self.prev_stored = 0
            self.iteration += 1
            return (dm, dV)
        
        # Positive definite
        # Invert St
        _, info = dpotri_routine(St, overwrite_c=True)
        if info:
            # Inversion failed
            print "DPOTRI failed with error code {}".format(info)
            dQi.fill(0)
            dri.fill(0)
            if SMOOTH:
                # Reset tilted memory
                self.prev_stored = 0
            self.iteration += 1
            return (dm, dV)
        # Copy the upper triangular into the bottom
        # This could be made faster and more memory efficient ... cython?
        St.T[self.upind] = St[self.upind]
        Qt = St
        # Unbiased precision estimate
        Qt *= (samp.shape[0]-self.dphi-2)/(samp.shape[0]-1)
        # Calc rt
        rt = self.v2
        np.dot(Qt, mt, out=rt)
        
        # Calculate the difference into the output array
        np.subtract(Qt, self.Q, out=dQi)
        np.subtract(rt, self.r, out=dri)
        
        self.iteration += 1
        return (dm, dV)
        
        
    def tilted_final(self):
        # Sample from the model
        with suppress_stdout():
            samp = self.sm.sampling(data=self.data, chains=self.nchains,
                                    iter=self.nsamp, warmup=self.warmup,
                                    thin=self.thin, seed=RAND, pars=self.pars,
                                    init=self.init)
        self.samp = samp.extract(pars='phi')['phi']



def get_last_sample(fit, out=None):
    """Extract the last sample from the PyStan fit object.
    
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


if __name__ == '__main__':
    main()



