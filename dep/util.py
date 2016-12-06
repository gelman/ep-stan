""" Utilities for the distributed EP algorithm.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import os
import pickle
import numpy as np
from scipy import linalg
from pystan import StanModel

from cython_util import (
    copy_triu_to_tril,
    auto_outer,
    ravel_triu,
    unravel_triu,
    fro_norm_squared
)

# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')

# Precalculated constant
_LOG_2PI = np.log(2*np.pi)


def invert_normal_params(A, b=None, out_A=None, out_b=None, cho_form=False):
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
    
    out_A, out_b : {None, ndarray, 'in-place'}, optional
        Spesifies where the output is calculate into; None (default) indicates
        that a new array is created, providing a string 'in-place' overwrites
        the corresponding input array.
    
    cho_form : bool
        If True, `A` is assumed to be the upper Cholesky of the real S or Q.
    
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
    if not isinstance(out_A, np.ndarray) and out_A == 'in-place':
        out_A = A
    elif out_A is None:
        out_A = A.copy(order='F')
    else:
        np.copyto(out_A, A)
    if not out_A.flags['FARRAY']:
        # Convert from C-order to F-order by transposing (note symmetric)
        out_A = out_A.T
        if not out_A.flags['FARRAY'] and out_A.shape[0] > 1:
            raise ValueError('Provided array A is inappropriate')
    if not b is None:
        if not isinstance(out_b, np.ndarray) and out_b == 'in-place':
            out_b = b
        elif out_b is None:
            out_b = b.copy()
        else:
            np.copyto(out_b, b)
    else:
        out_b = None
    
    # Invert
    if not cho_form:
        cho = linalg.cho_factor(out_A, overwrite_a=True)
    else:
        # Already in upper Cholesky form
        cho = (out_A, False)
    if not out_b is None:
        linalg.cho_solve(cho, out_b, overwrite_b=True)
    _, info = dpotri_routine(out_A, overwrite_c=True)
    if info:
        # This should never occour if cho_factor was succesful ... I think
        raise linalg.LinAlgError(
                "dpotri LAPACK routine failed with error code {}".format(info))
    # Copy the upper triangular into the bottom
    copy_triu_to_tril(out_A)
    return out_A, out_b


def olse(S, n, P=None, out=None):
    """Optimal linear shrinkage estimator.
    
    Estimate precision matrix form the given sample covariance matrix with
    optimal linear shrinkage method [1]_. using the naive prior matrix 1/d I,
    where d is the number of dimensions.
    
    Parameters
    ----------
    S : ndarray
        The sample covariance matrix.
    
    n : int
        Number of contributing samples
    
    P : {None, ndarray}, optional
        The prior matrix. Providing None uses the naive prior 1/d I, where d is
        the number of dimensions. Default is None.
    
    out : {None, ndarray, 'in-place'}, optional
        The output array for the precision matrix estimate.
    
    Returns
    -------
    out : ndarray
        The precision matrix estimate.
    
    References
    ----------
    .. [1] Bodnar, T., Gupta, A.K. and Parolya, N., Optimal Linear Shrinkage
       Estimator for Large Dimensional Precision Matrix, arXiv:1308.0931, 2014.
    
    """
    # Process parameters
    if not isinstance(out, np.ndarray) and out == 'in-place':
        out = S
    elif out is None:
        out = S.copy(order='F')
    else:
        np.copyto(out, S)
    if not out.flags['FARRAY']:
        # Convert from C-order to F-order by transposing (note symmetric)
        out = out.T
        if not out.flags['FARRAY']:
            raise ValueError('Provided array should be in F-order')
    # Calculate
    d = out.shape[0]
    invert_normal_params(out, out_A='in-place')
    tr = np.trace(out)
    tr2 = tr**2
    f2 = fro_norm_squared(out.T)
    if P is None:
        # Naive prior
        alpha = 1 - (d + tr2/(f2 - tr2/d))/n
        beta = tr*(1-d/n-alpha)
        out *= alpha
        out.flat[::out.shape[1]+1] += beta/d    # Add beta/d to the diagonal
    else:
        # Use provided prior
        f2p = fro_norm_squared(P.T)
        temp = out*P
        trSP = np.sum(temp)
        alpha = 1 - (d + tr2*f2p/(f2*f2p - trSP**2))/n
        beta = (trSP/f2p)*(1-d/n-alpha)
        out *= alpha
        out += np.multiply(beta, P, out=temp)
    return out


def _cv_estim(f, h, Eh, opt, cov_k=None, var_k=None, ddof_f=0, ddof_h=0,
              out=None):
    """Estimate f_hat. Used by function cv_moments."""
    n = f.shape[0]
    d = f.shape[1]
    if out is None:
        out = np.empty(d)
    # Calc mean of f and h
    np.sum(f, axis=0, out=out)
    out /= n - ddof_f
    fc = f - out
    hc = h - Eh
    # Estimate a
    if opt['multiple_cv']:
        var_h = hc.T.dot(hc).T
        cov_fh = fc.T.dot(hc).T
        if cov_k:
            cov_fh *= cov_k
        if var_k:
            var_h *= var_k
        a = linalg.solve(var_h, cov_fh, overwrite_a=True, overwrite_b=True)
    else:
        var_h = np.sum(hc**2, axis=0)
        cov_fh = np.sum(fc*hc, axis=0)
        if cov_k:
            cov_fh *= cov_k
        if var_k:
            var_h *= var_k
        a = cov_fh / var_h
    # Regulate a
    if opt['regulate_a']:
        a *= opt['regulate_a']
    if opt['max_a']:
        np.clip(a, -opt['max_a'], opt['max_a'], out=a)
    # Calc f_hat
    if ddof_h == 0:
        hm = np.mean(hc, axis=0)
    else:
        hm = np.sum(h, axis=0)
        hm /= n - ddof_h
        hm -= Eh
    if opt['multiple_cv']:
        out -= np.dot(hm, a)
    else:
        out -= np.multiply(hm, a, out=hm)
    return out, a
    

def cv_moments(samp, lp, Q_tilde, r_tilde, S_tilde=None, m_tilde=None,
               ldet_Q_tilde=None, multiple_cv=True, regulate_a=None, max_a=None,
               m_treshold=0.9, S_hat=None, m_hat=None, ret_a=False):
    """Approximate moments using control variate.
    
    N.B. This requires that the sample log probabilities are normalised!
    
    Parameters
    ----------
    samp : ndarray
        The samples from the distribution being approximated.
    
    lp : ndarray
        Log probability density at the samples.
    
    Q_tilde, r_tilde : ndarray
        The control variate distribution natural parameters.
    
    S_tilde, m_tilde : ndarray, optional
        The control variate distribution moment parameters.
    
    ldet_Q_tilde : float, optional
        Half of the logarithm of the determinant of Q_tilde, i.e. sum of the
        logarithm of the diagonal elements of Cholesky factorisation of Q_tilde.
    
    multiple_cv : bool, optional
        If this is set to True, each dimension of h is used to control each
        dimension of f. Otherwise each dimension of h control only the
        corresponding dimension of f. Default value is True.
    
    regulate_a : {None, float}, optional
        Regularisation multiplier for correlation term `a`. The estimate of `a`
        is multiplied with this value. Closer to zero may provide smaller bias
        but greater variance. Providing 1 or None corresponds to no
        regularisation.
    
    max_a : {None, float}, optional
        Maximum absolute value for correlation term `a`. If not provided or
        None, `a` is not limited.
    
    m_treshold : {float, None}, optional
        If the fraction of samples of h in one side of `m_tilde` is greater than
        this, the normal sample estimates are used instead. Providing None
        indicates that no treshold is used.
    
    S_hat, m_hat : ndarray, optional
        The output arrays (S_hat in F-order).
    
    ret_a : bool, optional
        Indicates whether a_S and a_m are returned. Default value is False.
    
    Returns
    -------
    S_hat, m_hat : ndarray
        The approximated moment parameters.
    
    treshold_exceeded : bool
        True if the control variate estimate was used and False if the normal
        sample estimate was used.
    
    a_S, a_m : float
        The respective estimates for `a`. Returned if `ret_a` is True.
    
    """
    
    opt = dict(
        multiple_cv = multiple_cv,
        regulate_a = regulate_a,
        max_a = max_a
    )
    n = samp.shape[0]
    if len(samp.shape) == 1:
        # Force samp to two dimensional
        samp = samp[:,np.newaxis]
    d = samp.shape[1]
    
    if S_hat is None:
       S_hat = np.empty((d,d), order='F')
    if m_hat is None:
       m_hat = np.empty(d)
    
    # Invert Q_tilde, r_tilde to moment params if not provided
    if S_tilde is None or m_tilde is None or ldet_Q_tilde is None:
        cho_tilde = linalg.cho_factor(Q_tilde)[0]
    if S_tilde is None or m_tilde is None:
        S_tilde, m_tilde = \
            invert_normal_params(cho_tilde, r_tilde, cho_form=True)
    
    # Calc lp_tilde
    if ldet_Q_tilde is None:
        const = np.sum(np.log(np.diag(cho_tilde))) - 0.5*d*_LOG_2PI
    else:
        const = ldet_Q_tilde - 0.5*d*_LOG_2PI
    dev_tilde = samp - m_tilde
    lp_tilde = np.sum(dev_tilde.dot(Q_tilde)*dev_tilde, axis=1)
    lp_tilde *= 0.5
    np.subtract(const, lp_tilde, out=lp_tilde)
    
    # Probability ratios
    pr = np.subtract(lp_tilde, lp, out=lp_tilde)
    pr = np.exp(pr, out=pr)    
    
    # ----------------------------------
    #   Mean
    # ----------------------------------
    f = samp
    h = samp*pr[:,np.newaxis]
    
    if m_treshold:
        # Check if the treshold ratio is exceeded
        if m_treshold < 0.5:
            m_treshold = 1 - m_treshold
        thratios = np.sum(samp < m_tilde, axis=0)/n
        if np.any(thratios > m_treshold) or np.any(thratios < 1 - m_treshold):
            # Return normal sample estimates instead
            np.mean(samp, axis=0, out=m_hat)
            samp -= m_hat
            np.dot(samp.T, samp, out=S_hat.T)
            S_hat /= n-1
            if ret_a:
                return S_hat, m_hat, False, 0, 0
            else:
                return S_hat, m_hat, False
    
    # Estimate f_hat
    _, a_m = _cv_estim(f, h, m_tilde, opt, cov_k = n, var_k = n-1, out = m_hat)
    if not ret_a:
        del a_m
    
    # ----------------------------------
    #   Covariance
    # ----------------------------------
    # Calc d+1 choose 2
    if d % 2 == 0:
        d2 = (d >> 1) * (d+1)
    else:
        d2 = ((d+1) >> 1) * d
    d2vec = np.empty(d2)
    
    # Calc h
    # dev_tilde = samp - m_tilde # Calculated before
    h = np.empty((n,d2))
    auto_outer(dev_tilde, h)
    h *= pr[:,np.newaxis]
    Eh = np.empty(d2)
    ravel_triu(S_tilde.T, Eh)
    
    # Calc f with either using the new m_hat or sample mean. If the former is
    # used, the unbiasness (ddof_f) should be examined.
    dev = samp - m_hat
    # dev = samp - np.mean(samp, axis=0)
    
    f = np.empty((n,d2))
    auto_outer(dev, f)
    
    # Estimate f_hat (for some reason ddof_h=1 might give better results)
    _, a_S = _cv_estim(f, h, Eh, opt, cov_k = n**2, var_k = (n-1)**2,
                       ddof_f = 1, ddof_h = 0, out = d2vec)
    if not ret_a:
        del a_S
    # Reshape f_hat into covariance matrix S_hat
    unravel_triu(d2vec, S_hat.T)
    
    if ret_a:
        return S_hat, m_hat, True, a_S, a_m
    else:
        return S_hat, m_hat, True


def copy_fit_samples(fit, pnames, out=None):
    """Copy the samples from PyStan fit object into F-order array.
    
    Parameters
    ----------
    fit : StanFit4<model_name>
        Instance containing the fitted results.
    pnames : list of string
        List containing the names of the parameters in desired order.
    out : ndarray, optional
        The output array.
    
	Returns
	-------
	ndarray
        Array of shape (n_samp, len(pnames)) containing the samples from all
        the chains with burn-in removed.
    
    """
    
    # The following works at least for pystan version 2.5.0.0
    ndim = len(pnames)
    nchains = fit.sim['chains']
    warmup = fit.sim['warmup2'][0]
    niter = len(fit.sim['samples'][0]['chains'][pnames[0]])
    nsamp_per_chain = niter - warmup
    nsamp = nchains * nsamp_per_chain
    if out is None:
        # Initialise output array
        out = np.empty((nsamp,ndim), order='F')
    else:
        if len(out) != nsamp or (ndim > 1 and out.shape[1] != ndim):
            raise ValueError('Invalid output array')
    # Extract the sample for each parameter and chain
    for pi in range(len(pnames)):
        i1 = 0
        for c in range(nchains):
            i2 = i1 + nsamp_per_chain
            dst = out[i1:i2,pi]
            src = fit.sim['samples'][c]['chains'][pnames[pi]][warmup:]
            np.copyto(dst, src)
            i1 = i2
    return out


def get_last_fit_sample(fit, out=None):
    """Extract the last sample from a PyStan fit object.
    
    Parameters
    ----------
    fit : StanFit4<model_name>
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
    if out is None:
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
                for d in range(fit.par_dims[i][0]):
                    out[c][p][d] = fit.sim['samples'][c]['chains'] \
                                   ['{}[{}]'.format(p,d)][-1]
            else:
                # Multidimensional parameter
                namefield = p + '[{}' + ',{}'*(len(fit.par_dims[i])-1) + ']'
                it = np.nditer(out[c][p], flags=['multi_index'],
                               op_flags=['writeonly'], order='F')
                while not it.finished:
                    it[0] = fit.sim['samples'][c]['chains'] \
                            [namefield.format(*it.multi_index)][-1]
                    it.iternext()
    return out


def load_stan(filename, overwrite=False):
    """Load or compile a stan model.
    
    Parameters
    ----------
    filename : string
        The name of the model file. It may or may not contain path and ending
        '.stan' or '.pkl'. If a respective file with ending '.pkl' is found,
        the model is not built but loaded from the pickle file (unless
        `overwrite` is True). Otherwise the model is compiled from the
        respective file ending with '.stan' and saved into '.pkl' file.
    overwrite : bool
        Compile and save a new model even if a pickled model with same name
        already exists.
    
    """
    # Remove '.pkl' or '.stan' endings
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    elif filename.endswith('.stan'):
        filename = filename[:-5]
    
    if not overwrite and os.path.isfile(filename+'.pkl'):
        # Use precompiled model
        with open(filename+'.pkl', 'rb') as f:
            sm = pickle.load(f)
    elif os.path.isfile(filename+'.stan'):
        # Compiling and save the model
        if not overwrite:
            print("Precompiled stan model {} not found.".format(filename+'.pkl'))
            print("Compiling and saving the model.")
        else:
            print("Compiling and saving the model {}.".format(filename+'.pkl'))
        if '/' in filename:
            model_name = filename.split('/')[-1]
        elif '\\' in filename:
            model_name = filename.split('\\')[-1]
        else:
            model_name = filename
        sm = StanModel(file=filename+'.stan', model_name=model_name)
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(sm, f)
        print("Compiling and saving done.")
    else:
        raise IOError("File {} or {} not found"
                      .format(filename+'.stan', filename+'.pkl'))
    return sm


def distribute_groups(J, K, Nj):
    """Distribute J groups to K sites.
    
    Parameters
    ----------
    J : int
        Number of groups
    
    K : int
        Number of sites
    
    Nj : ndarray or int
        Number of items in each group. Providing an integer corresponds to 
        constant number of items in each group.
    
    Returns
    -------
    Nk : ndarray
        Number of samples per site (shape: (K,) ).
    
    Nj_k or Nk_j : ndarray
        If K < J:  number of groups per site (shape (K,) )
           K == J: None
           K > J:  number of sites per group (shape (J,) )
    
    j_ind_k : ndarray
        Within site group indexes. Shape (N,), where N is the total nuber of
        samples, i.e. np.sum(Nj). Returned only if K < J, None otherwise.
    
    """
    # Check arguments
    if isinstance(Nj, int):
        Nj = Nj*np.ones(J, dtype=np.int64)
    elif len(Nj.shape) != 1 or Nj.shape[0] != J:
        raise ValueError("Invalid shape of arg. `Nj`.")
    if np.any(Nj <= 0):
        raise ValueError("Every group must have at least one item")
    N = Nj.sum()
    
    if K < 2:
        raise ValueError("K should be at least 2.")
    
    elif K < J:
        # ------ Many groups per site ------
        # Combine smallest pairs of consecutive groups until K has been reached
        j_lim = np.concatenate(([0], np.cumsum(Nj)))
        Nk = Nj.tolist()
        Njd = (Nj[:-1]+Nj[1:]).tolist()
        Nj_k = [1]*J
        for _ in range(J-K):
            ind = Njd.index(min(Njd))
            if ind+1 < len(Njd):
                Njd[ind+1] += Nk[ind]
            if ind > 0:
                Njd[ind-1] += Nk[ind+1]
            Nk[ind] = Njd[ind]
            Nk.pop(ind+1)
            Njd.pop(ind)
            Nj_k[ind] += Nj_k[ind+1]
            Nj_k.pop(ind+1)
        Nk = np.array(Nk)                       # Number of samples per site
        Nj_k = np.array(Nj_k)                   # Number of groups per site
        j_ind_k = np.empty(N, dtype=np.int32)   # Within site group index
        k_lim = np.concatenate(([0], np.cumsum(Nj_k)))
        for k in range(K):
            for ji in range(Nj_k[k]):
                ki = ji + k_lim[k]
                j_ind_k[j_lim[ki]:j_lim[ki+1]] = ji        
        return Nk, Nj_k, j_ind_k
    
    elif K == J:
        # ------ One group per site ------
        # Nothing to do here really
        return Nj, None, None
    
    elif K <= N:
        # ------ Multiple sites per group ------
        # Split biggest groups until enough sites are formed
        ppg = np.ones(J, dtype=np.int64)    # Parts per group
        Nj2 = Nj.astype(np.float)
        for _ in range(K-J):
            cur_max = Nj2.argmax()
            ppg[cur_max] += 1
            Nj2[cur_max] = Nj[cur_max]/ppg[cur_max]
        Nj2 = Nj//ppg
        rem = Nj%ppg
        # Form the number of samples for each site
        Nk = np.empty(K, dtype=np.int64)
        k = 0
        for j in range(J):
            for kj in range(ppg[j]):
                if kj < rem[j]:
                    Nk[k] = Nj2[j] + 1
                else:
                    Nk[k] = Nj2[j]
                k += 1
        return Nk, ppg, None
    
    else:
        raise ValueError("K cant be greater than number of samples")


# >>> Temp solution to suppres output from STAN model (remove when fixed)
# This part of the code is by jeremiahbuddha from:
# http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
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



