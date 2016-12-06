"""Sckript for testing olse method for estimating the precision matrix,
see util.olse.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
# import matplotlib.pyplot as plt

from .util import invert_normal_params, olse
from cython_util import copy_triu_to_tril


# ------------------------------------------------------------------------------
#     Configurations
# ------------------------------------------------------------------------------
np.random.seed(0)               # Seed
n = 100                         # Samples for one estimate
N = 5000                        # Number of estimates
d = 20                          # Dimension of the test distributions
rand_distr_every_iter = True    # Separate distribution for each estimate
indep_distr = False             # Target and prior distributions are independent
diff = 0.1                      # Difference between target and prior
oracle = False                  # Indicator if oracle or bona fide olse
                                # estimator is used
print_max = 4                   # Number of coefficients printed

# ------------------------------------------------------------------------------
#     Pre-defined distributions
# ------------------------------------------------------------------------------
use_pre_defined = False         # In order to use the following, set to True
S1 = np.array([[3.0,1.5,-1.0],[1.5,2.5,-0.2],[-1.0,-0.2,1.5]], order='F')
S2 = np.array([[3.1,1.2,-0.7],[1.2,2.4,-0.4],[-0.7,-0.4,1.8]], order='F')
m1 = np.array([1.0,0.0,-1.0])
m2 = np.array([1.1,0.0,-0.9])


def random_cov(d, diff=None):
    """Generate random covariance matrix.
    
    Generates a random covariance matrix, or two dependent covariance matrices
    if the argument `diff` is given.
    
    """
    S = 0.8*np.random.randn(d,d)
    copy_triu_to_tril(S)
    np.fill_diagonal(S,0)
    mineig = linalg.eigvalsh(S, eigvals=(0,0))[0]
    drand = 0.8*np.random.randn(d)
    if mineig < 0:
        S += np.diag(np.exp(drand)-mineig)
    else:
        S += np.diag(np.exp(drand))
    if not diff:
        return S.T
    S2 = S * np.random.randint(2, size=(d,d))*np.exp(diff*np.random.randn(d,d))
    copy_triu_to_tril(S2)
    np.fill_diagonal(S2,0)
    mineig = linalg.eigvalsh(S2, eigvals=(0,0))[0]
    drand += diff*np.random.randn(d)
    if mineig < 0:
        S2 += np.diag(np.exp(drand)-mineig)
    else:
        S2 += np.diag(np.exp(drand))
    return S.T, S2.T

# Preprocess
if use_pre_defined:
    rand_distr_every_iter = False
    d = m1.shape[0]
if not rand_distr_every_iter:
    if not use_pre_defined:
        # Generate random distr
        if oracle:
            S1 = random_cov(d)
            m1 = np.random.randn(d) + 0.8*np.random.randn()
        else:
            if indep_distr:
                S1 = random_cov(d)
                m1 = np.random.randn(d) + 0.8*np.random.randn()
                S2 = random_cov(d)
                m2 = np.random.randn(d) + 0.8*np.random.randn()
            else:
                S1, S2 = random_cov(d, diff=diff)
                m1 = np.random.randn(d) + 0.6*np.random.randn()
                m2 = m1 + diff*np.random.randn(d)
    # Convert to natural parameters
    N1 = multivariate_normal(mean=m1, cov=S1)
    Q1, r1 = invert_normal_params(S1, m1)
    if not oracle:
        Q2, r2 = invert_normal_params(S2, m2, out_A='in-place',out_b='in-place')

# Output arrays
Q_hats = np.empty((d,d,N), order='F')
r_hats = np.empty((d,N), order='F')
Q_samps = np.empty((d,d,N), order='F')
r_samps = np.empty((d,N), order='F')

if rand_distr_every_iter:
    r1s = np.empty((d,N), order='F')
    Q1s = np.empty((d,d,N), order='F')
    if not oracle:
        r2s = np.empty((d,N), order='F')
        Q2s = np.empty((d,d,N), order='F')
else:
    r1s = m1[:,np.newaxis]
    Q1s = S1[:,:,np.newaxis]
    if not oracle:
        r2s = m2[:,np.newaxis]
        Q2s = S2[:,:,np.newaxis]

# Sample estimates
for i in range(N):
    
    if rand_distr_every_iter:
        # Generate random distr
        if oracle:
            S1 = random_cov(d)
            m1 = np.random.randn(d) + 0.8*np.random.randn()
        else:
            if indep_distr:
                S1 = random_cov(d)
                m1 = np.random.randn(d) + 0.8*np.random.randn()
                S2 = random_cov(d)
                m2 = np.random.randn(d) + 0.8*np.random.randn()
            else:
                S1, S2 = random_cov(d, diff=diff)
                m1 = np.random.randn(d) + 0.6*np.random.randn()
                m2 = m1 + diff*np.random.randn(d)
        # Convert to natural parameters
        N1 = multivariate_normal(mean=m1, cov=S1)
        Q1, r1 = invert_normal_params(S1, m1)
        r1s[:,i] = r1
        Q1s[:,:,i] = Q1
        if not oracle:
            Q2, r2 = invert_normal_params(S2, m2, out_A='in-place',out_b='in-place')
            r2s[:,i] = r2
            Q2s[:,:,i] = Q2
    
    # Get samples
    samp = N1.rvs(n)
    mt = np.mean(samp, axis=0)
    samp -= mt
    St = np.dot(samp.T, samp).T
    
    # olse
    olse(St/n, n, P = Q1 if oracle else Q2, out = Q_hats[:,:,i])
    r_hats[:,i] = Q_hats[:,:,i].dot(mt)
    
    # Basic sample estimates
    St /= n-1
    invert_normal_params(St, mt, out_A=Q_samps[:,:,i], out_b=r_samps[:,i])
    # Unbiased natural parameter estimates
    unbias_k = (n-d-2)/(n-1)
    Q_samps[:,:,i] *= unbias_k
    r_samps[:,i] *= unbias_k

# Print result statistics
print('Statistics of {} estimates'.format(N))
print(('{:9}'+4*' {:>13}').format(
      'estimate', 'me (bias)', 'std', 'mse', '97.5se'))
print(65*'-')
for i in range(min(d,print_max)):
    print('r[{}]'.format(i))    
    print(('{:9}'+4*' {:>13.5f}').format(
          '  _olse',
          np.mean(r_hats[i] - r1s[i]),
          np.sqrt(np.var(r_hats[i], ddof=1)),
          np.mean((r_hats[i] - r1s[i])**2),
          np.percentile((r_hats[i] - r1s[i])**2, 97.5)))
    print(('{:9}'+4*' {:>13.5f}').format(
          '  _sample',
          np.mean(r_samps[i] - r1s[i]),
          np.sqrt(np.var(r_samps[i], ddof=1)),
          np.mean((r_samps[i] - r1s[i])**2),
          np.percentile((r_samps[i] - r1s[i])**2, 97.5)))
for i in range(min(d,print_max)):
    for j in range(i,min(d,print_max)):
        print('Q[{},{}]'.format(i,j))
        print(('{:9}'+4*' {:>13.5f}').format(
              '  _olse',
              np.mean(Q_hats[i,j] - Q1s[i,j]),
              np.sqrt(np.var(Q_hats[i,j], ddof=1)),
              np.mean((Q_hats[i,j] - Q1s[i,j])**2),
              np.percentile((Q_hats[i,j] - Q1s[i,j])**2, 97.5)))
        print(('{:9}'+4*' {:>13.5f}').format(
              '  _sample',
              np.mean(Q_samps[i,j] - Q1s[i,j]),
              np.sqrt(np.var(Q_samps[i,j], ddof=1)),
              np.mean((Q_samps[i,j] - Q1s[i,j])**2),
              np.percentile((Q_samps[i,j] - Q1s[i,j])**2, 97.5)))



