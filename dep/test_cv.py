
from __future__ import division
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from util import invert_normal_params, cv_moments
from cython_util import copy_triu_to_tril


np.random.seed(0)

def random_cov(d, diff=None):
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

rand_distr_every_iter = True
d = 3
indep_distr = False
diff = 0.1
regulate_a = None
max_a = None
multiple_cv = True
unnormalised_lp = False #np.log(0.4)

if not rand_distr_every_iter:
    # Predefined distributions
    #d = 3
    #S1 = np.array([[3.0,1.5,-1.0],[1.5,2.5,-0.2],[-1.0,-0.2,1.5]], order='F')
    #S2 = np.array([[3.1,1.2,-0.7],[1.2,2.4,-0.4],[-0.7,-0.4,1.8]], order='F')
    #m1 = np.array([1.0,0.0,-1.0])
    #m2 = np.array([1.1,0.0,-0.9])
    
    d = 2
    m1 = np.array([0.1, 1.1])
    m2 = np.array([-0.1, 1.0])
    S1 = np.array([[2.1,-1.0],[-1.0,1.5]])
    S2 = np.array([[2.0,-1.1],[-1.1,1.4]])
    
#    # Random
#    if indep_distr:
#        S1 = random_cov(d)
#        m1 = np.random.randn(d) + 0.8*np.random.randn()
#        S2 = random_cov(d)
#        m2 = np.random.randn(d) + 0.8*np.random.randn()
#    else:
#        S1, S2 = random_cov(d, diff=diff)
#        m1 = np.random.randn(d) + 0.6*np.random.randn()
#        m2 = m1 + diff*np.random.randn(d)
        
    
    N1 = multivariate_normal(mean=m1, cov=S1)
    
    # Convert S2,m2 to natural parameters
    Q2, r2 = invert_normal_params(S2, m2)
    ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))

# Sample sizes
n = 40         # Samples for one estimate
N = 1000       # Number of estimates

# Output arrays
d2 = (d*(d+1))/2
S_hats = np.empty((d,d,N), order='F')
m_hats = np.empty((d,N), order='F')
S_samps = np.empty((d,d,N), order='F')
m_samps = np.empty((d,N), order='F')
a_Ss = np.empty((d2,d2,N), order='F')
a_ms = np.empty((d,d,N), order='F')

if rand_distr_every_iter:
    m1s = np.empty((d,N), order='F')
    S1s = np.empty((d,d,N), order='F')
    m2s = np.empty((d,N), order='F')
    S2s = np.empty((d,d,N), order='F')
else:
    m1s = m1[:,np.newaxis]
    S1s = S1[:,:,np.newaxis]
    m2s = m2[:,np.newaxis]
    S2s = S2[:,:,np.newaxis]

# Sample estimates
for i in xrange(N):
    
    if rand_distr_every_iter:
        # Random
        if indep_distr:
            S1 = random_cov(d)
            m1 = np.random.randn(d) + 0.8*np.random.randn()
            S2 = random_cov(d)
            m2 = np.random.randn(d) + 0.8*np.random.randn()
        else:
            S1, S2 = random_cov(d, diff=diff)
            m1 = np.random.randn(d) + 0.6*np.random.randn()
            m2 = m1 + diff*np.random.randn(d)
        
        N1 = multivariate_normal(mean=m1, cov=S1)
        # Convert S2,m2 to natural parameters
        Q2, r2 = invert_normal_params(S2, m2)
        ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))
        
        m1s[:,i] = m1
        S1s[:,:,i] = S1
        m2s[:,i] = m2
        S2s[:,:,i] = S2
    
    if i == 2262:
        pass
    
    # Get samples
    samp = N1.rvs(n)
    lp = N1.logpdf(samp)
    if unnormalised_lp:
        lp += unnormalised_lp
    
    # cv estimates
    _, _, a_S, a_m = cv_moments(
        samp, lp, Q2, r2,
        S_tilde = S2,
        m_tilde = m2,
        ldet_Q_tilde = ldet_Q_tilde,
        regulate_a = regulate_a,
        max_a = max_a,
        multiple_cv = multiple_cv,
        S_hat = S_hats[:,:,i],
        m_hat = m_hats[:,i],
        ret_a = True
    )
    S_hats[:,:,i] /= n # Divided by n seems to give better results
    a_Ss[:,:,i] = a_S
    a_ms[:,:,i] = a_m
    
    # Basic sample estimates
    S_samps[:,:,i] = np.cov(samp, rowvar=0).T
    m_samps[:,i] = np.mean(samp, axis=0)


# Print
print '\nEstimate distributions'
print ('{:9}'+4*' {:>13}').format(
      'estimate', 'me (bias)', 'std', 'mse', '97.5se')
print 65*'-'
for i in xrange(d):
    print 'm[{}]'.format(i)    
    print ('{:9}'+4*' {:>13.5f}').format(
          '  _cv',
          np.mean(m_hats[i] - m1s[i]),
          np.sqrt(np.var(m_hats[i], ddof=1)),
          np.mean((m_hats[i] - m1s[i])**2),
          np.percentile((m_hats[i] - m1s[i])**2, 97.5))
    print ('{:9}'+4*' {:>13.5f}').format(
          '  _sample',
          np.mean(m_samps[i] - m1s[i]),
          np.sqrt(np.var(m_samps[i], ddof=1)),
          np.mean((m_samps[i] - m1s[i])**2),
          np.percentile((m_samps[i] - m1s[i])**2, 97.5))
for i in xrange(d):
    for j in xrange(i,d):
        print 'S[{},{}]'.format(i,j)
        print ('{:9}'+4*' {:>13.5f}').format(
              '  _cv',
              np.mean(S_hats[i,j] - S1s[i,j]),
              np.sqrt(np.var(S_hats[i,j], ddof=1)),
              np.mean((S_hats[i,j] - S1s[i,j])**2),
              np.percentile((S_hats[i,j] - S1s[i,j])**2, 97.5))
        print ('{:9}'+4*' {:>13.5f}').format(
              '  _sample',
              np.mean(S_samps[i,j] - S1s[i,j]),
              np.sqrt(np.var(S_samps[i,j], ddof=1)),
              np.mean((S_samps[i,j] - S1s[i,j])**2),
              np.percentile((S_samps[i,j] - S1s[i,j])**2, 97.5))

# Plot squared error vs a
#plt.figure()
#plt.scatter(a_ms, m_hats - m1s, alpha=0.3)
#plt.show()

#mse_samps = (m_samps - m1s)**2
#mse_hats = (m_hats - m1s)**2


# Plot hist of a[0]
#plt.figure()
#plt.hist(a_ms[0,:], bins=20)
#plt.figure()
#plt.hist(a_Ss[0,0,:], bins=20)
#plt.show()




