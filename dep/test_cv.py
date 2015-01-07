
from __future__ import division
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from util import invert_normal_params, cv_moments

np.random.seed(0)

rand_distr_every_iter = True
d = 3
indep_distr = False
diff = 1
regulate_a = 1

if not rand_distr_every_iter:
    # Predefined distributions
    #d = 3
    #S1 = np.array([[3.0,1.5,-1.0],[1.5,2.5,-0.2],[-1.0,-0.2,1.5]], order='F')
    #S2 = np.array([[3.1,1.2,-0.7],[1.2,2.4,-0.4],[-0.7,-0.4,1.8]], order='F')
    #m1 = np.array([1.0,0.0,-1.0])
    #m2 = np.array([1.1,0.0,-0.9])
    
#    d = 2
#    m1 = np.array([0.1, 1.1])
#    m2 = 100*np.array([-0.1, 1.0])
#    S1 = np.array([[2.1,-1.0],[-1.0,1.5]])
#    S2 = np.array([[2.0,-1.1],[-1.1,1.4]])
    
    # Random
    if indep_distr:
        S1 = 2*np.random.rand(d,d)-1
        S1 = (S1.dot(S1.T)+d*np.eye(d)).T
        m1 = np.random.randn(d)+np.random.randn()
        S2 = 2*np.random.rand(d,d)-1
        S2 = (S2.dot(S2.T)+d*np.eye(d)).T
        m2 = np.random.randn(d)+np.random.randn()
    else:
        S1 = 2*np.random.rand(d,d)-1
        S1 = (S1.dot(S1.T)+(d+2*diff)*np.eye(d)).T
        m1 = np.random.randn(d)+np.random.randn()
        S2 = S1 + (2*np.random.rand(d,d) -1)*diff
        m2 = m1 + (2*np.random.rand(d) -1)*diff
    
    N1 = multivariate_normal(mean=m1, cov=S1)
    
    # Convert S2,m2 to natural parameters
    Q2, r2 = invert_normal_params(S2, m2)
    ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))

# Sample sizes
n = 40         # Samples for one estimate
N = 10000       # Number of estimates

# Output arrays
S_hats = np.empty((d,d,N), order='F')
m_hats = np.empty((d,N), order='F')
S_samps = np.empty((d,d,N), order='F')
m_samps = np.empty((d,N), order='F')
a_Ss = np.empty((d,d,N), order='F')
a_ms = np.empty((d,N), order='F')
var_h_Ss = np.empty((d,d,N), order='F')
var_h_ms = np.empty((d,N), order='F')

if rand_distr_every_iter:
    m1s = np.empty((d,N), order='F')
    S1s = np.empty((d,d,N), order='F')
else:
    m1s = m1
    S1s = S1

# Sample estimates
for i in xrange(N):
    
    if rand_distr_every_iter:
        # Random
        if indep_distr:
            S1 = 2*np.random.rand(d,d)-1
            S1 = (S1.dot(S1.T)+d*np.eye(d)).T
            m1 = np.random.randn(d)+np.random.randn()
            S2 = 2*np.random.rand(d,d)-1
            S2 = (S2.dot(S2.T)+d*np.eye(d)).T
            m2 = np.random.randn(d)+np.random.randn()
        else:
            S1 = 2*np.random.rand(d,d)-1
            S1 = (S1.dot(S1.T)+(d+2*diff)*np.eye(d)).T
            m1 = np.random.randn(d)+np.random.randn()
            S2 = S1 + (2*np.random.rand(d,d) -1)*diff
            m2 = m1 + (2*np.random.rand(d) -1)*diff
        
        N1 = multivariate_normal(mean=m1, cov=S1)
        # Convert S2,m2 to natural parameters
        Q2, r2 = invert_normal_params(S2, m2)
        ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))
        
        m1s[:,i] = m1
        S1s[:,:,i] = S1
    
    if i == 2262:
        pass
    
    # Get samples
    samp = N1.rvs(n)
    lp = N1.logpdf(samp)
    
    # cv estimates
    _, _, a_S, a_m, var_h_S, var_h_m = cv_moments(
        samp, lp, Q2, r2,
        S_tilde=S2, m_tilde=m2,
        ldet_Q_tilde=ldet_Q_tilde, regulate_a=regulate_a,
        S_hat=S_hats[:,:,i], m_hat=m_hats[:,i]
    )
    a_Ss[:,:,i] = a_S
    a_ms[:,i] = a_m
    var_h_Ss[:,:,i] = var_h_S
    var_h_ms[:,i] = var_h_m
    
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
#plt.scatter(a_ms, ((m_hats - m1s)**2)/((m_samps - m1s)**2))
#plt.figure()
#plt.scatter(var_h_ms, ((m_hats - m1s)**2)/((m_samps - m1s)**2))
#plt.show()





mse_samps = (m_samps - m1s)**2
mse_hats = (m_hats - m1s)**2
# Bad inds
bi = (np.array([0, 0, 1, 1, 1, 2]), np.array([  13, 2262, 5989, 6446, 7594, 9711]))
# Worst relative
wr = 2262
#wors
w = 11743


# Plot hist of a[0]
#plt.figure()
#plt.hist(a_ms[0,:], bins=20)
#plt.figure()
#plt.hist(a_Ss[0,0,:], bins=20)
#plt.show()




