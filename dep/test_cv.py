
from __future__ import division
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal

from util import invert_normal_params, cv_moments

d = 3
S1 = np.array([[3.0,1.5,-1.0],[1.5,2.5,-0.2],[-1.0,-0.2,1.5]], order='F')
S2 = np.array([[3.1,1.2,-0.7],[1.2,2.4,-0.4],[-0.7,-0.4,1.8]], order='F')
m1 = np.array([1.0,0.0,-1.0])
m2 = np.array([1.1,0.0,-0.9])

Q2, r2 = invert_normal_params(S2, m2)
ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))

N1 = multivariate_normal(mean=m1, cov=S1)

n = 60
N = 8000

S_hats = np.empty((d,d,N), order='F')
m_hats = np.empty((d,N), order='F')
S_samps = np.empty((d,d,N), order='F')
m_samps = np.empty((d,N), order='F')

for i in xrange(N):
    
    # Get samples
    samp = N1.rvs(n)
    lp = N1.logpdf(samp)
    
    # cv estimates
    cv_moments(samp, lp, Q2, r2,
               S_tilde=S2, m_tilde=m2, ldet_Q_tilde=ldet_Q_tilde,
               S_hat=S_hats[:,:,i], m_hat=m_hats[:,i])
    S_hats[:,:,i] /= n-1
    
    # Basic sample estimates
    S_samps[:,:,i] = np.cov(samp, rowvar=0).T
    m_samps[:,i] = np.mean(samp, axis=0)
    
# Set precision to 3 display trailing zeroes
np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})

# Print
print '---- Mean ----'
print 'Real:'
print np.array2string(m1,precision=3)
print 'CV, expectation:'
print np.mean(m_hats, axis=-1)
#print 'CV, var:'
#print np.sqrt(np.var(m_hats, ddof=1, axis=-1))
print 'Sample, expectation:'
print np.mean(m_samps, axis=-1)
#print 'Sample, var:'
#print np.sqrt(np.var(m_samps, ddof=1, axis=-1))

print '---- Covariance ----'
print 'Real:'
print S1
print 'CV, expectation:'
print np.mean(S_hats, axis=-1)
#print 'CV, var:'
#print np.sqrt(np.var(S_hats, ddof=1, axis=-1))
print 'Sample, expectation:'
print np.mean(S_samps, axis=-1)
#print 'Sample, var:'
#print np.sqrt(np.var(S_samps, ddof=1, axis=-1))



