
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

#d = 2
#m1 = np.array([0.1, 1.1])
#m2 = np.array([-0.1, 1.0])
#S1 = np.array([[2.1,-1.0],[-1.0,1.5]])
#S2 = np.array([[2.0,-1.1],[-1.1,1.4]])


Q2, r2 = invert_normal_params(S2, m2)
ldet_Q_tilde = np.sum(np.log(np.diag(linalg.cho_factor(Q2)[0])))

N1 = multivariate_normal(mean=m1, cov=S1)

n = 60
N = 2000

S_hats = np.empty((d,d,N), order='F')
m_hats = np.empty((d,N), order='F')
S_samps = np.empty((d,d,N), order='F')
m_samps = np.empty((d,N), order='F')

for i in xrange(N):
    
    # Get samples
    samp = N1.rvs(n)
    lp = N1.logpdf(samp)
    
    # cv estimates
    _, _, a_cov = cv_moments(samp, lp, Q2, r2,
               S_tilde=S2, m_tilde=m2, ldet_Q_tilde=ldet_Q_tilde,
               S_hat=S_hats[:,:,i], m_hat=m_hats[:,i])
    S_hats[:,:,i] /= n
    S_hats[:,:,i] += a_cov*S2
    
    # Basic sample estimates
    S_samps[:,:,i] = np.cov(samp, rowvar=0).T
    m_samps[:,i] = np.mean(samp, axis=0)


# Print
print '-'*77
print '{:8} {:>8} |{:^28}|{:^28}|'.format(
      'variable', 'real', 'cv', 'sample')
print '{:8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} |'.format(
      '', '', 'mean', 'std', 'mse', 'mean', 'sdt', 'mse')
print '-'*77
for i in xrange(d):
    print ('{:8} {:>8.2f} |'+3*' {:>8.3f}'+' |'+3*' {:>8.3f}'+' |') \
        .format(
            'm[{}]'.format(i),
            m1[i],
            np.mean(m_hats[i]),
            np.sqrt(np.var(m_hats[i], ddof=1)),
            np.mean((m_hats[i]-m1[i])**2),
            np.mean(m_samps[i]),
            np.sqrt(np.var(m_samps[i], ddof=1)),
            np.mean((m_samps[i]-m1[i])**2)
        )
for i in xrange(d):
    for j in xrange(i,d):
        print ('{:8} {:>8.2f} |'+3*' {:>8.3f}'+' |'+3*' {:>8.3f}'+' |') \
            .format(
                'S[{},{}]'.format(i,j),
                S1[i,j],
                np.mean(S_hats[i,j]),
                np.sqrt(np.var(S_hats[i,j], ddof=1)),
                np.mean((S_hats[i,j]-S1[i,j])**2),
                np.mean(S_samps[i,j]),
                np.sqrt(np.var(S_samps[i,j], ddof=1)),
                np.mean((S_samps[i,j]-S1[i,j])**2)
            )



