import os, sys

import numpy as np
from scipy import linalg, stats
from scipy.linalg import cho_factor, cho_solve

import matplotlib.pyplot as plt

# Add parent dir to sys.path if not present already. This is only done because
# of easy importing of the package epstan. Adding the parent directory into the
# PYTHONPATH works as well.
# CUR_PATH = "/u/73/tsivula/unix/aalto/ep-stan/experiment"
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_PATH = os.path.abspath(os.path.join(CUR_PATH, os.pardir))
RES_PATH = os.path.join(CUR_PATH, 'results')
MOD_PATH = os.path.join(CUR_PATH, 'models')
# Double check that the package is in the parent directory
if os.path.exists(os.path.join(PARENT_PATH, 'epstan')):
    if PARENT_PATH not in os.sys.path:
        os.sys.path.insert(0, PARENT_PATH)

import fit
from epstan.util import invert_normal_params


model_name = 'm4b'

# Load target file
target_file = np.load(
    os.path.join(RES_PATH, 'target_{}.npz'.format(model_name)))
m_target = target_file['m_target']
S_target = target_file['S_target']
conf = target_file['conf'][()]
target_file.close()
# Load target samples if found
target_samp_file_path = os.path.join(
    RES_PATH, 'target_samp_{}.npz'.format(model_name))
if os.path.exists(target_samp_file_path):
    target_samp_file = np.load(target_samp_file_path)
    samp_target = target_samp_file['samp_target']
    target_samp_file.close()
else:
    samp_target = None

J = conf['J']
D = conf['D']
K = conf['J']
chains = 8
siter = 400

iters = 5
damp_n = 20


def kl_mvn(m0, S0, m1, S1, sum_log_diag_cho_S0=None):
    """Calculate KL-divergence for multiv normal distributions

    Calculates KL(p||q), where p ~ N(m0,S0) and q ~ N(m1,S1). Optional argument
    sum_log_diag_cho_S0 is precomputed sum(log(diag(cho(S0))).

    """
    choS1 = cho_factor(S1)
    if sum_log_diag_cho_S0 is None:
        sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(S0)[0])))
    dm = m1-m0
    KL_div = (
        0.5*(
            np.trace(cho_solve(choS1, S0))
            + dm.dot(cho_solve(choS1, dm))
            - len(m0)
        )
        - sum_log_diag_cho_S0 + np.sum(np.log(np.diag(choS1[0])))
    )
    return KL_div

conf = fit.configurations(J=J, D=D, K=K, chains=chains, siter=siter)
master = fit.main(model_name, conf, ret_master=True)

sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(S_target)[0])))



# Localise some instance variables
# Mean and cov of the posterior approximation
S = master.S
m = master.m
# Natural parameters of the approximation
Q = master.Q
r = master.r
# Natural site parameters
Qi = master.Qi
ri = master.ri
# Natural site proposal parameters
Qi2 = master.Qi2
ri2 = master.ri2
# Site parameter updates
dQi = master.dQi
dri = master.dri


posdefs = np.zeros(master.K, dtype=bool)

damps = np.linspace(0, 1, damp_n+2)[1:-1]
mses = np.full((iters, damp_n), np.nan)
lls = np.full((iters, damp_n), np.nan)
kls = np.full((iters, damp_n), np.nan)
damps_selected = np.full(iters, np.nan)
mses_selected = np.full(iters, np.nan)
lls_selected = np.full(iters, np.nan)
kls_selected = np.full(iters, np.nan)

# iters
for iter_ind in range(iters):
    print("Iteration {}/{}".format(iter_ind+1, iters))

    for k, worker in enumerate(master.workers):
        print("    Tilted for site {}/{}".format(k+1, master.K))
        sys.stdout.flush()
        posdefs[k] = worker.tilted(dQi[:,:,k], dri[:,k])
    if not np.all(posdefs):
        print("    Tilted fails at {}".format(k+1))
        break

    for di, df in enumerate(damps):
        # apply damp
        np.add(Qi, np.multiply(df, dQi, out=Qi2), out=Qi2)
        np.add(ri, np.multiply(df, dri, out=ri2), out=ri2)
        np.add(Qi2.sum(2, out=Q), master.Q0, out=Q)
        np.add(ri2.sum(1, out=r), master.r0, out=r)

        try:
            cho_Q = S
            np.copyto(cho_Q, Q)
            linalg.cho_factor(cho_Q, overwrite_a=True)
            invert_normal_params(
                cho_Q, r, out_A='in-place', out_b=m, cho_form=True)

            # cavity
            for k, worker in enumerate(master.workers):
                posdefs[k] = worker.cavity(Q, r, Qi2[:,:,k], ri2[:,k])

            if np.all(posdefs):
                # selection criteria
                # mse
                mses[iter_ind, di] = np.mean((m - m_target)**2)
                # likelihood
                lls[iter_ind, di] = np.sum(stats.multivariate_normal.logpdf(
                    samp_target, mean=m, cov=S.T))
                # approximate KL
                kls[iter_ind, di] = kl_mvn(
                    m_target, S_target, m, S.T, sum_log_diag_cho_S0)

        except linalg.LinAlgError:
            pass

    # select the best
    # best_idx = np.nanargmax(lls[iter_ind])
    best_idx = np.nanargmin(kls[iter_ind])
    df = damps[best_idx]
    damps_selected[iter_ind] = df
    mses_selected[iter_ind] = mses[iter_ind, best_idx]
    lls_selected[iter_ind] = lls[iter_ind, best_idx]
    kls_selected[iter_ind] = kls[iter_ind, best_idx]
    # apply damp
    np.add(Qi, np.multiply(df, dQi, out=Qi2), out=Qi2)
    np.add(ri, np.multiply(df, dri, out=ri2), out=ri2)
    np.add(Qi2.sum(2, out=Q), master.Q0, out=Q)
    np.add(ri2.sum(1, out=r), master.r0, out=r)
    for k, worker in enumerate(master.workers):
        worker.cavity(Q, r, Qi2[:,:,k], ri2[:,k])
    # switch Qi <> Qi2, ri <> ri2
    temp = Qi
    Qi = Qi2
    Qi2 = temp
    temp = ri
    ri = ri2
    ri2 = temp

# save
np.savez(
    os.path.join(RES_PATH, 'find_damp.npz'),
    damps = damps,
    mses = mses,
    lls = lls,
    kls = kls,
    damps_selected = damps_selected,
    mses_selected = mses_selected,
    lls_selected = lls_selected,
    kls_selected = kls_selected,
)

## load
# res_file = np.load('find_damp.npz')
# damps = res_file['damps']
# mses = res_file['mses']
# lls = res_file['lls']
# kls = res_file['kls']
# damps_selected = res_file['damps_selected']
# lls_selected = res_file['lls_selected']
# mses_selected = res_file['mses_selected']
# kls_selected = res_file['kls_selected']
# res_file.close()
# iters, damp_n = kls.shape


## plot
# plt.figure()
# plt.plot(damps_selected)
# plt.title('damps')
#
# plt.figure()
# plt.plot(mses_selected)
# plt.title('mses')
#
# plt.figure()
# plt.plot(lls_selected)
# plt.title('lls')
#
# plt.figure()
# plt.plot(kls_selected)
# plt.title('kls')
#
# fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
# for i, ax in enumerate(axes):
#     ax.plot(damps, mses[i], label=str(i+1))
# fig.legend()
# fig.suptitle('mses')
#
# fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
# for i, ax in enumerate(axes):
#     ax.plot(damps, lls[i], label=str(i+1))
# fig.legend()
# fig.suptitle('lls')
#
# fig, axes = plt.subplots(1, iters, sharex=True, sharey=True)
# for i, ax in enumerate(axes):
#     ax.plot(damps, kls[i], label=str(i+1))
# fig.legend()
# fig.suptitle('kls')
