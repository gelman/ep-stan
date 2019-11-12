"""Plot script for the paper."""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


MODEL_NAME = 'm4b'
KS = (2, 4, 8, 16)
# KS = (4, 8, 16)
EP_ID = 'd50'
SNEP_ID = ''
CONS_ID = ''
FULL_N = 11



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

# -------------
#   load data
# -------------

# Load target file
target_file = np.load('./results/target_{}.npz'.format(MODEL_NAME))
m_target = target_file['m_target']
S_target = target_file['S_target']
target_file.close()

# Load EP result file
m_s_ep_s = []
S_s_ep_s = []
time_s_ep_s = []
mstepsize_s_ep_s = []
mrhat_s_ep_s = []
for k in KS:
    if EP_ID:
        res_d_file = np.load(
            './results/res_d_{}_{}_K{}.npz'
            .format(MODEL_NAME, EP_ID, k)
        )
    else:
        res_d_file = np.load(
            './results/res_d_{}_K{}.npz'
            .format(MODEL_NAME, k)
        )
    m_s_ep_s.append(res_d_file['m_s_ep'])
    S_s_ep_s.append(res_d_file['S_s_ep'])
    time_s_ep_s.append(res_d_file['time_s_ep'])
    mstepsize_s_ep_s.append(res_d_file['mstepsize_s_ep'])
    mrhat_s_ep_s.append(res_d_file['mrhat_s_ep'])
    res_d_file.close()

# Load SNEP result file
m_s_snep_s = []
S_s_snep_s = []
time_s_snep_s = []
mstepsize_s_snep_s = []
mrhat_s_snep_s = []
snep_last_iter_s = []
for k in KS:
    if SNEP_ID:
        res_s_file = np.load(
            './results/res_s_{}_{}_K{}.npz'
            .format(MODEL_NAME, SNEP_ID, k)
        )
    else:
        res_s_file = np.load(
            './results/res_s_{}_K{}.npz'
            .format(MODEL_NAME, k)
        )
    m_s_snep_s.append(res_s_file['m_s_snep'])
    S_s_snep_s.append(res_s_file['S_s_snep'])
    time_s_snep_s.append(res_s_file['time_s_snep'])
    mstepsize_s_snep_s.append(res_s_file['mstepsize_s_snep'])
    mrhat_s_snep_s.append(res_s_file['mrhat_s_snep'])
    if 'last_iter' in res_s_file.files:
        snep_last_iter = res_s_file['last_iter'][()]
        m_s_snep_s[-1][snep_last_iter:,:] = np.nan
        S_s_snep_s[-1][snep_last_iter:,:,:] = np.nan
        time_s_snep_s[-1][snep_last_iter:] = np.nan
        mstepsize_s_snep_s[-1][snep_last_iter:] = np.nan
        mrhat_s_snep_s[-1][snep_last_iter:] = np.nan
    else:
        snep_last_iter = len(time_s_snep_s[-1])
    snep_last_iter_s.append(snep_last_iter)
    res_s_file.close()


# Load CONS result file
m_s_cons_s = []
S_s_cons_s = []
time_s_cons_s = []
mstepsize_s_cons_s = []
mrhat_s_cons_s = []
for k in KS:
    if CONS_ID:
        res_c_file = np.load(
            './results/res_c_{}_{}_K{}.npz'
            .format(MODEL_NAME, EP_ID, k)
        )
    else:
        res_c_file = np.load(
            './results/res_c_{}_K{}.npz'
            .format(MODEL_NAME, k)
        )
    m_s_cons_s.append(res_c_file['m_s_cons'])
    S_s_cons_s.append(res_c_file['S_s_cons'])
    time_s_cons_s.append(res_c_file['time_s_cons'])
    mstepsize_s_cons_s.append(res_c_file['mstepsize_s_cons'])
    mrhat_s_cons_s.append(res_c_file['mrhat_s_cons'])
    res_c_file.close()

# Load full result file
res_f_file = np.load('./results/res_f_{}.npz'.format(MODEL_NAME))
m_s_full = res_f_file['m_s_full']
S_s_full = res_f_file['S_s_full']
time_s_full = res_f_file['time_s_full']
mstepsize_s_full = res_f_file['mstepsize_s_full']
mrhat_s_full = res_f_file['mrhat_s_full']
res_f_file.close()


# -------------------
#   Calc MSE and KL
# -------------------

dphi = m_target.shape[0]
sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(S_target)[0])))

# EP
mse_ep_s = []
kl_ep_s = []
for m_s_ep, S_s_ep in zip(m_s_ep_s, S_s_ep_s):
    mse_ep = np.mean((m_s_ep - m_target)**2, axis=1)
    kl_ep = np.empty(len(m_s_ep))
    for i in range(len(m_s_ep)):
        kl_ep[i] = kl_mvn(
            m_target, S_target, m_s_ep[i], S_s_ep[i], sum_log_diag_cho_S0)
    mse_ep_s.append(mse_ep)
    kl_ep_s.append(kl_ep)

# SNEP
mse_snep_s = []
kl_snep_s = []
for m_s_snep, S_s_snep, snep_last_iter in zip(
        m_s_snep_s, S_s_snep_s, snep_last_iter_s):
    mse_snep = np.mean((m_s_snep - m_target)**2, axis=1)
    kl_snep = np.full(len(m_s_snep), np.nan)
    for i in range(snep_last_iter):
        kl_snep[i] = kl_mvn(
            m_target, S_target, m_s_snep[i], S_s_snep[i], sum_log_diag_cho_S0)
    mse_snep_s.append(mse_snep)
    kl_snep_s.append(kl_snep)

# consensus
mse_cons_s = []
kl_cons_s = []
for m_s_cons, S_s_cons in zip(m_s_cons_s, S_s_cons_s):
    mse_cons = np.mean((m_s_cons - m_target)**2, axis=1)
    kl_cons = np.empty(len(m_s_cons))
    for i in range(len(m_s_cons)):
        kl_cons[i] = kl_mvn(
            m_target, S_target, m_s_cons[i], S_s_cons[i], sum_log_diag_cho_S0)
    mse_cons_s.append(mse_cons)
    kl_cons_s.append(kl_cons)

# full
mse_full = np.mean((m_s_full - m_target)**2, axis=1)
kl_full = np.full(len(m_s_full), np.nan)
for i in range(len(m_s_full)):
    kl_full[i] = kl_mvn(
        m_target, S_target, m_s_full[i], S_s_full[i], sum_log_diag_cho_S0)


# ---------
#   plots
# ---------

# get colors for each K
K_colors = plt.get_cmap('tab10').colors[:len(KS)]




# # ----------- just snep
# plt.figure()
# for mse_snep, time_s_snep, k, color in zip(
#         mse_snep_s, time_s_snep_s, KS, K_colors):
#     plt.plot(time_s_snep/60, mse_snep, color=color, label=str(k))
# plt.legend()
#
#
# # ----------- one snep vs one ep
# fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
# cur_k_i = 2
#
# ax = axes[0]
# ax.set_title('MSE')
# ax.semilogy(time_s_ep_s[cur_k_i]/60, mse_ep_s[cur_k_i], label='ep')
# ax.semilogy(time_s_snep_s[cur_k_i]/60, mse_snep_s[cur_k_i], label='snep')
#
# ax = axes[1]
# ax.set_title('KL')
# ax.semilogy(time_s_ep_s[cur_k_i]/60, kl_ep_s[cur_k_i], label='ep')
# ax.semilogy(time_s_snep_s[cur_k_i]/60, kl_snep_s[cur_k_i], label='snep')
# ax.legend()

# ----------- all snep vs ep
last_iters_prc = 0.1

fig, axes = plt.subplots(2, len(KS), sharex='col', sharey='row', figsize=(8, 5))

for cur_k_i in range(len(KS)):
    ax = axes[0, cur_k_i]
    ax.semilogy(
        time_s_ep_s[cur_k_i]/60, mse_ep_s[cur_k_i],
        color='C0', label='moment matching'
    )
    ax.semilogy(
        time_s_snep_s[cur_k_i]/60, mse_snep_s[cur_k_i],
        color='C1', label='SNEP'
    )
    # last_iters_n = int(last_iters_prc*len(mse_ep_s[cur_k_i]))
    # ax.axhline(
    #     np.mean(mse_ep_s[cur_k_i][-last_iters_n:]),
    #     color='C0', ls='--'
    # )
    # last_iters_n = int(last_iters_prc*len(mse_snep_s[cur_k_i]))
    # ax.axhline(
    #     np.mean(mse_snep_s[cur_k_i][-last_iters_n:]),
    #     color='C1', ls='--'
    # )

    ax = axes[1, cur_k_i]
    ax.semilogy(
        time_s_ep_s[cur_k_i]/60, kl_ep_s[cur_k_i],
        color='C0', label='moment matching'
    )
    ax.semilogy(
        time_s_snep_s[cur_k_i]/60, kl_snep_s[cur_k_i],
        color='C1', label='SNEP'
    )
    # last_iters_n = int(last_iters_prc*len(kl_ep_s[cur_k_i]))
    # ax.axhline(
    #     np.mean(kl_ep_s[cur_k_i][-last_iters_n:]),
    #     color='C0', ls='--'
    # )
    # last_iters_n = int(last_iters_prc*len(kl_snep_s[cur_k_i]))
    # ax.axhline(
    #     np.mean(kl_snep_s[cur_k_i][-last_iters_n:]),
    #     color='C1', ls='--'
    # )

for cur_k_i in range(len(KS)):
    axes[0, cur_k_i].set_title('K = {}'.format(KS[cur_k_i]))
axes[0, 0].set_ylabel('MSE')
axes[1, 0].set_ylabel('KL')
for cur_k_i in range(len(KS)):
    axes[-1, cur_k_i].set_xlabel('time (min)')
axes[-1,1].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25)
)
plt.tight_layout()




# ----------- snep end vs end ep
end_mses_snep = np.array([mse_snep[-1] for mse_snep in mse_snep_s])
end_kls_snep = np.array([kl_snep[-1] for kl_snep in kl_snep_s])
end_mses_ep = np.array([mse_ep[-1] for mse_ep in mse_ep_s])
end_kls_ep = np.array([kl_ep[-1] for kl_ep in kl_ep_s])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax = axes[0]
ax.set_title('MSE')
ax.plot(KS, end_mses_ep, label='ep')
ax.plot(KS, end_mses_snep, label='snep')

ax = axes[1]
ax.set_title('KL')
ax.plot(KS, end_kls_ep, label='ep')
ax.plot(KS, end_kls_snep, label='snep')
ax.legend()



# --------- time as x-axis
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

lw = 1.0

# MSE
ax = axes[0]
ax.set_yscale('log')
# full
ax.plot(time_s_full[:FULL_N]/60, mse_full[:FULL_N], 'k', label='full', lw=lw)
# ep
for mse_ep, time_s_ep, k, color in zip(mse_snep_s, time_s_snep_s, KS, K_colors):
    ax.plot(time_s_ep[1:]/60, mse_ep[1:], color=color, label=str(k), lw=lw)
# snep
for mse_snep, time_s_snep, k, color in zip(
        mse_snep_s, time_s_snep_s, KS, K_colors):
    ax.plot(time_s_snep/60, mse_snep, color=color, ls=':', label=str(k), lw=lw)
# prior label
ax.axhline(
    mse_ep_s[0][0], lw=plt.rcParams['grid.linewidth'], color='0.65', zorder=1)
ax.text(
    x = -0.022,
    y = mse_ep_s[0][0],
    s = 'prior',
    transform = ax.get_yaxis_transform(),
    ha = 'right',
    va = 'center'
)
# cosmetics
ax.minorticks_off()
# ax.set_xlabel('time (min)')
ax.set_ylabel('MSE')

# KL
ax = axes[1]
ax.set_yscale('log')
# full
ax.plot(time_s_full[:FULL_N]/60, kl_full[:FULL_N], 'k', label='full', lw=lw)
# ep
for kl_ep, time_s_ep, k, color in zip(kl_snep_s, time_s_snep_s, KS, K_colors):
    ax.plot(time_s_ep[1:]/60, kl_ep[1:], color=color, label=str(k), lw=lw)
# snep
for kl_snep, time_s_snep, k, color in zip(
        kl_snep_s, time_s_snep_s, KS, K_colors):
    ax.plot(time_s_snep/60, kl_snep, color=color, ls=':', label=str(k), lw=lw)
# prior label
ax.axhline(
    kl_ep_s[0][0], lw=plt.rcParams['grid.linewidth'], color='0.65', zorder=1)
ax.text(
    x = -0.022,
    y = kl_ep_s[0][0],
    s = 'prior',
    transform = ax.get_yaxis_transform(),
    ha = 'right',
    va = 'center'
)
# cosmetics
ax.minorticks_off()
ax.set_xlabel('time (min)')
ax.set_ylabel('KL')

# legend
legend_k_lines = tuple(
    mlines.Line2D([], [], color=color, label='$K={}$'.format(k), lw=lw)
    for k, color in zip(KS, K_colors)
)
legend_style_lines = (
    mlines.Line2D([], [], color='gray', label='EP', lw=lw),
    mlines.Line2D([], [], color='gray', ls=':', label='snep.', lw=lw)
)
legend_full_lines = (mlines.Line2D([], [], color='k', label='full', lw=lw),)
legend_dummys = tuple(
    mlines.Line2D([0], [0], color="white")
    for _ in range(3)
)
axes[0].legend(
    handles = (
        legend_style_lines + legend_full_lines + legend_dummys +
        legend_k_lines
    ),
    ncol = 2
)

fig.subplots_adjust(
    top=0.97,
    bottom=0.07,
    left=0.12,
    right=0.96,
    hspace=0.08,
    wspace=0.2
)

# # limit x-axis
# axes[0].set_xlim([0, 100])
# # limit y-axis in axes[0]
# axes[0].set_ylim(bottom=0.005)
# # limit y-axis in axes[1]
# axes[1].set_ylim(bottom=1.0)

plt.show()
