"""Plot script for the paper with log likelihood."""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import stats

# figure size for latex
# put `\the\textwidth` in the latex content to write it out in the document
# LATEX_TEXTWIDTH_PT = 469.755
LATEX_TEXTWIDTH_PT = 384.0

def figsize4latex(width_scale, height_scale=None):
    inches_per_pt = 1.0 / 72.27
    fig_width = LATEX_TEXTWIDTH_PT * inches_per_pt * width_scale
    if height_scale is None:
        fig_height = fig_width * (np.sqrt(5.0)-1.0)/2.0
    else:
        fig_height = fig_width * height_scale
    return (fig_width, fig_height)


import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize4latex(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}"
    ]
}
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


gray_background = {
    'axes.axisbelow': True,
    'axes.edgecolor': 'white',
    'axes.facecolor': '#eaeaea',
    'axes.grid': True,
    'axes.linewidth': 0.0,
    'grid.color': 'white',
    'xtick.top': False,
    'xtick.bottom': False,
    'ytick.left': False,
    'ytick.right': False,
    'legend.facecolor': 'white'
}
plt.style.use(gray_background)
# plt.style.use('seaborn')



MODEL_NAME = 'm4b'
KS = (2, 4, 8, 16, 32, 64)
EP_ID = 'siter200'
CONS_ID = ''
FULL_N = 7



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

# Load target samples
target_samp_file = np.load('./results/target_samp_{}.npz'.format(MODEL_NAME))
samp_target = target_samp_file['samp_target']
target_samp_file.close()

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
#   Calc measures
# -------------------

dphi = m_target.shape[0]
sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(S_target)[0])))
triu_inds = np.triu_indices_from(S_target)

# EP
mse_ep_s = []
msestd_ep_s = []
kl_ep_s = []
ll_ep_s = []
for m_s_ep, S_s_ep in zip(m_s_ep_s, S_s_ep_s):

    mse_ep = np.mean((m_s_ep - m_target)**2, axis=1)
    mse_ep_s.append(mse_ep)

    msestd_ep = np.mean(
        np.square(
            np.sqrt(np.diagonal(S_s_ep, axis1=1, axis2=2)) -
            np.sqrt(np.diag(S_target))
        ),
        axis=1
    )
    msestd_ep_s.append(msestd_ep)

    kl_ep = np.empty(len(m_s_ep))
    for i in range(len(m_s_ep)):
        kl_ep[i] = kl_mvn(
            m_target, S_target, m_s_ep[i], S_s_ep[i], sum_log_diag_cho_S0)
    kl_ep_s.append(kl_ep)

    ll_ep = np.zeros(len(m_s_ep))
    for i in range(len(m_s_ep)):
        ll_ep[i] = np.sum(stats.multivariate_normal.logpdf(
            samp_target, mean=m_s_ep[i], cov=S_s_ep[i]))
    ll_ep_s.append(ll_ep)

# consensus
mse_cons_s = []
msestd_cons_s = []
kl_cons_s = []
ll_cons_s = []
for m_s_cons, S_s_cons in zip(m_s_cons_s, S_s_cons_s):

    mse_cons = np.mean((m_s_cons - m_target)**2, axis=1)
    mse_cons_s.append(mse_cons)

    msestd_cons = np.mean(
        np.square(
            np.sqrt(np.diagonal(S_s_cons, axis1=1, axis2=2)) -
            np.sqrt(np.diag(S_target))
        ),
        axis=1
    )
    msestd_cons_s.append(msestd_cons)

    kl_cons = np.empty(len(m_s_cons))
    for i in range(len(m_s_cons)):
        kl_cons[i] = kl_mvn(
            m_target, S_target, m_s_cons[i], S_s_cons[i], sum_log_diag_cho_S0)
    kl_cons_s.append(kl_cons)

    ll_cons = np.zeros(len(m_s_cons))
    for i in range(len(m_s_cons)):
        ll_cons[i] = np.sum(stats.multivariate_normal.logpdf(
            samp_target, mean=m_s_cons[i], cov=S_s_cons[i]))
    ll_cons_s.append(ll_cons)

# full

mse_full = np.mean((m_s_full - m_target)**2, axis=1)

msestd_full = np.mean(
    np.square(
        np.sqrt(np.diagonal(S_s_full, axis1=1, axis2=2)) -
        np.sqrt(np.diag(S_target))
    ),
    axis=1
)

kl_full = np.full(len(m_s_full), np.nan)
for i in range(len(m_s_full)):
    kl_full[i] = kl_mvn(
        m_target, S_target, m_s_full[i], S_s_full[i], sum_log_diag_cho_S0)

ll_full = np.zeros(len(m_s_full))
for i in range(len(m_s_full)):
    ll_full[i] = np.sum(stats.multivariate_normal.logpdf(
        samp_target, mean=m_s_full[i], cov=S_s_full[i]))


# ll negative
for ll_ep in ll_ep_s:
    ll_ep *= -1
for ll_cons in ll_cons_s:
    ll_cons *= -1
ll_full *= -1


# ---------
#   plots
# ---------

# get colors for each K
K_colors = plt.get_cmap('tab10').colors[:len(KS)]

# --------- time as x-axis
fig, axes = plt.subplots(1, 2, figsize=figsize4latex(0.99, 0.9))

lw = 1.0

# MSE mean
ax = axes[0]
ax.set_yscale('log')
# full
ax.plot(time_s_full[:FULL_N]/60, mse_full[:FULL_N], 'k', label='full', lw=lw)
# ep
for mse_ep, time_s_ep, k, color in zip(mse_ep_s, time_s_ep_s, KS, K_colors):
    ax.plot(time_s_ep[1:]/60, mse_ep[1:], color=color, label=str(k), lw=lw)
# cons
for mse_cons, time_s_cons, k, color in zip(
        mse_cons_s, time_s_cons_s, KS, K_colors):
    ax.plot(time_s_cons/60, mse_cons, color=color, ls=':', label=str(k), lw=lw)
# prior label
ax.axhline(mse_ep_s[0][0], lw=0.5, color='0.65', zorder=1)
ax.text(
    x = -0.05,
    y = mse_ep_s[0][0],
    s = 'prior',
    transform = ax.get_yaxis_transform(),
    ha = 'right',
    va = 'center'
)
# cosmetics
ax.minorticks_off()
ax.set_xlabel('time (min)')
ax.set_ylabel('MSE')

# ll
ax = axes[1]
ax.set_yscale('log')
# full
ax.plot(time_s_full[:FULL_N]/60, ll_full[:FULL_N], 'k', label='full', lw=lw)
# ep
for ll_ep, time_s_ep, k, color in zip(ll_ep_s, time_s_ep_s, KS, K_colors):
    ax.plot(time_s_ep[1:]/60, ll_ep[1:], color=color, label=str(k), lw=lw)
# cons
for ll_cons, time_s_cons, k, color in zip(
        ll_cons_s, time_s_cons_s, KS, K_colors):
    ax.plot(time_s_cons/60, ll_cons, color=color, ls=':', label=str(k), lw=lw)
# prior label
ax.axhline(ll_ep_s[0][0], lw=0.5, color='0.65', zorder=1)
ax.text(
    x = -0.05,
    y = ll_ep_s[0][0],
    s = 'prior',
    transform = ax.get_yaxis_transform(),
    ha = 'right',
    va = 'center'
)
# cosmetics
ax.minorticks_off()
ax.set_xlabel('time (min)')
ax.set_ylabel('MSE std')

# legend
legend_k_lines = tuple(
    mlines.Line2D([], [], color=color, label='$K={}$'.format(k), lw=lw)
    for k, color in zip(KS, K_colors)
)
legend_style_lines = (
    mlines.Line2D([], [], color='gray', label='EP', lw=lw),
    mlines.Line2D([], [], color='gray', ls=':', label='cons.', lw=lw)
)
legend_full_lines = (mlines.Line2D([], [], color='k', label='full', lw=lw),)
axes[1].legend(handles=legend_k_lines+legend_style_lines+legend_full_lines)

fig.subplots_adjust(
    top=0.95,
    bottom=0.1,
    left=0.1,
    right=0.955,
    hspace=0.2,
    wspace=0.3
)

plt.savefig("fig_ex1_timex.pdf")
plt.savefig("fig_ex1_timex.pgf")


# ------ as a function of K
fig, axes = plt.subplots(1, 2, figsize=figsize4latex(0.9, 0.5))
# mse
ax = axes[0]
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(
    KS,
    tuple(mse_ep[-1] for mse_ep in mse_ep_s),
    label='EP',
    color=K_colors[0]
)
ax.plot(
    KS,
    tuple(mse_cons[-1] for mse_cons in mse_cons_s),
    label='cons.',
    color=K_colors[1]
)
ax.set_xticks(KS)
ax.set_xticklabels(map(str, KS))
ax.set_xlabel('$K$')
ax.set_ylabel('MSE')
# kl
ax = axes[1]
ax.set_yscale('log')
ax.set_xscale('log')
ax.plot(
    KS,
    tuple(kl_ep[-1] for kl_ep in kl_ep_s),
    label='EP',
    color=K_colors[0]
)
ax.plot(
    KS,
    tuple(kl_cons[-1] for kl_cons in kl_cons_s),
    label='cons.',
    color=K_colors[1]
)
ax.set_xticks(KS)
ax.set_xticklabels(map(str, KS))
ax.set_xlabel('$K$')
ax.set_ylabel('KL')
ax.set_yticks((1,10,100))  # manual
ax.set_ylim((1,100))
# legend
axes[1].legend()
fig.tight_layout()

plt.savefig("fig_ex1_kx.pdf")
plt.savefig("fig_ex1_kx.pgf")

# ------ pointwise compare plot EP and target
fig, axes = plt.subplots(2, 2, figsize=figsize4latex(0.99, 0.9))
fig.subplots_adjust(
top=0.91,
bottom=0.085,
left=0.09,
right=1.0,
hspace=0.3,
wspace=0.0
)

scatterkwargs = dict(s=8, alpha=None, zorder=4)
diagcolor = (1.0, 0.6, 0.6)

# K=2

# mean
ax = axes[0, 0]
# diagonal line
ax.plot([0, 1], [0, 1], lw=1, color=diagcolor, transform=ax.transAxes)
# data
ax.scatter(m_target, m_s_ep_s[0][-1], **scatterkwargs)
# limits
limits = (
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect('equal', 'box')
ax.set_xticks((-2, 0, 2))
ax.set_yticks((-2, 0, 2))
# titles
ax.set_title('mean', y=1.1, family='serif')
ax.text(
    -0.4, 0.5, '$K=2$',
    transform=ax.transAxes, rotation='vertical', va='center'
)
ax.set_xlabel('target', family='serif')
ax.set_ylabel('EP', family='serif')

# std
ax = axes[0, 1]
# diagonal line
ax.plot([0, 1], [0, 1], lw=1, color=diagcolor, transform=ax.transAxes)
# data
ax.scatter(
    np.sqrt(np.diag(S_target)),
    np.sqrt(np.diag(S_s_ep_s[0][-1])),
    **scatterkwargs
)
# limits
limits = (
    0,  # manually set std min to 0
    max(ax.get_xlim()[1], ax.get_ylim()[1])
)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect('equal', 'box')
ax.set_xticks((0, 1, 2))
ax.set_yticks((0, 1, 2))
# titles
ax.set_title('standard deviation', y=1.1, family='serif')
ax.set_xlabel('target', family='serif')
ax.set_ylabel('EP', family='serif')

# K=64

# mean
ax = axes[1, 0]
# diagonal line
ax.plot([0, 1], [0, 1], lw=1, color=diagcolor, transform=ax.transAxes)
# data
ax.scatter(m_target, m_s_ep_s[-1][-1], **scatterkwargs)
# limits
limits = (
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect('equal', 'box')
ax.set_xticks((-2, 0, 2))
ax.set_yticks((-2, 0, 2))
# titles
ax.text(
    -0.4, 0.5, '$K=64$',
    transform=ax.transAxes, rotation='vertical', va='center'
)
ax.set_xlabel('target', family='serif')
ax.set_ylabel('EP', family='serif')

# std
ax = axes[1, 1]
# diagonal line
ax.plot([0, 1], [0, 1], lw=1, color=diagcolor, transform=ax.transAxes)
# data
ax.scatter(
    np.sqrt(np.diag(S_target)),
    np.sqrt(np.diag(S_s_ep_s[-1][-1])),
    **scatterkwargs
)
# limits
limits = (
    0,  # manually set std min to 0
    max(ax.get_xlim()[1], ax.get_ylim()[1])
)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect('equal', 'box')
ax.set_xticks((0, 1, 2))
ax.set_yticks((0, 1, 2))
# titles
ax.set_xlabel('target', family='serif')
ax.set_ylabel('EP', family='serif')


plt.savefig("fig_ex1_comp.pdf")
plt.savefig("fig_ex1_comp.pgf")
