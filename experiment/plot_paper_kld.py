"""Plot script for the paper."""

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
# LATEX_TEXTWIDTH_PT = 384.0
LATEX_TEXTWIDTH_PT = 433.62

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
    'legend.facecolor': 'white',
    'legend.framealpha': None,
    'legend.fancybox': False,
    'legend.edgecolor': 'white'
}
# plt.style.use(gray_background)
# plt.style.use('seaborn')



MODEL_NAME = 'm4b'
KS = (2, 4, 8, 16, 32, 64)
EP_ID = ''
CONS_ID = ''
FULL_N = 11



def kl_mvn(m0, S0, m1, S1):
    """Calculate marginal KL-divergence for normal distributions

    Calculates KL(p||q), where p ~ N(m0,s0) and q ~ N(m1,s1).

    """
    S0 = np.diag(np.diag(S0))
    S1 = np.diag(np.diag(S1))
    choS1 = cho_factor(S1)
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

# EP
mse_ep_s = []
kl_ep_s = []
for m_s_ep, S_s_ep in zip(m_s_ep_s, S_s_ep_s):
    mse_ep = np.mean((m_s_ep - m_target)**2, axis=1)
    kl_ep = np.empty(len(m_s_ep))
    for i in range(len(m_s_ep)):
        kl_ep[i] = kl_mvn(
            m_target, S_target, m_s_ep[i], S_s_ep[i])
    mse_ep_s.append(mse_ep)
    kl_ep_s.append(kl_ep)
# consensus
mse_cons_s = []
kl_cons_s = []
for m_s_cons, S_s_cons in zip(m_s_cons_s, S_s_cons_s):
    mse_cons = np.mean((m_s_cons - m_target)**2, axis=1)
    kl_cons = np.empty(len(m_s_cons))
    for i in range(len(m_s_cons)):
        kl_cons[i] = kl_mvn(
            m_target, S_target, m_s_cons[i], S_s_cons[i])
    mse_cons_s.append(mse_cons)
    kl_cons_s.append(kl_cons)
# full
mse_full = np.mean((m_s_full - m_target)**2, axis=1)
kl_full = np.full(len(m_s_full), np.nan)
for i in range(len(m_s_full)):
    kl_full[i] = kl_mvn(
        m_target, S_target, m_s_full[i], S_s_full[i])


# ---------
#   plots
# ---------

# get colors for each K
K_colors = plt.get_cmap('tab10').colors[:len(KS)]

# --------- time as x-axis
fig, ax = plt.subplots(1, 1, figsize=figsize4latex(0.95, 0.5))

lw = 1.0

# KL
ax.set_yscale('log')
# full
ax.plot(time_s_full[:FULL_N]/60, kl_full[:FULL_N], 'k', label='full', lw=1.5)
# ep
for kl_ep, time_s_ep, k, color in zip(kl_ep_s, time_s_ep_s, KS, K_colors):
    ax.plot(time_s_ep[1:]/60, kl_ep[1:], color=color, label=str(k), lw=lw)
# cons
for kl_cons, time_s_cons, k, color in zip(
        kl_cons_s, time_s_cons_s, KS, K_colors):
    ax.plot(time_s_cons/60, kl_cons, color=color, ls=':', label=str(k), lw=lw)
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
ax.set_xlim([0, 100])
# cosmetics
ax.minorticks_off()
ax.set_xlabel('time (min)')
ax.set_ylabel('KL')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# legend
legend_k_lines = tuple(
    mlines.Line2D([], [], color=color, label='$K={}$'.format(k), lw=lw)
    for k, color in zip(KS, K_colors)
)
legend_style_lines = (
    mlines.Line2D([], [], color='gray', label='EP', lw=lw),
    mlines.Line2D([], [], color='gray', ls=':', label='cons.', lw=lw)
)
legend_full_lines = (mlines.Line2D([], [], color='k', label='full', lw=1.5),)
legend_dummys = tuple(
    mlines.Line2D([0], [0], color="white")
    for _ in range(3)
)
fig.tight_layout()

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.legend(
    handles = (
        legend_style_lines + legend_full_lines + legend_dummys +
        legend_k_lines
    ),
    ncol = 1,
    framealpha=1.0,
    fancybox=False,
    loc='center left',
    bbox_to_anchor=(1.05, 0.5),
    frameon=False,
)

# fig.subplots_adjust(
#     top=0.97,
#     bottom=0.07,
#     left=0.12,
#     right=0.96,
#     hspace=0.08,
#     wspace=0.2
# )

# limit x-axis

# limit y-axis in ax
# ax.set_ylim(bottom=0.005)
# limit y-axis in ax
# ax.set_ylim(bottom=1.0)

plt.savefig("fig_ex1_timex_kld.pdf")
plt.savefig("fig_ex1_timex_kld.pgf")
