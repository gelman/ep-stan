
import numpy as np
from scipy import linalg, stats

import os, sys

# import rand corr vine
sys.path.append(os.path.abspath('../models'))
from common import rand_corr_vine

# LAPACK qr routine
dgeqrf_routine = linalg.get_lapack_funcs('geqrf')


# # figure size for latex
# # put `\the\textwidth` in the latex content to write it out in the document
# # LATEX_TEXTWIDTH_PT = 469.755
# LATEX_TEXTWIDTH_PT = 384.0
#
# def figsize4latex(width_scale, height_scale=None):
#     inches_per_pt = 1.0 / 72.27
#     fig_width = LATEX_TEXTWIDTH_PT * inches_per_pt * width_scale
#     if height_scale is None:
#         fig_height = fig_width * (np.sqrt(5.0)-1.0)/2.0
#     else:
#         fig_height = fig_width * height_scale
#     return (fig_width, fig_height)
#
# import matplotlib as mpl
# mpl.use("pgf")
# pgf_with_custom_preamble = {
#     "pgf.texsystem": "pdflatex",
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": [],
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 8,
#     "font.size": 8,
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.figsize": figsize4latex(0.9),  # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[T1]{fontenc}",
#         r"\usepackage[utf8]{inputenc}",
#         r"\usepackage{lmodern}"
#     ]
# }
# mpl.rcParams.update(pgf_with_custom_preamble)


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch




# ========================
# settings

n_dim = 3
seed = 11
n_samp = 200
n_trial = 2000
n_sites = 6

# true site variance range
min_site_var = np.exp(-1.0)
max_site_var = np.exp(1)

# prior: N(0, prior_var * I)
prior_var = 4.0**2 * max_site_var


# test site extreme
extreme_test_site_var = 2.0**2 * max_site_var
grid_test_site = [0.0, 0.1, 0.5, 1.0]

# cavity extreme
extreme_cavity_var = 1/((n_sites - 1)/extreme_test_site_var + 1/prior_var)
grid_cavity = [0.0, 0.1, 0.5, 1.0]


# snep_n_inner = 4
# snep_outer_is = set([1])
snep_n_inner = 1
snep_outer_is = set()
snep_damp = 1.0

use_precalculated = True  # loads precalculated results

# ========================


# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')
# LAPACK qr routine
dgeqrf_routine = linalg.get_lapack_funcs('geqrf')

# lower triangular indices
i_lower = np.tril_indices(n_dim, -1)

snep_n_samp = n_samp//snep_n_inner

# site indices
test_site_i = 0


def samp_mvt(df, m, cho_S, n_samp=1, rng=None):
    """Generate a random sample from multivariate t distribution."""
    if rng is None:
        rng = np.random.RandomState()
    n_dim = len(m)
    s_c = rng.chisquare(df, n_samp)/df
    s_n = rng.randn(n_samp, n_dim).dot(cho_S)
    np.sqrt(s_c, out=s_c)
    out = np.divide(s_n, s_c[:,None], out=s_n)
    out += m
    return out


def invert_params(Mat, vec, Mat_out=None, vec_out=None, cho_form=False):
    if Mat_out is None:
        Mat_out = np.copy(Mat, order='F')
    elif Mat_out is not Mat:
        np.copyto(Mat_out, Mat)
    if vec_out is None:
        vec_out = np.copy(vec, order='F')
    elif vec_out is not vec:
        np.copyto(vec_out, vec)
    if cho_form:
        cho = (Mat_out, False)
    else:
        cho = linalg.cho_factor(Mat_out, overwrite_a=True)
    linalg.cho_solve(cho, vec_out, overwrite_b=True)
    _, info = dpotri_routine(Mat_out, overwrite_c=True)
    if info:
        # should not happen if cholesky was ok
        raise linalg.LinAlgError(
            "dpotri LAPACK routine failed with error code {}".format(info))
    # copy upper triangular to bottom
    Mat_out[i_lower] = Mat_out.T[i_lower]
    return Mat_out, vec_out

def samp_n_natural(Q, r, n_samp=1, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    cho = linalg.cho_factor(Q)
    m = linalg.cho_solve(cho, r)
    z = rng.randn(r.shape[0], n_samp)
    samp = linalg.solve_triangular(cho[0], z).T
    samp += m
    return samp


def estim_moment(samp, multip=None):
    n_samp = samp.shape[0]
    if multip is None:
        multip = n_samp - 1
    mean_vec = np.mean(samp, axis=0)
    samp_f_c = np.subtract(samp, mean_vec, order='F')
    # Use QR-decomposition for obtaining Cholesky of the scatter
    # matrix (only R needed, Q-less algorithm would be nice).
    _, _, _, info = dgeqrf_routine(samp_f_c, overwrite_a=True)
    if info:
        raise linalg.LinAlgError(
            "dgeqrf LAPACK routine failed with error code {}".format(info))
    out_Q, out_r = invert_params(samp_f_c[:n_dim,:], mean_vec, cho_form=True)
    out_Q *= multip
    out_r *= multip
    return out_Q, out_r


def kl_mvn(m0, S0, m1, S1, sum_log_diag_cho_S0=None):
    """Calculate KL-divergence for multiv normal distributions

    Calculates KL(p||q), where p ~ N(m0,S0) and q ~ N(m1,S1). Optional argument
    sum_log_diag_cho_S0 is precomputed sum(log(diag(cho(S0))).

    """
    choS1 = linalg.cho_factor(S1)
    if sum_log_diag_cho_S0 is None:
        sum_log_diag_cho_S0 = np.sum(np.log(np.diag(linalg.cho_factor(S0)[0])))
    dm = m1-m0
    KL_div = (
        0.5*(
            np.trace(linalg.cho_solve(choS1, S0))
            + dm.dot(linalg.cho_solve(choS1, dm))
            - len(m0)
        )
        - sum_log_diag_cho_S0 + np.sum(np.log(np.diag(choS1[0])))
    )
    return KL_div

def kl_mvn_natural1(m0, S0, m1, Q1):
    """Calculate KL-divergence for multiv normal distributions

    Calculates KL(p||q), where p ~ N(m0,S0) and q ~ N(m1,S1).

    """
    dm = m1-m0
    KL_div = 0.5*(
        - np.log(linalg.det(S0))
        - np.log(np.abs(linalg.det(Q1))) # skip imaginary part
        + np.trace(Q1.dot(S0))
        + dm.dot(Q1.dot(dm))
        - len(m0)
    )
    return KL_div


def rand_cov(n_dim, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    # variance
    var_fk = rng.uniform(min_site_var, max_site_var, size=n_dim)
    # correlation
    S = rand_corr_vine(n_dim, seed=rng)
    # and covariance
    sqrt_vars = np.sqrt(var_fk)
    S *= sqrt_vars
    S *= sqrt_vars[:,None]
    return S


rng = np.random.RandomState(seed)

# placeholders
S_samp = np.zeros((n_dim, n_dim), order='F')
m_samp = np.zeros(n_dim)


# true sites
S_i_true = np.empty((n_dim, n_dim, n_sites), order='F')
m_i_true = np.empty((n_dim, n_sites), order='F')
Q_i_true = np.empty((n_dim, n_dim, n_sites), order='F')
r_i_true = np.empty((n_dim, n_sites), order='F')
for site_i in range(n_sites):
    S_i_true_i = rand_cov(n_dim, rng=rng)
    m_i_true_i = rng.randn(n_dim)
    Q_i_true_i, r_i_true_i = invert_params(S_i_true_i, m_i_true_i)
    S_i_true[:,:,site_i] = S_i_true_i
    m_i_true[:,site_i] = m_i_true_i
    Q_i_true[:,:,site_i] = Q_i_true_i
    r_i_true[:,site_i] = r_i_true_i

# prior
S_p = prior_var * np.eye(n_dim).T
m_p = np.zeros(n_dim)
Q_p, r_p = invert_params(S_p, m_p)

# target distribution
Q_t = np.sum(Q_i_true, axis=-1) + Q_p
r_t = np.sum(r_i_true, axis=-1) + r_p
S_t, m_t = invert_params(Q_t, r_t)
# for KL calculation
cho_S_t = linalg.cholesky(S_t)
sum_log_diag_cho_S_t = np.sum(np.log(np.diag(cho_S_t)))

# converged cavity
Q_c0_true = Q_t - Q_i_true[:,:,test_site_i]
r_c0_true = r_t - r_i_true[:,test_site_i]
S_c0_true, m_c0_true = invert_params(Q_c0_true, r_c0_true)

# extreme test sites
S_0_extreme = extreme_test_site_var * np.eye(n_dim).T
m_0_extreme = np.zeros(n_dim)

# extreme cavity
S_c0_extreme = extreme_cavity_var * np.eye(n_dim).T
m_c0_extreme = np.zeros(n_dim)

# test
if not use_precalculated:
    mse_init = np.zeros((len(grid_test_site), len(grid_cavity)))
    mse_moment = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    mse_natural = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    mse_snep = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    mse_site_init = np.zeros((len(grid_test_site), len(grid_cavity)))
    mse_site_moment = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    mse_site_natural = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    mse_site_snep = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_init = np.zeros((len(grid_test_site), len(grid_cavity)))
    kl_moment = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_natural = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_snep = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_site_init = np.zeros((len(grid_test_site), len(grid_cavity)))
    kl_site_moment = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_site_natural = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    kl_site_snep = np.zeros((len(grid_test_site), len(grid_cavity), n_trial))
    for grid_t_i, multip_t in enumerate(grid_test_site):

        # cur site approximatios
        S_0 = S_i_true[:,:,test_site_i].copy(order='F')
        m_0 = m_i_true[:,test_site_i].copy(order='F')
        S_0 *= 1 - multip_t
        S_0 += multip_t*S_0_extreme
        m_0 *= 1 - multip_t
        m_0 += multip_t*m_0_extreme
        Q_0, r_0 = invert_params(S_0, m_0)

        # calc initial site mse and kl
        mse_site_init[grid_t_i, :] = np.mean(
            (m_0 - m_i_true[:,test_site_i])**2)
        kl_site_init[grid_t_i, :] = kl_mvn(
            m_i_true[:,test_site_i],
            S_i_true[:,:,test_site_i],
            m_0,
            S_0,
        )

        for grid_c_i, multip_c in enumerate(grid_cavity):

            # current cavity distribution
            S_c0 = S_c0_true.copy(order='F')
            m_c0 = m_c0_true.copy(order='F')
            S_c0 *= 1 - multip_c
            S_c0 += multip_c*S_c0_extreme
            m_c0 *= 1 - multip_c
            m_c0 += multip_c*m_c0_extreme
            Q_c0, r_c0 = invert_params(S_c0, m_c0)

            # current global approx
            Q_g = Q_c0 + Q_0
            r_g = r_c0 + r_0
            S_g, m_g = invert_params(Q_g, r_g)

            # calc initial mse and kl
            mse_init[grid_t_i, grid_c_i] = np.mean((m_g - m_t)**2)
            kl_init[grid_t_i, grid_c_i] = kl_mvn(
                m_t,
                S_t,
                m_g,
                S_g,
                sum_log_diag_cho_S0=sum_log_diag_cho_S_t
            )

            # current tilted distribution
            Q_t0 = Q_c0 + Q_i_true[:,:,test_site_i]
            r_t0 = r_c0 + r_i_true[:,test_site_i]
            S_t0, m_t0 = invert_params(Q_t0, r_t0)

            for trial_i in range(n_trial):
                # sample
                samp = samp_n_natural(Q_t0, r_t0, n_samp=n_samp, rng=rng)

                m_samp = np.mean(samp, axis=0)
                samp -= m_samp
                Scatter = samp.T.dot(samp)
                # calc qr from it
                qr, _, _, info = dgeqrf_routine(samp)
                if info:
                    raise linalg.LinAlgError(
                        "dgeqrf LAPACK routine failed with error code {}"
                        .format(info)
                    )
                # calc unnormalised new global approx Q and r
                Q_g_new_un, r_g_new_un = invert_params(
                    qr[:n_dim,:], m_samp, cho_form=True)


                # estim unbias moment
                S_samp = Scatter / (n_samp-1)
                # calc mse
                mse_moment[grid_t_i, grid_c_i, trial_i] = np.mean(
                    (m_samp - m_t)**2)
                # calc KL
                kl_moment[grid_t_i, grid_c_i, trial_i] = kl_mvn(
                    m_t,
                    S_t,
                    m_samp,
                    S_samp,
                    sum_log_diag_cho_S0=sum_log_diag_cho_S_t
                )
                # calc new site distribution
                # Q_g_new, r_g_new = invert_params(S_samp, m_samp)
                Q_g_new = Q_g_new_un*(n_samp-1)
                r_g_new = r_g_new_un*(n_samp-1)
                Q_0_new = Q_g_new - Q_c0
                r_0_new = r_g_new - r_c0
                # not necessarily pos def
                m_0_new = linalg.solve(Q_0_new, r_0_new, assume_a='sym')
                # calc site mse and kl
                mse_site_moment[grid_t_i, grid_c_i, trial_i] = np.mean(
                    (m_0_new - m_i_true[:,test_site_i])**2)
                kl_site_moment[grid_t_i, grid_c_i, trial_i] = kl_mvn_natural1(
                    m_i_true[:,test_site_i],
                    S_i_true[:,:,test_site_i],
                    m_0_new,
                    Q_0_new,
                )

                # estim unbias natural
                S_samp = Scatter / (n_samp-n_dim-2)
                # calc mse
                mse_natural[grid_t_i, grid_c_i, trial_i] = np.mean(
                    (m_samp - m_t)**2)
                # calc KL
                kl_natural[grid_t_i, grid_c_i, trial_i] = kl_mvn(
                    m_t,
                    S_t,
                    m_samp,
                    S_samp,
                    sum_log_diag_cho_S0=sum_log_diag_cho_S_t
                )
                # calc new site distribution
                # Q_g_new, r_g_new = invert_params(S_samp, m_samp)
                Q_g_new = Q_g_new_un*(n_samp-n_dim-2)
                r_g_new = r_g_new_un*(n_samp-n_dim-2)
                Q_0_new = Q_g_new - Q_c0
                r_0_new = r_g_new - r_c0
                # not necessarily pos def
                m_0_new = linalg.solve(Q_0_new, r_0_new, assume_a='sym')
                # calc site mse and kl
                mse_site_natural[grid_t_i, grid_c_i, trial_i] = np.mean(
                    (m_0_new - m_i_true[:,test_site_i])**2)
                kl_site_natural[grid_t_i, grid_c_i, trial_i] = kl_mvn_natural1(
                    m_i_true[:,test_site_i],
                    S_i_true[:,:,test_site_i],
                    m_0_new,
                    Q_0_new,
                )

                # estim SNEP
                Q_aux = Q_g.copy(order='F')
                r_aux = r_g.copy()
                S_0_snep = S_0.copy(order='F')
                m_0_snep = m_0.copy()
                Q_0_snep = Q_0.copy(order='F')
                r_0_snep = r_0.copy()
                for snep_i in range(snep_n_inner):
                    # sample (different sampling mechanism)
                    Q_sampling = Q_aux - Q_0_snep + Q_i_true[:,:,0]
                    r_sampling = r_aux - r_0_snep + r_i_true[:,0]
                    samp = samp_n_natural(
                        Q_sampling, r_sampling, n_samp=snep_n_samp, rng=rng)
                    m_samp = np.mean(samp, axis=0)
                    samp -= m_samp
                    S_samp = samp.T.dot(samp) / (samp.shape[0]-1)
                    if snep_i == 0:
                        S_samp -= S_g
                        m_samp -= m_g
                    else:
                        S_prev, m_prev = invert_params(Q_c0+Q_0_snep, r_c0+r_0_snep)
                        S_samp -= S_prev
                        m_samp -= m_prev
                    S_samp *= snep_damp
                    m_samp *= snep_damp
                    S_0_snep += S_samp
                    m_0_snep += m_samp
                    Q_0_snep, r_0_snep = invert_params(S_0_snep, m_0_snep)
                    if snep_i in snep_outer_is:
                        Q_aux = Q_c0 + Q_0_snep
                        r_aux = r_c0 + r_0_snep
                Q_new = Q_c0 + Q_0_snep
                r_new = r_c0 + r_0_snep
                S_new, m_new = invert_params(Q_new, r_new)
                # calc mse
                mse_snep[grid_t_i, grid_c_i, trial_i] = np.mean((m_new - m_t)**2)
                # calc KL
                kl_snep[grid_t_i, grid_c_i, trial_i] = kl_mvn(
                    m_t,
                    S_t,
                    m_new,
                    S_new,
                    sum_log_diag_cho_S0=sum_log_diag_cho_S_t
                )
                # calc site mse and kl
                mse_site_snep[grid_t_i, grid_c_i, trial_i] = np.mean(
                    (m_0_snep - m_i_true[:,test_site_i])**2)
                kl_site_snep[grid_t_i, grid_c_i, trial_i] = kl_mvn(
                    m_i_true[:,test_site_i],
                    S_i_true[:,:,test_site_i],
                    m_0_snep,
                    S_0_snep,
                )


    # save
    np.savez(
        'kl_test_snep_res.npz',
        mse_init=mse_init,
        mse_moment=mse_moment,
        mse_natural=mse_natural,
        mse_snep=mse_snep,
        mse_site_init=mse_site_init,
        mse_site_moment=mse_site_moment,
        mse_site_natural=mse_site_natural,
        mse_site_snep=mse_site_snep,
        kl_init=kl_init,
        kl_moment=kl_moment,
        kl_natural=kl_natural,
        kl_snep=kl_snep,
        kl_site_init=kl_site_init,
        kl_site_moment=kl_site_moment,
        kl_site_natural=kl_site_natural,
        kl_site_snep=kl_site_snep,
        n_dim=n_dim,
        seed=seed,
        n_samp=n_samp,
        n_trial=n_trial,
        n_sites=n_sites,
        min_site_var=min_site_var,
        max_site_var=max_site_var,
        prior_var=prior_var,
        extreme_test_site_var=extreme_test_site_var,
        grid_test_site=grid_test_site,
        extreme_cavity_var=extreme_cavity_var,
        grid_cavity=grid_cavity,
        snep_n_inner=snep_n_inner,
        snep_outer_is=snep_outer_is,
        snep_damp=snep_damp,
    )
else:
    # load results
    res_file = np.load('kl_test_snep_res.npz')
    mse_init = res_file['mse_init']
    mse_moment = res_file['mse_moment']
    mse_natural = res_file['mse_natural']
    mse_snep = res_file['mse_snep']
    mse_site_init = res_file['mse_site_init']
    mse_site_moment = res_file['mse_site_moment']
    mse_site_natural = res_file['mse_site_natural']
    mse_site_snep = res_file['mse_site_snep']
    kl_init = res_file['kl_init']
    kl_moment = res_file['kl_moment']
    kl_natural = res_file['kl_natural']
    kl_snep = res_file['kl_snep']
    kl_site_init = res_file['kl_site_init']
    kl_site_moment = res_file['kl_site_moment']
    kl_site_natural = res_file['kl_site_natural']
    kl_site_snep = res_file['kl_site_snep']
    res_file.close()




def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# # fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize4latex(0.98, 0.85))
# fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,9))
# for ax, data, name in zip(
#         axes,
#         (mse_moment, mse_natural, mse_snep),
#         ('naive moment', 'normal precision', 'SNEP')):
#     # ax.set_xscale('log')
#     ax.hist(data, 30, color='C0')
#     low, high = np.percentile(data, (2.5, 97.5))
#     ax.axvline(np.mean(data), color='C1', label='mean')
#     ax.axvline(np.median(data), color='C2', label='median')
#     ax.axvline(low, color='C2', ls='--', label='2.5 % - 97.5 %')
#     ax.axvline(high, color='C2', ls='--')
#     ax.set_ylabel(name, rotation=0, ha='right')
#     ax.set_yticks([])
# axes[0].legend()
# axes[-1].set_xlabel(r'$\mathrm{MSE}(g_{\backslash k}(\theta)||g(\theta))$')
# plt.tight_layout()
# # plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.15)
#
# # plt.savefig("fig_mse_test_snep.pdf")
# # plt.savefig("fig_mse_test_snep.pgf")
#
#
# # fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize4latex(0.98, 0.85))
# fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,9))
# for ax, data, name in zip(
#         axes,
#         (kl_moment, kl_natural, kl_snep),
#         ('naive moment', 'normal precision', 'SNEP')):
#     # ax.set_xscale('log')
#     ax.hist(data, 30, color='C0')
#     low, high = np.percentile(data, (2.5, 97.5))
#     ax.axvline(np.mean(data), color='C1', label='mean')
#     ax.axvline(np.median(data), color='C2', label='median')
#     ax.axvline(low, color='C2', ls='--', label='2.5 % - 97.5 %')
#     ax.axvline(high, color='C2', ls='--')
#     ax.set_ylabel(name, rotation=0, ha='right')
#     ax.set_yticks([])
# axes[0].legend()
# axes[-1].set_xlabel(r'$\mathrm{KL}(g_{\backslash k}(\theta)||g(\theta))$')
# plt.tight_layout()
# # plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.15)
#
# # plt.savefig("fig_kl_test_snep.pdf")
# # plt.savefig("fig_kl_test_snep.pgf")




# # fig = plt.figure(figsize4latex(0.98, 0.65))
# fig = plt.figure(figsize=(10,9))
# outer = gridspec.GridSpec(
#     len(grid_test_site), len(grid_cavity), wspace=0.4, hspace=0.4)
# for i_t in range(len(grid_test_site)):
#     for i_c in range(len(grid_cavity)):
#         inner = gridspec.GridSpecFromSubplotSpec(
#             2,
#             1,
#             subplot_spec=outer[i_t*len(grid_test_site) + i_c],
#             wspace=0.2,
#             hspace=0.2
#         )
#         ax1 = plt.Subplot(fig, inner[0])
#         ax2 = plt.Subplot(fig, inner[1])
#         ax1.hist(kl_natural[i_t, i_c], 30)
#         ax2.hist(kl_snep[i_t, i_c], 30)
#
#         # sharex does not work here need to do manual
#         new_lims = (
#             min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
#             max(ax1.get_xlim()[1], ax2.get_xlim()[1])
#         )
#         ax1.set_xlim(new_lims)
#         ax2.set_xlim(new_lims)
#         ax1.set_xticklabels([])
#
#         fig.add_subplot(ax1)
#         fig.add_subplot(ax2)
#
# # plt.savefig("fig_test_snep.pdf")
# # plt.savefig("fig_test_snep.pgf")


# fig, axes = plt.subplots(
#     len(grid_test_site), len(grid_cavity),
#     sharex=True, figsize=(10,9)
# )
# # plt.subplots_adjust()
# for i_t in range(len(grid_test_site)):
#     for i_c in range(len(grid_cavity)):
#         ax1 = axes[-i_t-1, i_c]
#         # MSE
#         mse_n_025, mse_n_500, mse_n_975 = np.percentile(
#             mse_natural[i_t, i_c], [2.5, 50, 97.5])
#         mse_s_025, mse_s_500, mse_s_975 = np.percentile(
#             mse_snep[i_t, i_c], [2.5, 50, 97.5])
#         ax1.bar(
#             [-2, -1],
#             [mse_n_500, mse_s_500],
#             yerr=[
#                 [mse_n_500-mse_n_025, mse_s_500-mse_s_025],
#                 [mse_n_975-mse_n_500, mse_s_975-mse_s_500]
#             ],
#             color='C0'
#         )
#         # ax1.set_yticks([0, max(mse_n_500, mse_s_500)])
#         ax1.set_yticks([0, ax1.get_ylim()[1]])
#         ax1.tick_params('y', colors='C0')
#         remove_frame(ax1)
#         ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#         ax1.tick_params(axis=u'both', which=u'both', length=0)
#         # KL
#         ax2 = ax1.twinx()
#         kl_n_025, kl_n_500, kl_n_975 = np.percentile(
#             kl_natural[i_t, i_c], [2.5, 50, 97.5])
#         kl_s_025, kl_s_500, kl_s_975 = np.percentile(
#             kl_snep[i_t, i_c], [2.5, 50, 97.5])
#         ax2.bar(
#             [1, 2],
#             [kl_n_500, kl_s_500],
#             yerr=[
#                 [kl_n_500-kl_n_025, kl_s_500-kl_s_025],
#                 [kl_n_975-kl_n_500, kl_s_975-kl_s_500]
#             ],
#             color='C1'
#         )
#         # ax2.set_yticks([0, max(kl_n_500, kl_s_500)])
#         ax2.set_yticks([0, ax2.get_ylim()[1]])
#         ax2.tick_params('y', colors='C1')
#         remove_frame(ax2)
#         ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#         ax2.tick_params(axis=u'both', which=u'both', length=0)
# for ax in axes[-1,:]:
#     ax.set_xticks([-2, -1, 1, 2])
#     ax.set_xticklabels(
#         ['MSE EP', 'MSE SNEP', 'KL EP', 'KL SNEP'],
#         rotation=90, ha='left'
#     )
# plt.tight_layout()




# calc change
mse_change_natural = mse_natural - mse_init[:,:,None]
mse_change_snep = mse_snep - mse_init[:,:,None]
kl_change_natural = kl_natural - kl_init[:,:,None]
kl_change_snep = kl_snep - kl_init[:,:,None]
# calc site change
mse_schange_natural = mse_site_natural - mse_site_init[:,:,None]
mse_schange_snep = mse_site_snep - mse_site_init[:,:,None]
kl_schange_natural = kl_site_natural - kl_site_init[:,:,None]
kl_schange_snep = kl_site_snep - kl_site_init[:,:,None]




for data_n, data_s, fig_name in zip(
        (mse_schange_natural, kl_schange_natural),
        (mse_schange_snep, kl_schange_snep),
        ('mse', 'kl')):

    # fig, axes = plt.subplots(
    #     len(grid_test_site), len(grid_cavity),
    #     sharex=True, figsize=figsize4latex(0.45, 0.65)
    # )
    fig, axes = plt.subplots(
        len(grid_test_site), len(grid_cavity),
        sharex=True, figsize=(6,7)
    )
    for i_t in range(len(grid_test_site)):
        for i_c in range(len(grid_cavity)):
            ax = axes[-i_t-1, i_c]
            n_025, n_500, n_975 = np.percentile(
                data_n[i_t, i_c], [2.5, 50, 97.5])
            s_025, s_500, s_975 = np.percentile(
                data_s[i_t, i_c], [2.5, 50, 97.5])
            ax.bar(
                [-0.5, 0.5],
                [n_500, s_500],
                # yerr=[
                #     [n_500-n_025, s_500-s_025],
                #     [n_975-n_500, s_975-s_500]
                # ],
                color=['C0', 'C1'],

            )
            # ylims = ax.get_ylim()
            # if ylims[0] < 0 and ylims[1] > 0:
            #     ax.set_yticks([ax.get_ylim()[0], 0, ax.get_ylim()[1]])
            # elif ylims[0] < 0:
            #     ax.set_yticks([ax.get_ylim()[0], 0])
            # else:
            #     ax.set_yticks([0, ax.get_ylim()[1]])
            ax.set_yticks(ax.get_ylim())
            ax.tick_params('y', colors='C0')
            if True:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                remove_frame(ax)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            ax.tick_params(axis=u'both', which=u'both', length=0)
    for ax in axes[-1,:]:
        ax.set_xticks([])
        # ax.set_xticks([-0.5, 0.5])
        # ax.set_xticklabels(
        #     ['EP', 'SNEP'],
        #     rotation=90, ha='left'
        # )

    plt.tight_layout()

    for i_c in range(len(grid_cavity)):
        ax = axes[-1, i_c]
        label = r'$d_\mathrm{cavity}='+'{:.1f}'.format(grid_cavity[i_c])+r'$'
        ax.text(0.5, -0.22, label, size=12, ha="center",
             transform=ax.transAxes)
    for i_t in range(len(grid_test_site)):
        ax = axes[-i_t-1, 0]
        label = r'$d_\mathrm{site}='+'{:.1f}'.format(grid_test_site[i_t])+r'$'
        ax.text(-0.3, 0.5, label, size=12, ha="right",
             transform=ax.transAxes)

    # legend
    legend_elements = (
        Patch(facecolor='C0', edgecolor='C0', label='moment matching'),
        Patch(facecolor='C1', edgecolor='C1', label='SNEP')
    )
    axes[-1,1].legend(
        handles = legend_elements,
        loc='upper center',
        bbox_to_anchor=(1.5, -0.3)
    )

    plt.subplots_adjust(bottom=0.125, left=0.179)

    # plt.savefig("fig_snep_{}.pdf".format(fig_name))
    # plt.savefig("fig_snep_{}.pgf".format(fig_name))
