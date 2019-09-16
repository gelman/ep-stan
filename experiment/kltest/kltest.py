
import numpy as np
from scipy import linalg, stats

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







n_dim = 16
seed = 11
n_samp = 200
n_trial = 8000
df = 4
use_precalculated = False  # loads precalculated results
use_t_instead_of_n = False


# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')
# LAPACK qr routine
dgeqrf_routine = linalg.get_lapack_funcs('geqrf')

# lower triangular indices
i_lower = np.tril_indices(n_dim, -1)


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


def rand_cov(n_dim, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    # variance
    var_fk = np.exp(rng.randn(n_dim))
    var_fk[var_fk<0.9] = 0.9+rng.uniform(-0.1, 0.1, size=sum(var_fk<0.9))
    var_fk[var_fk>2.9] = 2.9+rng.uniform(-0.1, 0.1, size=sum(var_fk>2.9))
    # correlation
    alphas = np.full(n_dim, 4*n_dim, dtype=np.float)
    i_high_alphas = rng.choice(n_dim, size=round(n_dim/4), replace=False)
    alphas[i_high_alphas] *= 4
    eigs = stats.dirichlet.rvs(alphas, random_state=rng)[0]
    eigs *= n_dim/np.sum(eigs)  # ensure unit len
    S = stats.random_correlation.rvs(eigs, random_state=rng)
    # and covariance
    sqrt_vars = np.sqrt(var_fk)
    S *= sqrt_vars
    S *= sqrt_vars[:,None]
    return S


rng = np.random.RandomState(seed)

# target distribution
S_t = rand_cov(n_dim, rng=rng)
# mean
m_t = rng.randn(n_dim)
# invert
Q_t, r_t = invert_params(S_t, m_t)
# for KL calculation
cho_S_t = linalg.cholesky(S_t)
sum_log_diag_cho_S_t = np.sum(np.log(np.diag(cho_S_t)))
# for t sampling
cho_S_t_scaled = np.sqrt((df-2)/df)*cho_S_t


# placeholders
S_samp = np.zeros((n_dim, n_dim), order='F')
m_samp = np.zeros(n_dim)

# test
if not use_precalculated:
    kl_moment = np.zeros(n_trial)
    kl_natural = np.zeros(n_trial)
    for trial_i in range(n_trial):
        # sample
        if not use_t_instead_of_n:
            # normal
            samp = samp_n_natural(Q_t, r_t, n_samp=n_samp, rng=rng)
        else:
            # t
            samp = samp_mvt(df, m_t, cho_S_t_scaled, n_samp=n_samp, rng=rng)

        m_samp = np.mean(samp, axis=0)
        samp -= m_samp
        Scatter = samp.T.dot(samp)

        # estim unbias moment
        S_samp = Scatter / (n_samp-1)
        # calc KL
        kl_moment[trial_i] = kl_mvn(
            m_t,
            S_t,
            m_samp,
            S_samp,
            sum_log_diag_cho_S0=sum_log_diag_cho_S_t
        )

        # estim unbias natural
        S_samp = Scatter / (n_samp-n_dim-2)
        # calc KL
        kl_natural[trial_i] = kl_mvn(
            m_t,
            S_t,
            m_samp,
            S_samp,
            sum_log_diag_cho_S0=sum_log_diag_cho_S_t
        )
    # save
    np.savez(
        'kltest_res.npz',
        kl_moment=kl_moment,
        kl_natural=kl_natural,
        n_dim=n_dim,
        seed=seed,
        n_samp=n_samp,
        n_trial=n_trial,
        df=df,
    )
else:
    # load results
    res_file = np.load('kltest_res.npz')
    kl_moment = res_file['kl_moment']
    kl_natural = res_file['kl_natural']
    res_file.close()


fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize4latex(0.9, 0.65))
for ax, data, name in zip(
        axes, (kl_moment, kl_natural), ('unbiased', 'biased')):
    # ax.set_xscale('log')
    ax.hist(data, 30, color='C0')
    low, high = np.percentile(data, (2.5, 97.5))
    ax.axvline(np.mean(data), color='C1', label='mean')
    ax.axvline(np.median(data), color='C2', label='median')
    ax.axvline(low, color='C2', ls='--', label='2.5 % - 97.5 %')
    ax.axvline(high, color='C2', ls='--')
    ax.set_ylabel(name, rotation=0, ha='right')
    ax.set_yticks([])
axes[0].legend()
axes[-1].set_xlabel(r'$\mathrm{KL}(g_{\backslash k}(\theta)||g(\theta))$')
plt.tight_layout()
# plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.15)

plt.savefig("fig_kltest.pdf")
plt.savefig("fig_kltest.pgf")
