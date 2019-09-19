
import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt


n_sites = 16
n_dim = 8
prior_sigma2 = 5.0**2
seed = 11
n_samp = 1000
n_trial = 40
n_iter = 6


# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')
# LAPACK qr routine
dgeqrf_routine = linalg.get_lapack_funcs('geqrf')

# lower triangular indices
i_lower = np.tril_indices(n_dim, -1)


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

def rand_cov(n_dim, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    # variance
    var_fk = np.exp(rng.randn(n_dim))
    var_fk[var_fk<0.9] = 0.9+rng.uniform(-0.1, 0.1, size=sum(var_fk<0.9))
    var_fk[var_fk>2.9] = 2.9+rng.uniform(-0.1, 0.1, size=sum(var_fk>2.9))
    # correlation
    alphas = np.full(n_dim, 2*n_dim, dtype=np.float)
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


# true site distributions
Q_fk = np.empty((n_dim, n_dim, n_sites), order='F')
r_fk = np.empty((n_dim, n_sites), order='F')
for k in range(n_sites):
    # mean
    m_fk = rng.randn(n_dim)
    # covariance
    S_fk = rand_cov(n_dim=n_dim, rng=rng)
    # invert
    invert_params(S_fk, m_fk, Mat_out=Q_fk[:,:,k], vec_out=r_fk[:,k])


# initial site approximations
Qk_0 = np.zeros((n_dim, n_dim, n_sites), order='F')
rk_0 = np.zeros((n_dim, n_sites), order='F')

# prior distribution
S_0 = prior_sigma2*np.eye(n_dim, order='F')
m_0 = np.zeros(n_dim)
Q_0, r_0 = invert_params(S_0, m_0)

# true distribution
Q_g_true = Q_fk.sum(axis=-1)
Q_g_true += Q_0
r_g_true = r_fk.sum(axis=-1)
r_g_true += r_0
S_g_true, m_g_true = invert_params(Q_g_true, r_g_true)


# ----------------------------------------------------------------------------
# sequential MCMC EP
# ----------------------------------------------------------------------------

S_g = np.zeros((n_dim, n_dim, n_iter+1, n_trial), order='F')
m_g = np.zeros((n_dim, n_iter+1, n_trial), order='F')

for trial_i in range(n_trial):

    Qk = np.copy(Qk_0, order='F')
    rk = np.copy(rk_0, order='F')

    Q_g = Qk.sum(axis=-1)
    Q_g += Q_0
    r_g = rk.sum(axis=-1)
    r_g += r_0

    invert_params(
        Q_g, r_g,
        Mat_out=S_g[:,:,0,trial_i],
        vec_out=m_g[:,0,trial_i]
    )

    for cur_i in range(1, n_iter+1):

        for k in range(n_sites):
            Q_tilted = Q_g - Qk[:,:,k] + Q_fk[:,:,k]
            r_tilted = r_g - rk[:,k] + r_fk[:,k]

            samp = samp_n_natural(Q_tilted, r_tilted, n_samp=n_samp, rng=rng)
            Q_t_samp, r_t_samp = estim_moment(samp)

            Qd = Q_t_samp - Q_g
            rd = r_t_samp - r_g
            Q_g = Q_t_samp
            r_g = r_t_samp
            Qk[:,:,k] += Qd
            rk[:,k] += rd

        invert_params(
            Q_g, r_g,
            Mat_out=S_g[:,:,cur_i,trial_i],
            vec_out=m_g[:,cur_i,trial_i]
        )


# ----------------------------------------------------------------------------
# sequential MCMC EP bias
# ----------------------------------------------------------------------------

S_g_bias = np.zeros((n_dim, n_dim, n_iter+1, n_trial), order='F')
m_g_bias = np.zeros((n_dim, n_iter+1, n_trial), order='F')

for trial_i in range(n_trial):

    Qk = np.copy(Qk_0, order='F')
    rk = np.copy(rk_0, order='F')

    Q_g = Qk.sum(axis=-1)
    Q_g += Q_0
    r_g = rk.sum(axis=-1)
    r_g += r_0

    invert_params(
        Q_g, r_g,
        Mat_out=S_g_bias[:,:,0,trial_i],
        vec_out=m_g_bias[:,0,trial_i]
    )

    for cur_i in range(1, n_iter+1):

        for k in range(n_sites):
            Q_tilted = Q_g - Qk[:,:,k] + Q_fk[:,:,k]
            r_tilted = r_g - rk[:,k] + r_fk[:,k]

            samp = samp_n_natural(Q_tilted, r_tilted, n_samp=n_samp, rng=rng)
            Q_t_samp, r_t_samp = estim_moment(samp, multip=n_samp-n_dim-2)

            Qd = Q_t_samp - Q_g
            rd = r_t_samp - r_g
            Q_g = Q_t_samp
            r_g = r_t_samp
            Qk[:,:,k] += Qd
            rk[:,k] += rd

        invert_params(
            Q_g, r_g,
            Mat_out=S_g_bias[:,:,cur_i,trial_i],
            vec_out=m_g_bias[:,cur_i,trial_i]
        )


# ============================================================================
# plotting
# ============================================================================

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

# calc kl for each iteration
sum_log_diag_cho_S0 = np.sum(np.log(np.diag(linalg.cho_factor(S_g_true)[0])))
kl_ep = np.zeros((n_iter+1, n_trial))
kl_ep_bias = np.zeros((n_iter+1, n_trial))
for trial_i in range(n_trial):
    for cur_i in range(n_iter+1):
        kl_ep[cur_i,trial_i] = kl_mvn(
            m_g_true,
            S_g_true,
            m_g[:,cur_i,trial_i],
            S_g[:,:,cur_i,trial_i],
            sum_log_diag_cho_S0=sum_log_diag_cho_S0
        )
        kl_ep_bias[cur_i,trial_i] = kl_mvn(
            m_g_true,
            S_g_true,
            m_g_bias[:,cur_i,trial_i],
            S_g_bias[:,:,cur_i,trial_i],
            sum_log_diag_cho_S0=sum_log_diag_cho_S0
        )

# calc mse for each iteration
mse_ep = np.mean((m_g - m_g_true[:,None,None])**2, axis=0)
mse_ep_bias = np.mean((m_g_bias - m_g_true[:,None,None])**2, axis=0)

# calc kl ignore cor for each iteration
S_g_true_nocor = np.diag(np.diag(S_g_true)).T
sum_log_diag_cho_S0_nocor = np.sum(
    np.log(np.diag(linalg.cho_factor(S_g_true_nocor)[0])))
klnoc_ep = np.zeros((n_iter+1, n_trial))
klnoc_ep_bias = np.zeros((n_iter+1, n_trial))
for trial_i in range(n_trial):
    for cur_i in range(n_iter+1):
        klnoc_ep[cur_i,trial_i] = kl_mvn(
            m_g_true,
            S_g_true_nocor,
            m_g[:,cur_i,trial_i],
            np.diag(np.diag(S_g[:,:,cur_i,trial_i])).T,
            sum_log_diag_cho_S0=sum_log_diag_cho_S0_nocor
        )
        klnoc_ep_bias[cur_i,trial_i] = kl_mvn(
            m_g_true,
            S_g_true_nocor,
            m_g_bias[:,cur_i,trial_i],
            np.diag(np.diag(S_g_bias[:,:,cur_i,trial_i])).T,
            sum_log_diag_cho_S0=sum_log_diag_cho_S0_nocor
        )

# plot
fig, axes = plt.subplots(3, 1, sharex=True)
for ax, d1, d2 in zip(
            axes,
            (kl_ep, mse_ep, klnoc_ep),
            (kl_ep_bias, mse_ep_bias, klnoc_ep_bias)
        ):
    ax.set_yscale('log')
    line_mean = np.mean(d1, axis=1)
    line_up_low = np.percentile(d1, [2.5, 97.5], axis=1)
    ax.plot(line_mean, label='EP MCMC', color='C0')
    ax.plot(line_up_low.T, color='C0', ls='--')
    line_mean = np.mean(d2, axis=1)
    line_up_low = np.percentile(d2, [2.5, 97.5], axis=1)
    ax.plot(line_mean, label='EP MCMC bias', color='C1')
    ax.plot(line_up_low.T, color='C1', ls='--')
axes[-1].legend()
