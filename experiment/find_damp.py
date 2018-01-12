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
import epstan
from epstan.util import invert_normal_params


CHAINS = 8
SITER = 400
N_DAMP = 31


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


def main(model_name, K=None, iters=None):

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

    if K is None:
        K = J

    if iters is None:
        iters = fit.DEFAULT_ITERS_TO_RUN(K)

    conf = fit.configurations(J=J, D=D, K=K, chains=CHAINS, siter=SITER)
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

    # selected damps
    df0 = fit.default_df0(K, iters)

    damps = np.linspace(0, 1, N_DAMP+2)[1:-1]
    mses = np.full((iters, N_DAMP), np.nan)
    lls = np.full((iters, N_DAMP), np.nan)
    kls = np.full((iters, N_DAMP), np.nan)
    damps_selected = np.full(iters, np.nan)
    mses_selected = np.full(iters+1, np.nan)
    lls_selected = np.full(iters+1, np.nan)
    kls_selected = np.full(iters+1, np.nan)

    # initial selection criteria
    init_S, init_m = master.cur_approx()
    # mse
    mses_selected[0] = np.mean((init_m - m_target)**2)
    # likelihood
    lls_selected[0] = np.sum(stats.multivariate_normal.logpdf(
        samp_target, mean=init_m, cov=init_S.T))
    # approximate KL
    kls_selected[0] = kl_mvn(
        m_target, S_target, init_m, init_S.T, sum_log_diag_cho_S0)


    posdefs = np.zeros(master.K, dtype=bool)

    # iters
    for iter_ind in range(iters):
        curiter = iter_ind + 1
        print("Iteration {}/{}".format(curiter, iters))

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
        # best_idx = np.nanargmin(kls[iter_ind])
        # if iter_ind <= iters / 2:
        #     best_idx = np.nanargmin(mses[iter_ind])
        # else:
        #     best_idx = np.nanargmin(kls[iter_ind])
        # df = damps[best_idx]

        # preselected
        df = df0(curiter)
        # check it was proper

        while True:
            # apply damp
            damps_selected[iter_ind] = df
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
                    mses_selected[iter_ind+1] = np.mean((m - m_target)**2)
                    # likelihood
                    lls_selected[iter_ind+1] = np.sum(stats.multivariate_normal.logpdf(
                        samp_target, mean=m, cov=S.T))
                    # approximate KL
                    kls_selected[iter_ind+1] = kl_mvn(
                        m_target, S_target, m, S.T, sum_log_diag_cho_S0)
                else:
                    # decay damp
                    print('    lowering-df-1')
                    df *= epstan.method.Master.DEFAULT_KWARGS['df_decay']
                    if df < epstan.method.Master.DEFAULT_KWARGS['df_treshold']:
                        print('    df_threshold reached')
                        break
                    continue

            except linalg.LinAlgError:
                # decay damp
                print('    lowering-df-2')
                df *= epstan.method.Master.DEFAULT_KWARGS['df_decay']
                if df < epstan.method.Master.DEFAULT_KWARGS['df_treshold']:
                    print('    df_threshold reached')
                    break
                continue

            # all ok
            break

        # switch Qi <> Qi2, ri <> ri2
        temp = Qi
        Qi = Qi2
        Qi2 = temp
        temp = ri
        ri = ri2
        ri2 = temp

    # save
    np.savez(
        os.path.join(RES_PATH, 'find_damp_K{}.npz'.format(K)),
        damps = damps,
        mses = mses,
        lls = lls,
        kls = kls,
        damps_selected = damps_selected,
        mses_selected = mses_selected,
        lls_selected = lls_selected,
        kls_selected = kls_selected,
    )


if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 2:
        kwargs['K'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        kwargs['iters'] = int(sys.argv[3])
    main(sys.argv[1], **kwargs)
