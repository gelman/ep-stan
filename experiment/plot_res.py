"""Plot the results of the experiment from a result files.

Execute with:
    $ python plot_res.py <model_name> [<model_id>, [<dist_id>]]

The optional <dist_id> can be used to select the distributed method results
(EP and consensus) from file
    ..._<model_name>_<model_id>_<dist_id>.npz
while the true values, target values, and full method values are still obtained
from
    ..._<model_name>_<model_id>.npz
If <dist_id> is omitted, the same file ending is used also for distributed
results. If also <model_id> is omitted, no file ending are used.

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy import stats
import matplotlib.pyplot as plt


# Get the results directory
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(CUR_PATH, 'results')


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


def compare_plot(a, b, a_err=None, b_err=None, a_label=None, b_label=None,
                 ax=None):
    """Compare values of `a` in the ones in `b`."""
    # Ensure arrays
    a = np.asarray(a)
    b = np.asarray(b)

    # Plot into new axes or to the given one
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    # Plot basics
    ax.plot(b, a, 'bo')[0].get_axes()

    # Set common axis limits
    limits = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
              max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    # Plot diagonal
    ax.plot(limits, limits, 'r-')

    # Plot optional error bars
    if not a_err is None:
        a_err = np.asarray(a_err)
        if len(a_err.shape) == 2:
            a_p = a_err[0]
            a_m = a_err[1]
        else:
            a_p = a_err
            a_m = a_err
        ax.plot(np.tile(b, (2,1)), np.vstack((a+a_p, a-a_m)), 'b-')
    if not b_err is None:
        b_err = np.asarray(b_err)
        if len(b_err.shape) == 2:
            b_p = b_err[0]
            b_m = b_err[1]
        else:
            b_p = b_err
            b_m = b_err
        ax.plot(np.vstack((b+b_p, b-b_m)), np.tile(a, (2,1)), 'b-')

    # Optional labels
    if not a_label is None:
        ax.set_ylabel(a_label)
    if not b_label is None:
        ax.set_xlabel(b_label)

    return ax


def plot_results(model_name, model_id=None, dist_id=None):
    """Plot some results."""

    # -------------
    #   load data
    # -------------

    # Handle optional model id and dist id
    if model_id:
        file_ending = model_name + '_' + model_id
    else:
        file_ending = model_name
    if dist_id:
        file_ending_dist = file_ending + '_' + dist_id
    else:
        file_ending_dist = file_ending

    # Load true values
    true_vals_file = np.load(
        os.path.join(RES_PATH, 'true_vals_{}.npz'.format(file_ending)))
    pnames = ['phi']
    pnames.extend(true_vals_file['pnames'])
    true_vals = [true_vals_file[par] for par in pnames]
    true_vals_file.close()

    # Load target file
    target_file = np.load(
        os.path.join(RES_PATH, 'target_{}.npz'.format(file_ending)))
    m_target = target_file['m_target']
    S_target = target_file['S_target']
    target_file.close()
    # Load target samples if found
    # target_samp_file_path = os.path.join(
    #     RES_PATH, 'target_samp_{}.npz'.format(file_ending))
    # if os.path.exists(target_samp_file_path):
    #     target_samp_file = np.load(target_samp_file_path)
    #     samp_target = target_samp_file['samp_target']
    #     target_samp_file.close()
    # else:
    #     samp_target = None

    # Load EP result file
    res_d_file = np.load(
        os.path.join(RES_PATH, 'res_d_{}.npz'.format(file_ending_dist)))
    m_s_ep = res_d_file['m_s_ep']
    S_s_ep = res_d_file['S_s_ep']
    time_s_ep = res_d_file['time_s_ep']
    mstepsize_s_ep = res_d_file['mstepsize_s_ep']
    mrhat_s_ep = res_d_file['mrhat_s_ep']
    if 'm_phi_ep' in res_d_file.files:
        mix = True
        res_d = [
            (   res_d_file['m_'+par],
                (   res_d_file['v_'+par+'_ep']
                    if par != 'phi' else
                    np.diag(res_d_file['S_'+par+'_ep'])
                )
            )
            for par in pnames
        ]
    else:
        mix = False
    res_d_file.close()

    # Load full result file
    res_f_file = np.load(
        os.path.join(RES_PATH, 'res_f_{}.npz'.format(file_ending)))
    m_s_full = res_f_file['m_s_full']
    S_s_full = res_f_file['S_s_full']
    time_s_full = res_f_file['time_s_full']
    mstepsize_s_full = res_f_file['mstepsize_s_full']
    mrhat_s_full = res_f_file['mrhat_s_full']
    res_f_file.close()

    # Load consensus result file
    res_c_file = np.load(
        os.path.join(RES_PATH, 'res_c_{}.npz'.format(file_ending_dist)))
    m_s_cons = res_c_file['m_s_cons']
    S_s_cons = res_c_file['S_s_cons']
    time_s_cons = res_c_file['time_s_cons']
    mstepsize_s_cons = res_c_file['mstepsize_s_cons']
    mrhat_s_cons = res_c_file['mrhat_s_cons']
    res_c_file.close()

    # ---------
    #   plots
    # ---------

    niter = m_s_ep.shape[0]
    dphi = m_s_ep.shape[1]

    # Ravel params if necessary
    for pi in range(1,len(pnames)):
        if true_vals[pi].ndim != 1:
            true_vals[pi] = true_vals[pi].ravel()
            if mix:
                res_d[pi] = (res_d[pi][0].ravel(), res_d[pi][1].ravel())

    # Plot approx KL-divergence and MSE
    sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(S_target)[0])))
    # EP
    mse_ep = np.mean((m_s_ep - m_target)**2, axis=1)
    kl_ep = np.empty(niter)
    for i in range(niter):
        kl_ep[i] = kl_mvn(
            m_target, S_target, m_s_ep[i], S_s_ep[i], sum_log_diag_cho_S0)
    # full
    mse_full = np.mean((m_s_full - m_target)**2, axis=1)
    kl_full = np.empty(niter)
    for i in range(niter):
        kl_full[i] = kl_mvn(
            m_target, S_target, m_s_full[i], S_s_full[i], sum_log_diag_cho_S0)
    # consensus
    mse_cons = np.mean((m_s_cons - m_target)**2, axis=1)
    kl_cons = np.empty(niter)
    for i in range(niter):
        kl_cons[i] = kl_mvn(
            m_target, S_target, m_s_cons[i], S_s_cons[i], sum_log_diag_cho_S0)
    # iteration as x-axis
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(np.arange(niter), mse_ep, label='ep')
    ax.plot(np.arange(niter), mse_full, label='full')
    ax.plot(np.arange(niter), mse_cons, label='cons')
    ax.set_xlabel('iter')
    ax.set_ylabel('MSE')
    ax = axes[1]
    ax.plot(np.arange(niter), kl_ep, label='ep')
    ax.plot(np.arange(niter), kl_full, label='full')
    ax.plot(np.arange(niter), kl_cons, label='cons')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('iter')
    ax.set_ylabel('KL')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    # time as x-axis
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.plot(time_s_ep/60, mse_ep, label='ep')
    ax.plot(time_s_full/60, mse_full, label='full')
    ax.plot(time_s_cons/60, mse_cons, label='cons')
    ax.set_xlabel('time (min)')
    ax.set_ylabel('MSE')
    ax = axes[1]
    ax.plot(time_s_ep/60, kl_ep, label='ep')
    ax.plot(time_s_full/60, kl_full, label='full')
    ax.plot(time_s_cons/60, kl_cons, label='cons')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('time (min)')
    ax.set_ylabel('KL')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    # # Plot mean and variance as a function of the iteration
    # fig, axs = plt.subplots(2, 1, sharex=True)
    # axs[0].set_xlim([0,niter-1])
    # fig.subplots_adjust(hspace=0.1)
    # if mix:
    #     # TODO
    #     pass
    # else:
    #     axs[0].plot(np.arange(niter), m_s_ep)
    #     axs[1].plot(
    #         np.arange(niter),
    #         np.sqrt(np.diagonal(S_s_ep, axis1=1, axis2=2))
    #     )
    # axs[0].set_ylabel('Mean of $\phi$')
    # axs[1].set_ylabel('Std of $\phi$')
    # axs[1].set_xlabel('Iteration')
    #
    # if mix:
    #     # Plot compare plots for every variable
    #     # TODO
    #     pass
    #
    # else:
    #     # Plot compare plots for phi
    #     fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    #     fig.subplots_adjust(left=0.08, right=0.94)
    #     fig.suptitle('phi')
    #     # Mean
    #     compare_plot(
    #         m_s_full, m_s_ep[-1],
    #         a_label='full',
    #         b_label='epstan',
    #         ax=axs[0]
    #     )
    #     axs[0].set_title('mean')
    #     # Var
    #     compare_plot(
    #         np.diag(S_s_full), np.diag(S_s_ep[-1]),
    #         a_label='full',
    #         b_label='epstan',
    #         ax=axs[1]
    #     )
    #     axs[1].set_title('var')

    plt.show()


if __name__ == '__main__':
    if len(os.sys.argv) > 1 and len(os.sys.argv) < 5:
        plot_results(*os.sys.argv[1:])
    else:
        raise TypeError("Provide the model name as command line argument")
