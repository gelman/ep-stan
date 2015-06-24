"""Plot the results of the experiment from a result files.

Execute with:
    $ python plot_res.py <model_name> [<model_id>, [<dist_id>]]

The optional <dist_id> can be used to select the distributed results from file 
    res_d_<model_name>_<model_id>_<dist_id>.npz
while the true values and the full values are still obtained from
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

from __future__ import division
import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt


# Get the results directory
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(CUR_PATH, 'results')


def kl_mvn(m0, S0, m1, S1, sum_log_diag_cho_S0=None):
    """Calculate KL-divergence from multivariate N(m0,S0) to N(m1,S1).
    
    Optional argument sum_log_diag_cho_S0 is precomputed sum(log(diag(cho(S0))).
    
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
    
    # Handle optional model id and dist id
    if model_id:
        file_ending = model_name + '_' + model_id
    else:
        file_ending = model_name
    if dist_id:
        file_ending_dist = file_ending + '_' + dist_id
    else:
        file_ending_dist = file_ending
    
    # Load distributed result file
    res_d_file = np.load(
        os.path.join(RES_PATH, 'res_d_{}.npz'.format(file_ending_dist)))
    m_phi_i = res_d_file['m_phi_i']
    cov_phi_i = res_d_file['cov_phi_i']
    m_phi = res_d_file['m_phi']
    cov_phi = res_d_file['cov_phi']
    res_d_file.close()
    
    # Load full result file
    res_f_file = np.load(
        os.path.join(RES_PATH, 'res_f_{}.npz'.format(file_ending)))
    m_phi_full = res_f_file['m_phi_full']
    cov_phi_full = res_f_file['cov_phi_full']
    np.diag(res_f_file['cov_phi_full'])
    res_f_file.close()
    
    niter = m_phi_i.shape[0]
    dphi = m_phi_i.shape[1]
    
    # Plot approx KL-divergence and MSE
    sum_log_diag_cho_S0 = np.sum(np.log(np.diag(cho_factor(cov_phi_full)[0])))
    KL_divs = np.empty(niter)
    for i in xrange(niter):
        KL_divs[i] = kl_mvn(m_phi_full, cov_phi_full, m_phi_i[i], cov_phi_i[i],
                            sum_log_diag_cho_S0)
    mses = np.mean(np.square(m_phi_full - m_phi_i), axis=1)
    plt.figure()
    plt.plot(np.arange(niter), KL_divs, label='KL')
    plt.plot(np.arange(niter), mses, label='MSE')
    plt.ylabel('Approximated errors')
    plt.xlabel('Iteration')
    
    # Plot compare plots
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    #~ fig.subplots_adjust(left=0.08, right=0.94)
    compare_plot(
        m_phi_full,
        m_phi,
        a_label='full',
        b_label='distributed',
        ax=axs[0]
    )
    compare_plot(
        np.sqrt(np.diag(cov_phi_full)),
        np.sqrt(np.diag(cov_phi)),
        a_label='full',
        b_label='distributed',
        ax=axs[1]
    )
    axs[0].set_title('mean')
    axs[1].set_title('std')
    
    plt.show()


if __name__ == '__main__':
    if len(os.sys.argv) > 1 and len(os.sys.argv) < 5:
        plot_results(*os.sys.argv[1:])
    else:
        raise TypeError("Provide the model name as command line argument")



