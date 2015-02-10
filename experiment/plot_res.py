"""Plot the results of the experiment from a result file.

Execute with:
    $ python plot_res.py <model_name>

The most recent version of the code can be found on GitHub:
https://github.com/gelman/ep-stan

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt


def compare_plot(a, b, a_err=None, b_err=None, a_label=None, b_label=None):
    """Compare values of `a` in the ones in `b`."""
    
    a = np.asarray(a)
    b = np.asarray(b)
    
    fig = plt.figure()
    ax = plt.plot(b, a, 'bo')[0].get_axes()
    limits = (min(ax.get_xlim()[0], ax.get_ylim()[0]),
              max(ax.get_xlim()[1], ax.get_ylim()[1]))
    ax.set_xlim(limits)
    ax.set_ylim(limits)
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
    ax.plot(limits, limits, 'r-')
    if not a_label is None:
        ax.set_ylabel(a_label)
    if not b_label is None:
        ax.set_xlabel(b_label)
    
    return fig


def plot_results(model_name):
    """Plot three plots from the results."""
    
    # Load true values
    true_vals = np.load('results/true_vals_{}.npz'.format(model_name))
    phi_true = true_vals['phi']
    alpha_true = true_vals['alpha']
    beta_true = true_vals['beta']
    true_vals.close()
    
    # Load distributed result file
    res_d = np.load('results/res_d_{}.npz'.format(model_name))
    m_phi = res_d['m_phi']
    var_phi = res_d['var_phi']
    m_phi_mix = res_d['m_phi_mix']
    var_phi_mix = res_d['var_phi_mix']
    m_alpha = res_d['m_alpha']
    var_alpha = res_d['var_alpha']
    m_beta = res_d['m_beta']
    var_beta = res_d['var_beta']
    res_d.close()
    
    # Load full result file
    res_f = np.load('results/res_f_{}.npz'.format(model_name))
    m_phi_full = res_f['m_phi_full']
    var_phi_full = res_f['var_phi_full']
    m_alpha_full = res_f['m_alpha_full']
    var_alpha_full = res_f['var_alpha_full']
    m_beta_full = res_f['m_beta_full']
    var_beta_full = res_f['var_beta_full']
    res_f.close()
    
    niter = m_phi.shape[0]
    dphi = m_phi.shape[1]
    
    # Ravel beta if necessary
    if beta_true.ndim != 1:
        beta_true = beta_true.ravel()
        m_beta = m_beta.ravel()
        var_beta = var_beta.ravel()
        m_beta_full = m_beta_full.ravel()
        var_beta_full = var_beta_full.ravel()
    
    # Plot mean and variance as a function of the iteration
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    axs[0].plot(np.arange(niter+1), np.vstack((m_phi, m_phi_mix)))
    axs[0].set_ylabel('Mean of params')
    axs[1].plot(np.arange(niter+1), np.sqrt(np.vstack((var_phi, var_phi_mix))))
    axs[1].set_ylabel('Std of params')
    axs[1].set_xlabel('Iteration')
    
    # Plot estimates vs true values
    compare_plot(
        phi_true,
        m_phi_mix,
        b_err=3*np.sqrt(var_phi_mix),
        a_label='True values',
        b_label='Estimated values ($\pm 3 \sigma$)'
    )
    plt.title('phi')
    
    # Plot full vs distributed
    compare_plot(
        m_phi_full,
        m_phi_mix,
        a_err=1.96*np.sqrt(var_phi_full),
        b_err=1.96*np.sqrt(var_phi_mix),
        a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
        b_label='Estimased from the dep model ($\pm 1.96 \sigma$)'
    )
    plt.title('phi')
    
    # Plot estimates vs true values
    compare_plot(
        alpha_true,
        m_alpha,
        b_err=3*np.sqrt(var_alpha),
        a_label='True values',
        b_label='Estimated values ($\pm 3 \sigma$)'
    )
    plt.title('alpha')
    
    # Plot estimates vs true values
    compare_plot(
        beta_true,
        m_beta,
        b_err=3*np.sqrt(var_beta),
        a_label='True values',
        b_label='Estimated values ($\pm 3 \sigma$)'
    )
    plt.title('beta')
    
    # Plot full vs distributed
    compare_plot(
        m_alpha_full,
        m_alpha,
        a_err=1.96*np.sqrt(var_alpha_full),
        b_err=1.96*np.sqrt(var_alpha),
        a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
        b_label='Estimased from the dep model ($\pm 1.96 \sigma$)'
    )
    plt.title('alpha')
    
    # Plot full vs distributed
    compare_plot(
        m_beta_full,
        m_beta,
        a_err=1.96*np.sqrt(var_beta_full),
        b_err=1.96*np.sqrt(var_beta),
        a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
        b_label='Estimased from the dep model ($\pm 1.96 \sigma$)'
    )
    plt.title('beta')
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_results(sys.argv[1])
    else:
        raise TypeError("Provide the model name as command line argument")



