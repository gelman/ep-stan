"""Plot the results of the experiment from a result file.

Execute with:
    $ python plot_res.py <filename>
where <filename> is the name of the result '.npz' file. If <filename> is
omitted, the default filename 'res.npz' is used.

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


def plot_results(filename='res.npz'):
    """Plot three plots from the results."""
    
    # Load result file
    res = np.load(filename)
    # Read necessary variables into current namespace
    niter = res['niter']
    phi_true = res['phi_true']
    m_phi = res['m_phi']
    var_phi = res['var_phi']
    m_mix = res['m_mix']
    var_mix = res['var_mix']
    m_phi_full = res['m_phi_full']
    var_phi_full = res['var_phi_full']
    res.close()
    
    # Plot mean and variance as a function of the iteration
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.1)
    axs[0].plot(np.arange(niter+1), np.vstack((m_phi, m_mix)))
    axs[0].set_ylabel('Mean of params')
    axs[1].plot(np.arange(niter+1), np.sqrt(np.vstack((var_phi, var_mix))))
    axs[1].set_ylabel('Std of params')
    axs[1].set_xlabel('Iteration')
    
    # Plot estimates vs true values
    compare_plot(phi_true, m_mix, b_err=3*np.sqrt(var_mix),
                 a_label='True values',
                 b_label='Estimated values ($\pm 3 \sigma$)')
    
    # Plot full vs distributed
    compare_plot(m_phi_full, m_mix,
                 a_err=1.96*np.sqrt(var_phi_full), b_err=1.96*np.sqrt(var_mix),
                 a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
                 b_label='Estimased from the dep model ($\pm 1.96 \sigma$)')
    
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        plot_results(sys.argv[1])
    else:
        plot_results()



