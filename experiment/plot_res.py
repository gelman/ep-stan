"""Plot the results of the experiment from the file `res.npz`.

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
import matplotlib.pyplot as plt

# Add parent dir to sys.path if not present already. This is only done because
# of easy importing of the package dep. Adding the parent directory into the
# PYTHONPATH works as well.
parent_dir = os.path.abspath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir))
# Double check that the package is in the parent directory
if os.path.exists(os.path.join(parent_dir, 'dep')):
    if parent_dir not in os.sys.path:
        os.sys.path.insert(0, parent_dir)

from dep.util import compare_plot


# ------------------------------------------------------------------------------
#     Load res.npz
# ------------------------------------------------------------------------------

res = np.load('res.npz')
# Read each variable into current namespace
for var in res.files:
    vars()[var] = res[var]
res.close()


# ------------------------------------------------------------------------------
#     Plot
# ------------------------------------------------------------------------------

# Mean and variance as a function of the iteration
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.1)
axs[0].plot(np.arange(niter+1), np.vstack((m_phi, m_mix)))
axs[0].set_ylabel('Mean of params')
axs[1].plot(np.arange(niter+1), np.sqrt(np.vstack((var_phi, var_mix))))
axs[1].set_ylabel('Std of params')
axs[1].set_xlabel('Iteration')

# Estimates vs true values
compare_plot(phi_true, m_mix, b_err=3*np.sqrt(var_mix),
             a_label='True values',
             b_label='Estimated values ($\pm 3 \sigma$)')

# Full vs distributed
compare_plot(m_phi_full, m_mix,
             a_err=1.96*np.sqrt(var_phi_full), b_err=1.96*np.sqrt(var_mix),
             a_label='Estimased from the full model ($\pm 1.96 \sigma$)',
             b_label='Estimased from the dep model ($\pm 1.96 \sigma$)')

plt.show()



