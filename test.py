""" A simple test for Scipy compatibility.

The program does not work correctly in some scipy builds because of an issue in 
the in-place operation of dpotri Lapack-routine with C- or F-order matrices. In 
some builds, the in-place operation works for F-order matrices but not for 
C-order matrices, whereas in some builds, it works the opposite way. The former 
behaviour is assumed in this program. This script test if this holds.

"""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.


import numpy as np
from scipy import linalg

# LAPACK positive definite inverse routine
dpotri_routine = linalg.get_lapack_funcs('potri')

print("A simple test for Scipy compatibility")

A = 2*np.eye(6).T
dpotri_routine(A, overwrite_c=True)
if np.sum(A - (1/2**2)*np.eye(6).T) > 1e-10:
    print("Test failed")
else:
    print("Test successful")
