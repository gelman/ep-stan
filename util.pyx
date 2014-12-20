"""This module contains some Cython utilities."""

# Licensed under the 3-clause BSD license.
# http://opensource.org/licenses/BSD-3-Clause
#
# Copyright (C) 2014 Tuomas Sivula
# All rights reserved.

import numpy as np
cimport cython
cimport numpy as np
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def copy_triu_to_tril(np.ndarray[DTYPE_t, ndim=2] A):
    """Copy upper triangular into the lower triangular.
    
    Parameters
    ----------
    A : ndarray
        The array to operate on. It has to be a square matrix of
        type np.float64.
    
    """
    assert A.dtype == np.float64
    cdef int n = A.shape[0]
    assert n == A.shape[1]
    cdef unsigned int x, y
    for x in range(n-1):
        for y in range(x+1,n):
            A[y,x] = A[x,y]

