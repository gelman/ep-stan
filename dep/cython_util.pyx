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
@cython.wraparound(False)
def copy_triu_to_tril(np.ndarray[DTYPE_t, ndim=2] A):
    """Copy upper triangular into the lower triangular.
    
    Parameters
    ----------
    A : ndarray
        The array to operate on. It has to be a square matrix.
    
    Notes
    -----
    Works slightly faster with either C of F -contiguous arrays depending on the
    system and the size of the array. 
    
    """
    cdef Py_ssize_t n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Input array is not square")
    cdef Py_ssize_t x, y
    for x in range(n-1):
        for y in range(x+1,n):
            A[y,x] = A[x,y]


@cython.boundscheck(False)
@cython.wraparound(False)
def auto_outer(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] out):
    """Outer product with itself.
    
    Calculates the outer product of each row of `A` with itself. Each row of the
    two diensional output array contains the product of each combination of the
    elements in the corresponding row in the input array. The order of the
    combinations is the same as with np.triu_indices.
    
    Parameters
    ----------
    A : ndarray
        The input array of shape (n,d).
    
    out : ndarray
        Output array of shape (n,d'), where d' = d+1 choose 2 = d*(d+1)/2.
    
    """
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t d = A.shape[1]
    cdef Py_ssize_t d2
    # Check shapes
    if d % 2 == 0:
        d2 = d >> 1
        d2 *= d+1
    else:
        d2 = (d+1) >> 1
        d2 *= d
    if out.shape[0] != n or out.shape[1] != d2:
        raise ValueError("Shapes of `A` and `out` does not match")
    cdef Py_ssize_t x, y, z, c
    for z in range(n):
        c = 0
        for x in range(d):
            for y in range(x,d):
                out[z,c] = A[z,x] * A[z,y]
                c += 1



