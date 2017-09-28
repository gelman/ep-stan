"""Compile the Cython module in file `epstan/cython_util.pyx`.

Compile with:
    $ python setup.py build_ext --inplace

"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("epstan/cython_util.pyx")
)
