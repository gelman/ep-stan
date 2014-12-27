"""Compile Cython module in file `cython_util.pyx`.

Compile with:
$ python cython_setup.py build_ext --inplace

"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_util.pyx")
)

