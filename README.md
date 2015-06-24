
Python code supplement for "Expectation propagation as a way of life"
---------------------------------------------------------------------

### Requirements (tested version):
- python (2.7.6)
- numpy (1.9.1)
- scipy (0.14.0)
- cython (0.21.1)
- pystan (2.5.0.0)
- sklearn (0.14.1)
- matplotlib (1.4.0) (only for plotting the results)

### Setup
Compile the Cython utilities with `python setup.py build_ext --inplace`.

### Usage
The folder experiment contains simple hierarchical logistic regression examples.
See `python fit.py -h` or the respective module docstring for help. For more
information, see e.g. the class documentation of dep.method.Master.

### License
[Released under the 3-clause BSD license.](http://opensource.org/licenses/BSD-3-Clause)
 
### Notes
The program does not work correctly in some scipy builds because of an 
issue in the in-place operation of dpotri Lapack-routine with C- or F-order 
matrices. In some builds, the in-place operation works for F-order matrices but 
not for C-order matrices, whereas in some builds, it works the opposite way. 
The former behaviour is assumed in this program. This compatibility problem can
be tested by running a simple test script by `python test.py`.
