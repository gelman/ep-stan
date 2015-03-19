
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
 
