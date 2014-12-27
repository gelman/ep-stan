
Python code supplement for "Expectation propagation as a way of life"
---------------------------------------------------------------------

### Requirements (tested version):
- python (2.7.6)
- numpy (1.9.1)
- scipy (0.14.0)
- cython (0.21.1)
- matplotlib (1.4.0)
- pystan (2.5.0.0)

### Compiling
- `python cython_setup.py build_ext --inplace`: for Cython utilities
- `python compile_stan.py`: for Stan model

### Usage
After compiling the Stan model and Cython utilities, run
`python experiment.py` for a simple hierarchical logistic regression example.

See experiment.py and class DistributedEP in distributed_ep.py for more
documentation.

### License
[Released under the 3-clause BSD license.](http://opensource.org/licenses/BSD-3-Clause)
 
