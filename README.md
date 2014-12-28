
Python code supplement for "Expectation propagation as a way of life"
---------------------------------------------------------------------

### Requirements (tested version):
- python (2.7.6)
- numpy (1.9.1)
- scipy (0.14.0)
- cython (0.21.1)
- matplotlib (1.4.0)
- pystan (2.5.0.0)

### Setup
Compile the Cython utilities with `python setup.py build_ext --inplace`.

### Usage
The script run.py in the directory experiment contains a simple hierarchical
logistic regression example.

See the script experiment/run.py and class documentation of dep.serial.Master
for more information.

### License
[Released under the 3-clause BSD license.](http://opensource.org/licenses/BSD-3-Clause)
 
