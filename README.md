CUDAMat
=======

The aim of the cudamat project is to make it easy to perform basic matrix calculations on CUDA-enabled GPUs from Python. cudamat provides a Python matrix class that performs calculations on a GPU. At present, some of the operations our GPU matrix class supports include:

* Easy conversion to and from instances of `numpy.ndarray`.
* Limited slicing support.
* Matrix multiplication and transpose.
* Elementwise addition, subtraction, multiplication, and division.
* Elementwise application of exp, log, pow, sqrt.
* Summation along rows or columns.
* Conversion of CUDA errors into Python exceptions.

The current feature set of cudamat is biased towards features needed for implementing some common machine learning algorithms. We have included implementations of feedforward neural networks and restricted Boltzmann machines in the examples that come with cudamat.

Example:

```python
import numpy as np
import cudamat as cm

cm.cublas_init()

# create two random matrices and copy them to the GPU
a = cm.CUDAMatrix(np.random.rand(32, 256))
b = cm.CUDAMatrix(np.random.rand(256, 32))

# perform calculations on the GPU
c = cm.dot(a, b)
d = c.sum(axis = 0)

# copy d back to the host (CPU) and print
print d.asarray()
```

Documentation
-------------

An overview of the main features of cudamat can be found in the technical report:

[CUDAMat: A CUDA-based matrix class for Python](http://www.cs.toronto.edu/~vmnih/docs/cudamat_tr.pdf), Volodymyr Mnih, UTML TR 2009-004.

Download
--------

You can obtain the latest release from the repository by typing:

```bash
git clone https://github.com/cudamat/cudamat.git
```

You can also download one of the releases from the [releases](https://github.com/cudamat/cudamat/releases) section.

cudamat has the following prerequisites:

* CUDA
* Numpy
* nose

