#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import sys


# compile the extension (we assume CUDA is already installed)
makefile_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "cudamat")
if os.system("make -C '%s' cudamat" % makefile_dir) != 0:
    print("Failed to compile shared cudamat libraries.  Can't continue.")
    sys.exit(1)

setup(name="cudamat",
      version="0.3",
      description="Performs linear algebra computation on the GPU via CUDA",
      packages=find_packages(exclude=['examples', 'test']),
      include_package_data=True,
      package_data={'cudamat': ['libcudamat.so', 'libcudalearn.so',
                                'rnd_multipliers_32bit.txt']},
      author="Volodymyr Mnih",
      url="https://github.com/cudamat/cudamat",
)
