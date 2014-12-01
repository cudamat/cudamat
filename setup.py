#!/usr/bin/env python

import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys

# CUDA specific config
# nvcc is assumed to be in user's PATH
nvcc_compile_args = ['-O', '--ptxas-options=-v', '--compiler-options',
                     "'-fPIC'"]
cuda_libs = ['cublas']

cudamat_ext = Extension('cudamat.libcudamat',
                        sources=['cudamat/cudamat.cu',
                                 'cudamat/cudamat_kernels.cu'],
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args)
cudalearn_ext = Extension('cudamat.libcudalearn',
                          sources=['cudamat/learn.cu',
                                   'cudamat/learn_kernels.cu'],
                          libraries=cuda_libs,
                          extra_compile_args=nvcc_compile_args)


class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        build_ext.build_extensions(self)

setup(name="cudamat",
      version="0.3",
      description="Performs linear algebra computation on the GPU via CUDA",
      ext_modules=[cudamat_ext, cudalearn_ext],
      packages=find_packages(exclude=['examples', 'test']),
      include_package_data=True,
      package_data={'cudamat': ['rnd_multipliers_32bit.txt']},
      author="Volodymyr Mnih",
      url="https://github.com/cudamat/cudamat",
      cmdclass={'build_ext': CUDA_build_ext})
