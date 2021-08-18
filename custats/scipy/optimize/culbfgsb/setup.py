#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension
from Cython.Build import cythonize
import os


include_dirs = ['/usr/local/cuda/samples/common/inc', 'build',
                '/usr/local/cuda/include/']
path = os.path.abspath(os.path.dirname(__file__))
library_dirs = [path+'/build/culbfgsb', '/usr/local/cuda/lib64']
libraries = ['cuLBFGSB', 'cudart']


ext = cythonize([
    Extension(
        'bfgs',
        sources=['bfgs.pyx', 'call_obj.pyx', 'driver.cpp'],
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        # extra_compile_args={'nvcc': ['-c', '--compiler-options']},
        extra_compile_args=["-std=c++11"]
    )])


setup(
    name='python-culbfgsb',
    description='python extension for cuLBFGSB',
    author='Yi Dong',
    version='0.1',
    ext_modules=ext,
    zip_safe=False,
)

