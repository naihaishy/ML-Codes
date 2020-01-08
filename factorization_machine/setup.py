# -*- coding:utf-8 -*-
# @Time : 2019/12/30 19:23
# @Author : naihai

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension('fm_fast', ['fm_fast.pyx'], include_dirs=[np.get_include()]), ]

ext_modules = cythonize(extensions, annotate=True)

setup(
    name="fm",
    description='factorization machine in cython',
    long_description='factorization machine in cython, speed up train',
    author='Naihai',
    url='http://github.com/naihaisky/',
    author_email='open@zhfsky.com',
    license='MIT',
    ext_modules=ext_modules,
    requires=['Cython', 'numpy']
)

# 使用  python setup.py build_ext --inplace
