# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 04/05/2022
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_module=cythonize('qd_grad.pyx'))

