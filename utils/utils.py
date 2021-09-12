# -*- coding: utf-8 -*-
"""
Useful functions for central-spin problem
@author: Xiaoxu Zhou
Latest Update: 09/13/2021
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

def pauli():
    sigmax = qt.sigmax()
    sigmay = qt.sigmay()
    sigmaz = qt.sigmaz()
    return sigmax, sigmay, sigmaz

def make_basis():
    """
    Make basis vectors
    """
    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)
    ket00 = qt.tensor([ket0, ket0])
    ket01 = qt.tensor([ket0, ket1])
    ket10 = qt.tensor([ket1, ket0])
    ket11 = qt.tensor([ket1, ket1])
    sgqubit = np.array([ket0, ket1])
    twoqubit = np.array([ket00, ket01, ket10, ket11])
    return sgqubit, twoqubit

def tensor_power(mat, n):
    """
    Calculate a matrix to the nth power
    Args:
        mat: a matrix
        n: exponent
    """
    if n==0:
        res = 1
    elif n==1:
        res = mat
    elif isinstance(n, int):
        res = mat
        for _ in range(2, n+1):
            res = qt.tensor([res, mat])
    else:
        print('Invalid value of the exponent')
    return res

def sigmazi(i, N):
    """
    Calculate ith spin in the environment interacting with itself
    Args:
        N: the number of spins in the environment
    """
    if i==1:
        inter = qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), N-1)])
    elif i==N:
        inter = qt.tensor([tensor_power(qt.qeye(2), N-1), qt.sigmaz()])
    else:
        inter = qt.tensor([tensor_power(qt.qeye(2), i-1), qt.sigmaz()])
        inter = qt.tensor([inter, tensor_power(qt.qeye(2), N-i)])
    return inter

