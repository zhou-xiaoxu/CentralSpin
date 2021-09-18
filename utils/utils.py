# -*- coding: utf-8 -*-
"""
Useful functions for central-spin problem
@author: Xiaoxu Zhou
Latest update: 09/18/2021
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

def sigmaxi(i, N):
    """
    Calculate sigma_x operator acting on the ith spin in the environment
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.sigmax()
    elif i==1:
        res = qt.tensor([qt.sigmax(), tensor_power(qt.qeye(2), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(2), N-1), qt.sigmax()])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(2), i-1), qt.sigmax()])
        res = qt.tensor([res1, tensor_power(qt.qeye(2), N-i)])
    return res

def sigmayi(i, N):
    """
    Calculate sigma_y operator acting on the ith spin in the environment
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.sigmay()
    elif i==1:
        res = qt.tensor([qt.sigmay(), tensor_power(qt.qeye(2), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(2), N-1), qt.sigmay()])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(2), i-1), qt.sigmay()])
        res = qt.tensor([res1, tensor_power(qt.qeye(2), N-i)])
    return res

def sigmazi(i, N):
    """
    Calculate sigma_z operator acting on the ith spin in the environment
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.sigmaz()
    elif i==1:
        res = qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(2), N-1), qt.sigmaz()])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(2), i-1), qt.sigmaz()])
        res = qt.tensor([res1, tensor_power(qt.qeye(2), N-i)])
    return res

