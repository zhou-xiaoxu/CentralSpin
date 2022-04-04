# -*- coding: utf-8 -*-
"""
Useful functions for selective transformation problem
@author: Xiaoxu Zhou
Latest update: 04/05/2022
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
#            print("res:", res)
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

def qtrace(mat):
    """
    ----03/12/2022 update: Could be replaced by Q.tr()----
    Calculate the trace of a 2*2 matrix in Qobj
    Args:
        mat: matrix to be calculated
    """
    tr11 = mat[0][0][0][0]
    tr22 = mat[0][1][0][1]
    tr = tr11 + tr22
    return tr
    
def fid_spin(tar, state):
    return (state*tar).tr() + 2*np.sqrt(np.linalg.det(state) * np.linalg.det(tar))

def fid_gen(tar, state):
    return np.square(np.absolute((tar.sqrtm() * state * tar.sqrtm()).sqrtm().tr()))

def oper_norm(F):
    """
    Normalize an operator
    """
    norm = F / np.sqrt(np.trace(F * qt.dag(F)))
    return norm

def fidm(A, B):
    """
    Calculate fidelity between operators A and B
    """
    A_n = oper_norm(A)  # normalize operators
    B_n = oper_norm(B)
    deno = np.sqrt(np.trace(A_n*qt.dag(A_n)) * np.trace(B_n*qt.dag(B_n)))
    fid = np.abs(np.trace(qt.dag(A_n)*B_n)) / deno
    
#    fid = np.trace(qt.dag(A) * B)
    
    return fid

def fidmu(A, B, d):
    """
    Calculate fidelity between unitary operators A and B
    """
#    A_n = oper_norm(A)  # normalize operators
#    B_n = oper_norm(B)
#    fid = np.abs(np.trace(qt.dag(A_n) * B_n))
    
    fid = np.abs(np.trace(qt.dag(A) * B)) / d
    
    return fid
