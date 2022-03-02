# -*- coding: utf-8 -*-
"""
Useful functions for central-spin problem
@author: Xiaoxu Zhou
Latest update: 03/02/2022
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt


def electron_basis():
    """
    Make basis vectors for electron spins
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

def nucleus_basis():
    """
    Make basis vectors for spin-3/2 nuclei
    """
    ket0 = qt.basis(4, 0)
    ket1 = qt.basis(4, 1)
    ket2 = qt.basis(4, 1)
    ket3 = qt.basis(4, 3)
    ket00 = qt.tensor([ket0, ket0])
    ket01 = qt.tensor([ket0, ket1])
    ket02 = qt.tensor([ket0, ket2])
    ket03 = qt.tensor([ket0, ket3])
    ket10 = qt.tensor([ket1, ket0])
    ket11 = qt.tensor([ket1, ket1])
    ket12 = qt.tensor([ket1, ket2])
    ket13 = qt.tensor([ket1, ket3])
    ket20 = qt.tensor([ket2, ket0])
    ket21 = qt.tensor([ket2, ket1])
    ket22 = qt.tensor([ket2, ket2])
    ket23 = qt.tensor([ket2, ket3])
    ket30 = qt.tensor([ket3, ket0])
    ket31 = qt.tensor([ket3, ket1])
    ket32 = qt.tensor([ket3, ket2])
    ket33 = qt.tensor([ket3, ket3])
    sgqubit = np.array([ket0, ket1, ket2, ket3])
    twoqubit = np.array([ket00, ket01, ket02, ket03, ket10, ket11, ket12, ket13, 
                         ket20, ket21, ket22, ket23, ket30, ket31, ket32, ket33])
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

def Ixi(i, N):
    """
    Calculate sigma_x operator acting on the ith spin in the environment
    where nuclear spins are spin-3/2
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.spin_Jx(3/2)
    elif i==1:
        res = qt.tensor([qt.spin_Jx(3/2), tensor_power(qt.qeye(4), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(4), N-1), qt.spin_Jx(3/2)])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(4), i-1), qt.spin_Jx(3/2)])
        res = qt.tensor([res1, tensor_power(qt.qeye(4), N-i)])
    return res

def Iyi(i, N):
    """
    Calculate sigma_y operator acting on the ith spin in the environment
    where nuclear spins are spin-3/2
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.spin_Jy(3/2)
    elif i==1:
        res = qt.tensor([qt.spin_Jy(3/2), tensor_power(qt.qeye(4), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(4), N-1), qt.spin_Jy(3/2)])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(4), i-1), qt.spin_Jy(3/2)])
        res = qt.tensor([res1, tensor_power(qt.qeye(4), N-i)])
    return res

def Izi(i, N):
    """
    Calculate sigma_z operator acting on the ith spin in the environment
    where nuclear spins are spin-3/2
    Args:
        N: the number of spins in the environment
    """
    if i==1 and N==1:
        res = qt.spin_Jz(3/2)
    elif i==1:
        res = qt.tensor([qt.spin_Jz(3/2), tensor_power(qt.qeye(4), N-1)])
    elif i==N:
        res = qt.tensor([tensor_power(qt.qeye(4), N-1), qt.spin_Jz(3/2)])
    else:
        res1 = qt.tensor([tensor_power(qt.qeye(4), i-1), qt.spin_Jz(3/2)])
        res = qt.tensor([res1, tensor_power(qt.qeye(4), N-i)])
    return res

def qtrace2(mat):
    """
    Calculate the trace of a 2*2 matrix in Qobj
    Args:
        mat: matrix to be calculated
    """
    tr00 = mat[0][0][0][0]
    tr11 = mat[0][1][0][1]
    tr = tr00 + tr11
    return tr

def qtrace4(mat):
    """
    Calculate the trace of a 4*4 matrix in Qobj
    Args:
        mat: matrix to be calculated
    """
    tr00 = mat[0][0][0][0]
    tr11 = mat[0][1][0][1]
    tr22 = mat[0][2][0][2]
    tr33 = mat[0][3][0][3]
    tr = tr00 + tr11 + tr22 + tr33
    return tr

