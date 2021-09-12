# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 09/12/2021
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

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


class CentralSpin(object):
    """
    Central spin model
    """
    def __init__(self, params):
        self.params = params
        self.N = int(params['N'])
        self.omega = params['omega']
        self.lamb = params['lambda']
        
        self.T = int(params['T'])
        self.step = int(params['step'])
        self.tlist = np.linspace(0, self.T, self.step+1)
        
        if self.params["option"] == 'U':
            self.cops = []
        elif self.params['option'] == 'D':
            self.cops = []  # to be completed
        
    def make_basis(self):
        self.ket0 = qt.basis(2, 0)
        self.ket1 = qt.basis(2, 1)
        self.ket00 = qt.tensor([self.ket0, self.ket0])
        self.ket01 = qt.tensor([self.ket0, self.ket1])
        self.ket10 = qt.tensor([self.ket1, self.ket0])
        self.ket11 = qt.tensor([self.ket1, self.ket1])
       
    def nmr(self):
        """
        Args:
            N: the number of spins in the environment
            omega: the list of Larmor frequencies in NMR system; the first is
            for central spin, others are for env spins respectively
            lamb: the list of coupling strengths between central and env spin
            cops: collapse operator
        """
        c_init = (qt.qeye(2) + qt.sigmax()) / 2
        env_init = tensor_power(qt.qeye(2)/2, self.N)
        rho_init = qt.tensor([c_init, env_init])
        
        dim_env = np.power(2, self.N)
        Ham_env = qt.Qobj(np.zeros((dim_env, dim_env)))
#        qt.Qobj(Ham_env)
        for i in range(1,self.N+1):
            Ham_env += 0.5 * self.omega[i] * sigmazi(i, self.N).data.reshape((dim_env,dim_env))
        
        dim_int = np.power(2, self.N+1)
        Ham_int = qt.Qobj(np.zeros((dim_int, dim_int)))
        for i in range(1,self.N+1):
            Ham_int += 0.5 * self.lamb[i-1] * qt.tensor([qt.sigmaz(), sigmazi(i, self.N)]).data.reshape((dim_int,dim_int))
        
        Ham = qt.tensor([0.5 * self.omega[0] * qt.sigmaz(), tensor_power(qt.qeye(2), self.N)]).data.reshape((dim_int,dim_int)) + \
              qt.tensor([qt.sigmaz(), Ham_env]).data.reshape((dim_int,dim_int)) + \
              Ham_int
        
        evol = qt.mesolve(Ham, rho_init, self.tlist, self.cops)
        fid = [(np.trace(np.sqrt(np.sqrt(rho_init) * state * np.sqrt(rho_init))))**2 for state in evol.states]
        
        return fid
 
params = dict()
params = {
          "N": 2,
          "omega": [1.,1.,1.],
          "lambda": [0.1,0.1],
          "T": 10,
          "step": 1e3,
          "option": 'U'
          }

model = CentralSpin(params)
fid = model.nmr()
count = np.arange(0,len(fid),1)

plt.figure()
l1, = plt.plot(count, fid)
plt.xlabel(r'time steps')
plt.ylabel(r'fidelity')
plt.legend(handles=[l1, ], labels=['N=%d'%params['N']], loc='best')
plt.title(r'Central spin model')
plt.show()

    