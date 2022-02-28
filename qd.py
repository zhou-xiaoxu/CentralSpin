# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 02/23/2022
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from utils.utils import tensor_power, sigmaxi, sigmayi, sigmazi, qtrace


class CentralSpin(object):
    """
    Central spin model
    Index 0 is for central spin, index 1 to N are for the 1st to the Nth
    environment spin respectively
    """
    def __init__(self, params, c_init, env_init):
        self.params = params
        self.N = int(params['N'])
        self.omega = params['omega']
        self.lamb = params['lambda']
        
        self.T = int(params['T'])
        self.dt = int(params['dt'])
        self.tlist = np.arange(0,params['T']+params['dt'],params['dt'])
        
        if self.params["option"] == 'U':
            self.cops = []
        elif self.params['option'] == 'D':
            self.cops = []  # TBC. This project doesn't take 'D' into considerations.
        
        self.dim = np.power(2, self.N+1)
        self.c_init = c_init
        self.env_init = env_init
        self.s_tar = c_init  # target state
#        self.rho_init = qt.ket2dm(qt.tensor([c_init, env_init]))
        self.rho_init = qt.tensor([self.c_init, self.env_init])
        print("rho_init:", self.rho_init)
        print("self.dim:", self.dim)
#        self.rho_init = self.rho_init.data.reshape((self.dim, self.dim))
        
#        print("rho_init[0][3]:", self.rho_init[3][0])
#        print("rho_init dim:", self.rho_init.dims)
#        print(self.rho_init.dims[0][0])
       
    def evolve(self):
        """
        Calculate the evolution of a quantum system
        Args:
            c_init: initial central spin state
            env_init: initial environment state
        Paras:
            N: the number of spins in the environment
            omega: the list of Larmor frequencies in NMR system; the first is
            for central spin, others are for env spins respectively
            lamb: the list of coupling strengths between central and env spin
            cops: collapse operator
        """
        # environment term in Hamiltonian
#        dim_env = np.power(2, self.N)
        basis = qt.ket2dm(qt.Qobj(np.zeros((2, 1))))
#        print("basis:", basis)
#        Ham_env = qt.Qobj(np.zeros((dim_env, dim_env)))
        Ham_env = tensor_power(basis, self.N)
        print("Ham_env_init:", Ham_env)
#        qt.Qobj(Ham_env)
        for i in range(1,self.N+1):
#            Ham_env += 0.5 * self.omega[i] * sigmazi(i, self.N).data.reshape((dim_env,dim_env))
            Ham_env += 0.5 * self.omega[i] * sigmazi(i, self.N)
        print("Ham_env:", Ham_env)
        # interaction term in Hamiltonian
#        dim_int = np.power(2, self.N+1)
#        Ham_int = qt.Qobj(np.zeros((dim_int, dim_int)))
        Ham_int = tensor_power(basis, self.N+1)
        for i in range(1,self.N+1):
#            Ham_int += 0.5 * self.lamb[i-1] * \
#                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
#                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)])).data.reshape((dim_int,dim_int))
            Ham_int += 0.5 * self.lamb[i-1] * \
                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]))
        print("Ham_int:", Ham_int)
        # total Hamiltonian in interaction picture
#        Ham = qt.tensor([0.5 * self.omega[0] * qt.sigmaz(), tensor_power(qt.qeye(2), self.N)]).data.reshape((dim_int,dim_int)) + \
#              qt.tensor([qt.qeye(2), Ham_env]).data.reshape((dim_int,dim_int)) + \
#              Ham_int
        Ham = qt.tensor([0.5 * self.omega[0] * qt.sigmaz(), tensor_power(qt.qeye(2), self.N)]) + \
              qt.tensor([qt.qeye(2), Ham_env]) + \
              Ham_int
#        Ham = Ham_int
        print("Ham:", Ham)
        
        # evolve
        evol = qt.mesolve(Ham, self.rho_init, self.tlist, self.cops, [])
        state_list = []
        for i in range(0,self.N+1):
            state_list.append([[qt.ptrace(s,i)] for s in evol.states])  # s=state
#        print(state_list)
        
        return evol.states, state_list
        
    def fid(self, state_list):
        """
        Calculate fidelity
        """
#        fid = []
#        for i in range(0,self.N+1):
#            fid.append([np.trace(s[0] * self.s_tar) + 2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(self.s_tar)) for s in state_list[i]])
#            fid.append([np.trace(s[j] * self.s_tar) + 2 * np.sqrt(np.linalg.det(s[j]) * np.linalg.det(self.s_tar)) for s in state_list[i]])
#        fidx = []
#        sx = state_list[0]
#        for i in range(0,len(self.tlist)+1):
#            fidx.append(qtrace(sx * self.s_tar) + 2 * np.sqrt(np.linalg.det(sx)[0] * np.linalg.det(self.s_tar)[0]) for _ in sx)
#        fidx.append(qtrace(sx * self.s_tar) + 2 * np.sqrt(np.linalg.det(sx)[0] * np.linalg.det(self.s_tar)[0]) for _ in sx)
#        fidy = []
#        sy = state_list[1]
#        for i in range(0,len(self.tlist)+1):
#            fidy.append(np.trace(sy[i] * self.s_tar)[0] + 2 * np.sqrt(np.linalg.det(sy[i])[0] * np.linalg.det(self.s_tar)[0]))
#        fidz = []
#        sz = state_list[2]
#        for i in range(0,len(self.tlist)+1):
#            fidz.append(np.trace(sz[i] * self.s_tar)[0] + 2 * np.sqrt(np.linalg.det(sz[i])[0] * np.linalg.det(self.s_tar)[0]))
#        fid = [fidx, fidy, fidz]
        
        fid = []
        for i in range(0,self.N+1):
            fid.append([qtrace(s * self.s_tar) + 2 * np.sqrt(np.linalg.det(s) * np.linalg.det(self.s_tar)) for s in state_list[i]])
        
        return fid
    
    def expect(self, state_list):
        """
        Calculate expectation of observable sigma_x, sigma_y, sigma_z
        """
        exp_x = []
        exp_y = []
        exp_z = []
        for i in range(0,self.N+1):
            exp_x.append([qt.expect(qt.sigmax(), [s[0] for s in state_list[i]])])
            exp_y.append([qt.expect(qt.sigmay(), [s[0] for s in state_list[i]])])
            exp_z.append([qt.expect(qt.sigmaz(), [s[0] for s in state_list[i]])])
        return exp_x, exp_y, exp_z


params = dict()
params = {
          "N": 2,
          "omega": [10,0.1,0.1,0.1,0.1,1.,1.,1.],
          "lambda": [0.1,0.12,0.14,0.16,0.18,0.1,0.1],
          "T": 2,
          "dt": 2e-3,
          "option": 'U'
          }

c_init = qt.ket2dm(qt.basis(2,0))
#c_init = qt.basis(2,0)
print("c_init:", c_init)
env_init = tensor_power(qt.ket2dm(qt.basis(2,1)), params['N'])  # alternative
#env_init = tensor_power(qt.basis(2,1), params['N'])
print("env_init:", env_init)
model = CentralSpin(params, c_init, env_init)
states, state_list = model.evolve()
print(("states:"), len(states))
fid = model.fid(state_list)
exp_x, exp_y, exp_z = model.expect(state_list)

count = model.tlist

# plot fidelity
fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(count, fid[0], label='central spin')
#for i in range(1,int(params['N'])+1):
#    ax.plot(count, fid[i], label='bath spin %d'%i)
ax.plot(count, fid[1], label='bath spin')
ax.legend(fontsize=9)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel('fidelity', fontsize=12)
ax.set_title(r'$F$-t, N=%d'%params['N'], fontsize=16)

# plot expectation
fig, ax = plt.subplots(figsize=(8,6))
# sigmax
#ax.plot(count, exp_x[0][0], label=r'$\langle \sigma_x \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_x[i][0].T, label=r'$\langle \sigma_x \rangle$ on bath spin %d'%i)
# sigmay
#ax.plot(count, exp_y[0][0], label=r'$\langle \sigma_y \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_y[i][0], label=r'$\langle \sigma_y \rangle$ on bath spin %d'%i)
#ax.plot(count, exp_x[0][0], label=r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin')
# sigmaz
ax.plot(count, exp_z[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_z[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#ax.plot(count, exp_z[1][0], label=r'$\langle \sigma_z \rangle$ on each bath spin')

ax.legend(fontsize=9)
ax.set_xlabel('t', fontsize=12)
ax.set_ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#ax.set_ylim((0.98,1.02))
ax.set_title(r'$\langle \sigma_i \rangle$-t, N=%d'%params['N'], fontsize=16)


#plt.figure(figsize=(8,6))
#plt.plot(count, exp_x[0][0], label=r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin')
#plt.plot(count, exp_z[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    plt.plot(count, exp_z[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#plt.xlabel('t', fontsize=12)
#plt.ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#plt.title(r'$\langle \sigma_i \rangle$-t, N=%d'%params['N'], fontsize=16)
