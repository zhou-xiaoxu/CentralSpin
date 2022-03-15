# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 03/15/2022
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from utils.utils import tensor_power, sigmaxi, sigmayi, sigmazi


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
        self.A = params['A']
        
        self.T = params['T']
        self.dt = params['dt']
        self.tlist = np.arange(0,params['T']+params['dt'],params['dt'])
        
        if self.params["option"] == 'U':
            self.cops = []
        elif self.params['option'] == 'D':
            self.cops = []  # TBC. This project doesn't take 'D' into considerations.
        
        self.c_init = c_init
        self.env_init = env_init
        self.c_tar = c_init  # target electron spin state
        self.env_tar = c_init
        self.rho_init = qt.tensor([self.c_init, self.env_init])
       
    def ham(self):
        """
        Hamiltonian terms of the system
        Args:
            c_init: initial central spin state
            env_init: initial environment state
        Paras:
            N: the number of spins in the environment
            omega: the list of Larmor frequencies in NMR system; the first is
            for central spin, others are for env spins respectively
            A: the list of coupling strengths between central and env spin
        """
        zero = qt.ket2dm(qt.Qobj(np.zeros((2, 1))))
        zeroN = tensor_power(zero, self.N)
        zeroN1 = tensor_power(zero, self.N+1)
        sigmax0 = qt.tensor([qt.sigmax(), tensor_power(qt.qeye(2), self.N)])
        sigmay0 = qt.tensor([qt.sigmay(), tensor_power(qt.qeye(2), self.N)])
        sigmaz0 = qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), self.N)])
        
        # electron term in Hamiltonian
        Ham_e = 0.5 * self.omega[0] * sigmaz0
        
        # environment term in Hamiltonian
        Ham_env = zeroN
        for i in range(1,self.N+1):
            Ham_env += 0.5 * self.omega[i] * sigmazi(i, self.N)
        Ham_env = qt.tensor([qt.qeye(2), Ham_env])
        
        # interaction term in Hamiltonian
        Ham_int = zeroN1
        Ham_int_r = zeroN1
        for i in range(1,self.N+1):
            Ham_int += 1/4 * self.A[i-1] * \
                       (sigmax0 * sigmaxi(i, self.N+1) + \
                        sigmay0 * sigmayi(i, self.N+1))
            Ham_int_r += 1/4 * self.A[i-1] * \
                         (np.cos(self.omega[0]-self.omega[i]) * (sigmax0*sigmaxi(i,self.N+1) + sigmay0*sigmayi(i,self.N+1)) + \
                          np.sin(self.omega[0]-self.omega[i]) * (sigmax0*sigmayi(i,self.N+1) - sigmay0*sigmaxi(i,self.N+1)))
        
        # total Hamiltonian in interaction picture
        Ham = Ham_e + Ham_env + Ham_int
        Ham_r = Ham_int_r
        
        return Ham, Ham_r
        
    def evolve(self, Ham):
        """
        Evolution under a Hamiltonian
        Args:
            Ham: Hamiltonian
            seg: time segment of a period T
        """
        # evolve
        evol = qt.mesolve(Ham, self.rho_init, self.tlist, self.cops, [])
        state_list = []
        for i in range(0,self.N+1):
            state_list.append([[qt.ptrace(s,i)] for s in evol.states])  # s=state
        
        return evol.states, state_list
        
    def fid(self, state_list):
        """
        Calculate fidelity
        """
        fid = []  # fidelity especially for spin system
        fid.append([(s[0] * self.c_tar).tr() + \
                    2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(self.c_tar)) \
                    for s in state_list[0]])
        for i in range(1,self.N+1):
            fid.append([(s[0] * self.env_tar).tr() + \
                        2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(self.env_tar)) \
                        for s in state_list[i]])
        
        fid2 = []  # general form of fidelity
        fid2.append([np.square(np.absolute((self.c_tar.sqrtm() * s[0] * self.c_tar.sqrtm()).sqrtm().tr())) \
                     for s in state_list[0]])
        for i in range(1,self.N+1):
            fid2.append([np.square(np.absolute((self.env_tar.sqrtm() * s[0] * self.env_tar.sqrtm()).sqrtm().tr())) \
                         for s in state_list[i]])
        
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
          "N": 4,
          "omega": [4.6*1e6,4.6*1e6,2.9*1e6,1.2*1e6,1e6,1e6,1e6],
          "A": [1.20*1e6,1.18*1e6,1.15*1e6,1e6,1e6,1e6,1e6],
          "T": 8e-6,
          "dt": 8e-8,
          "option": 'U'
          }

c_init = qt.ket2dm(qt.basis(2, 0))
env_init = tensor_power(qt.ket2dm(qt.basis(2, 1)), params['N'])  # alternative
model = CentralSpin(params, c_init, env_init)
Ham, Ham_r = model.ham()
states, state_list = model.evolve(Ham)
fid = model.fid(state_list)
exp_x, exp_y, exp_z = model.expect(state_list)

count = params['omega'][0] * model.tlist

# plot fidelity
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(count, fid[0], label='central spin')
for i in range(1,int(params['N'])+1):
    ax.plot(count, fid[i], label='bath spin %d'%i)
ax.legend(fontsize=16, loc='upper right')
ax.set_xlabel(r'$\omega t$', fontsize=16)
ax.set_ylabel('fidelity', fontsize=16)
ax.set_title(r'$ F-t, N=%d $'%params['N'], fontsize=18)


# plot expectation
#fig, ax = plt.subplots(figsize=(8,6))
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
#ax.plot(count, exp_z[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_z[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#ax.plot(count, exp_z[1][0], label=r'$\langle \sigma_z \rangle$ on each bath spin')

#ax.legend(fontsize=9, loc='upper right')
#ax.set_xlabel(r'$\omega t$', fontsize=12)
#ax.set_ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#ax.set_title(r'$ \langle \sigma_i \rangle-t, N=%d $'%params['N'], fontsize=16)


#plt.figure(figsize=(8,6))
#plt.plot(count, exp_x[0][0], label=r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin')
#plt.plot(count, exp_z[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    plt.plot(count, exp_z[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#plt.xlabel(r'$t$', fontsize=12)
#plt.ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#plt.title(r'$ \langle \sigma_i \rangle-t, N=%d $'%params['N'], fontsize=16)
