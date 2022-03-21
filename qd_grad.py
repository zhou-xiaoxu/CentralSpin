# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 03/21/2022
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from utils.utils import tensor_power, sigmaxi, sigmayi, sigmazi, fidm


class CentralSpin(object):
    """
    Central spin model
    Index 0 is for central spin, index 1 to N are for the 1st to the Nth
    environment spin respectively
    """
    def __init__(self, params, omega0, c_init, env_init):
        self.params = params
        self.N = int(params['N'])
        self.omega = params['omega']
        self.omega[0] = omega0
        self.A = params['A']
        self.n = int(params['n'])
        
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
        Ham_e = 0.5 * self.omega[0] * qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), self.N)])

        # environment term in Hamiltonian
        Ham_env = zeroN
        for i in range(1,self.N+1):
            Ham_env += 0.5 * self.omega[i] * sigmazi(i, self.N)
        Ham_env = qt.tensor([qt.qeye(2), Ham_env])
        
        # interaction term in Hamiltonian
        Ham_int = zeroN1
        for i in range(1,self.N+1):
            Ham_int += 1/4 * self.A[i-1] * \
                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]))
        
        # total Hamiltonian in interaction picture
        Ham = Ham_e + Ham_env + Ham_int
        
        # Hamiltonian in rotating frame
        Ham_r = zeroN1
        for i in range(1,self.N+1):
            Ham_r += 1/4 * self.A[i-1] * \
                         (np.cos(self.omega[0]-self.omega[i]) * (sigmax0*sigmaxi(i,self.N+1) + sigmay0*sigmayi(i,self.N+1)) + \
                          np.sin(self.omega[0]-self.omega[i]) * (sigmax0*sigmayi(i,self.N+1) - sigmay0*sigmaxi(i,self.N+1)))
        
        # target Hamiltonian
        Ham_n = 0.5 * self.omega[self.n] * sigmazi(self.n, self.N)
        Ham_n = qt.tensor([qt.qeye(2), Ham_n])
        Ham_tar = Ham_e + Ham_n + \
                  1/4 * self.A[self.n-1] * \
                  (qt.tensor([qt.sigmax(), sigmaxi(self.n, self.N)]) + \
                   qt.tensor([qt.sigmay(), sigmayi(self.n, self.N)]))
        
        return Ham, Ham_r, Ham_tar
        
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
          "c": [0,1],
          "omega": [2500*1e3,10000*1e3,8500*1e3,7000*1e3,5500*1e3,4000*1e3,2500*1e3,1000*1e3],
          "A": [1.20*1e6,1.18*1e6,1.16*1e6,1.14*1e6,1.12*1e6,1.10*1e6,1.08*1e6],
          "n": 1,
          "T": 40e-6,
          "dt": 40e-9,
          "option": 'U'
          }

c_init = qt.ket2dm(params['c'][0]*qt.basis(2,0)+params['c'][1]*qt.basis(2,1))
env_init = tensor_power(qt.ket2dm(qt.basis(2,0)), params['N'])  # (2,0) is ground state

find='2'  # 1 for changing initial electron state, 2 for finding omega0

if find=='1':
    model = CentralSpin(params, params['omega'][0], c_init, env_init)
    Ham, Ham_r, Ham_tar = model.ham()
    states, state_list = model.evolve(Ham)
    fid = model.fid(state_list)
    fidm = fidm(Ham, Ham_tar)
    print("fidm:", fidm)
    
    # plot fidelity
    count = params['omega'][0] * model.tlist
    fig = plt.figure(figsize=(10,6))
    l1, = plt.plot(count, fid[0])
    l2, = plt.plot(count, fid[1])
    l3, = plt.plot(count, fid[2])
    l4, = plt.plot(count, fid[3])
    l5, = plt.plot(count, fid[4])
#    l6, = plt.plot(count, fid[5])
#    l7, = plt.plot(count, fid[6])
#    l8, = plt.plot(count, fid[7])

#    plt.legend(handles=[l1, l2, l3, l4, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3'], 
#               loc='center right', fontsize=16)
    plt.legend(handles=[l1, l2, l3, l4, l5, ], 
               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
                       'nucleus 4'], 
               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', 'nucleus 6', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', 'nucleus 6', 'nucleus 7'], 
#               loc='center right', fontsize=16)    
    
    plt.xlabel(r'$\omega t$', fontsize=16)
    plt.ylabel('fidelity', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('$F-t$', fontsize=20)

elif find=='2':
    omega0_list = np.arange(0.1*1e6, 11.1*1e6, 0.1*1e6)
    fidm_e, fidm_1, fidm_2, fidm_3, fidm_4 = [], [], [], [], []  # fid max
    fidmw_e, fidmw_1, fidmw_2, fidmw_3, fidmw_4 = 2.0, 2.0, 2.0, 2.0, 2.0  # omega0 giving fid max
    for omega0 in omega0_list:
        model = CentralSpin(params, omega0, c_init, env_init)
        Ham, Ham_r, Ham_tar = model.ham()
        states, state_list = model.evolve(Ham)
        fid = model.fid(state_list)
        
        fidm_e.append(max(fid[0]))
        fidm_1.append(max(fid[1]))
        fidm_2.append(max(fid[2]))
        fidm_3.append(max(fid[3]))
        fidm_4.append(max(fid[4]))
#        fidm_5.append(max(fid[5]))
#        fidm_6.append(max(fid[6]))
#        fidm_7.append(max(fid[7]))
        
        if len(fidm_e)>=2:
            if fidm_e[-1]>max(fidm_e[:-1]):
                fidmw_e = omega0 * 1e-6
            if fidm_1[-1]>max(fidm_1[:-1]):
                fidmw_1 = omega0 * 1e-6
            if fidm_2[-1]>max(fidm_2[:-1]):
                fidmw_2 = omega0 * 1e-6
            if fidm_3[-1]>max(fidm_3[:-1]):
                fidmw_3 = omega0 * 1e-6
            if fidm_4[-1]>max(fidm_4[:-1]):
                fidmw_4 = omega0 * 1e-6
#            if fidm_5[-1]>max(fidm_5[:-1]):
#                fidmw_5 = omega0 * 1e-6
#            if fidm_6[-1]>max(fidm_6[:-1]):
#                fidmw_6 = omega0 * 1e-6
#            if fidm_7[-1]>max(fidm_7[:-1]):
#                fidmw_7 = omega0 * 1e-6
        
        #exp_x, exp_y, exp_z = model.expect(state_list)
        
        # plot fidelity
        count = omega0 * model.tlist
        
    #    fig, ax = plt.subplots(figsize=(8,6))
    #    ax.plot(count, fid[0], label='electron')
    #    for i in range(1,int(params['N'])+1):
    #        ax.plot(count, fid[i], label='nucleus %d'%i)
    #    ax.legend(fontsize=16, loc='upper right')
    #    ax.set_xlabel(r'$\omega t$', fontsize=16)
    #    ax.set_ylabel('fidelity', fontsize=16)
    #    ax.tick_params(labelsize=12)
    #    ax.set_title(r'$ F-t, \omega_0=%.1f \times 10^6 rad/s$'%(omega0*1e-6), fontsize=18)
        
        fig = plt.figure(figsize=(8,6))
        l1, = plt.plot(count, fid[0])
        l2, = plt.plot(count, fid[1])
        l3, = plt.plot(count, fid[2])
        l4, = plt.plot(count, fid[3])
        l5, = plt.plot(count, fid[4])
#        l6, = plt.plot(count, fid[5])
#        l7, = plt.plot(count, fid[6])
#        l8, = plt.plot(count, fid[7])
        
#        plt.legend(handles=[l1, l2, l3, l4, ], 
#                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3'], 
#                   loc='center right', fontsize=16)
        plt.legend(handles=[l1, l2, l3, l4, l5, ], 
                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
                           'nucleus 4', ], 
                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                           'nucleus 4', 'nucleus 5', ], 
#                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                           'nucleus 4', 'nucleus 5', 'nucleus 6', ], 
#                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
#                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                           'nucleus 4', 'nucleus 5', 'nucleus 6', 'nucleus 7'], 
#                   loc='center right', fontsize=16)        
        
        plt.xlabel(r'$\omega t$', fontsize=16)
        plt.ylabel('fidelity', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(r'$ F-t, \omega_0=%.1f \times 10^6 rad/s$'%(omega0*1e-6), fontsize=20)
        plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=4\%.1f.png'%(omega0*1e-6))  #改N
    
    
    print("fidmw:", fidmw_e, fidmw_1, fidmw_2, fidmw_3, fidmw_4)
    
    omega0_list_ = [i*1e-6 for i in omega0_list]
    fig = plt.figure(figsize=(8,6))
    l1, = plt.plot(omega0_list_, fidm_e)
    l2, = plt.plot(omega0_list_, fidm_1)
    l3, = plt.plot(omega0_list_, fidm_2)
    l4, = plt.plot(omega0_list_, fidm_3)
    l5, = plt.plot(omega0_list_, fidm_4)
#    l6, = plt.plot(omega0_list_, fidm_5)
#    l7, = plt.plot(omega0_list_, fidm_6)
#    l8, = plt.plot(omega0_list_, fidm_7)
    
#    plt.legend(handles=[l1, l2, l3, l4, ], 
#                   labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3'], 
#                   loc='upper right', fontsize=16)
    plt.legend(handles=[l1, l2, l3, l4, l5, ],
              labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
                      'nucleus 4', ], 
              loc='upper right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', ], 
#               loc='upper right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', 'nucleus 6', ], 
#               loc='upper right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
#               labels=['electron', 'nucleus 1', 'nucleus 2', 'nucleus 3', 
#                       'nucleus 4', 'nucleus 5', 'nucleus 6', 'nucleus 7'], 
#               loc='upper right', fontsize=16)  
    
    plt.xlabel('$\omega_0 (*10^6 rad/s)$', fontsize=16)
    plt.ylabel('Maximal fidelity', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Maximal fidelity $- \omega_0$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=4\fidmax.png')

