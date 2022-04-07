# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 04/07/2022
"""

import numpy as np
import qutip as qt
from scipy.linalg import logm
import random
from itertools import product

import matplotlib.pyplot as plt

from utils.utils import bi2basis, tensor_power, sigmaxi, sigmayi, sigmazi, fid_spin, fid_gen, fidm, fidmu


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
#            ## neglect zz interaction term
            Ham_int += 1/4 * self.A[i-1] * \
                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]))
#            ## full form
#            Ham_int += 1/4 * self.A[i-1] * \
#                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
#                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]) + \
#                        qt.tensor([qt.sigmaz(), sigmazi(i, self.N)]))
        
        # total Hamiltonian in interaction picture
        Ham = Ham_e + Ham_env + Ham_int
        
        # Hamiltonian in rotating frame
        Ham_r = zeroN1
        for i in range(1,self.N+1):
            Ham_r += 1/4 * self.A[i-1] * \
                         (np.cos(self.omega[0]-self.omega[i]) * (sigmax0*sigmaxi(i,self.N+1) + sigmay0*sigmayi(i,self.N+1)) + \
                          np.sin(self.omega[0]-self.omega[i]) * (sigmax0*sigmayi(i,self.N+1) - sigmay0*sigmaxi(i,self.N+1)))
        
        # target Hamiltonian
#        Ham_e_t = 0.5 * self.omega[0] * sigmazi(1,2)
        Ham_n = 0.5 * self.omega[self.n] * sigmazi(self.n, self.N)
        Ham_n = qt.tensor([qt.qeye(2), Ham_n])
#        Ham_n = qt.tensor([qt.qeye(2), Ham_n])
        Ham_tar = Ham_e + Ham_n + \
                  1/4 * self.A[self.n-1] * \
                  (qt.tensor([qt.sigmax(), sigmaxi(self.n, self.N)]) + \
                   qt.tensor([qt.sigmay(), sigmayi(self.n, self.N)]))
#        Ham_tar = 0.5 * self.omega[0] * sigmazi(1,2) + \
#                  0.5 * self.omega[self.n] * sigmazi(2,2) + \
#                  1/4 * self.A[self.n-1] * \
#                  (qt.tensor([qt.sigmax(), qt.sigmax()]) + \
#                   qt.tensor([qt.sigmay(), qt.sigmay()]))
        
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
#        append = state_list.append
        for i in range(0,self.N+1):
            state_list.append([qt.ptrace(s,i) for s in evol.states])  # s=state
        
#        print(state_list)
        
        return evol.states, state_list
    
    def fid(self, state_list):
        """
        Calculate fidelity
        """
        tar = [self.c_tar for i in range(len(self.tlist))]
        
        # fidelity especially for spin system
        fid = []
#        append = fid.append
#        fid.append([(s[0] * self.c_tar).tr() + \
#                    2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(self.c_tar)) \
#                    for s in state_list[0]])
#        for i in range(1,self.N+1):
#            fid.append([(s[0] * self.env_tar).tr() + \
#                        2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(self.env_tar)) \
#                        for s in state_list[i]])
        for i in range(0,self.N+1):
            fid.append(list(map(fid_spin, tar, state_list[i])))
                
        # general form of fidelity
        fid2 = []
#        append2 = fid2.append
#        fid2.append([np.square(np.absolute((self.c_tar.sqrtm() * s[0] * self.c_tar.sqrtm()).sqrtm().tr())) \
#                     for s in state_list[0]])
#        for i in range(1,self.N+1):
#            fid2.append([np.square(np.absolute((self.env_tar.sqrtm() * s[0] * self.env_tar.sqrtm()).sqrtm().tr())) \
#                         for s in state_list[i]])
        for i in range(0,self.N+1):
            fid2.append(list(map(fid_gen, tar, state_list[i])))
#        print(len(fid2[0]))
        
        return fid2
    
    def expect(self, state_list):
        """
        Calculate expectation of observable sigma_x, sigma_y, sigma_z
        """
        exp_x, exp_y, exp_z = [], [], []
#        appendx = exp_x.append
#        appendy = exp_y.append
#        appendz = exp_z.append
        for i in range(0,self.N+1):
            exp_x.append([qt.expect(qt.sigmax(), [s[0] for s in state_list[i]])])
            exp_y.append([qt.expect(qt.sigmay(), [s[0] for s in state_list[i]])])
            exp_z.append([qt.expect(qt.sigmaz(), [s[0] for s in state_list[i]])])
        
        return exp_x, exp_y, exp_z
    
    def entropy(self, state):
        """
        Calculate von Neumann entropy
        """
        S = []
#        append = S.append
        for i in range(0,len(self.tlist)):
            S.append([-np.trace(state[i] * logm(state[i]))])
        
        return S
    
    def entropy_rel(self, state):
        """
        Calculate relative entropy between nuclear spin state and target state
        """
        S_rel = []
        for i in range(0,len(self.tlist)):
            S_rel.append([np.trace(state[i] * logm(state[i])) - \
                          np.trace(state[i] * logm(self.env_tar))])
        
        return S_rel


params = dict()
params = {
          "N": 7,
          "ce": [0,1],
          "omega": [2500*1e3,10000*1e3,8500*1e3,7000*1e3,5500*1e3,4000*1e3,2500*1e3,1000*1e3],
          "A": [1.20*1e6,1.18*1e6,1.16*1e6,1.14*1e6,1.12*1e6,1.10*1e6,1.08*1e6],
          "n": 1,
          "T": 30e-6,
          "dt": 15e-8,
          "option": 'U'
          }

c_init = qt.ket2dm(params['ce'][0]*qt.basis(2,1)+params['ce'][1]*qt.basis(2,0))
env_init = tensor_power(qt.ket2dm(qt.basis(2,1)), params['N'])  # (2,1) is ground state
#env_init = tensor_power(qt.Qobj([[1/2,0],[0,1/2]]), params['N'])  # mixed state

dim = np.power(2,params['N'])

#cn = []  # random nuclear spin state
#cn.append(random.uniform(0,1))
#for _ in range(0,int(dim)-2):
#    cn.append(random.uniform(0,1-sum(cn)))
#cn.append(1-sum(cn))
#cn = np.sqrt(cn)

#cn = [0.309548,
#0.710792,
#0.251635,
#0.351685,
#0.324818,
#0.20578,
#0.183135,
#0.174821,
#]

#bi = list(product(range(2), repeat=params['N']))  # binary sequence
#zero = qt.Qobj(np.zeros((2, 1)))
#env_init = tensor_power(zero, params['N'])
#for i in range(0,dim):
#    sub = bi2basis(bi[i])
#    env_init += cn[i] * sub
#env_init = qt.ket2dm(env_init)

# selection
## 1 for finding other fidelities when one nuclear spin reaches its climax under specific omega0
## 2 for finding omega0
## 3 for operator fidelity
find='1'

if find=='1':
    model = CentralSpin(params, params['omega'][0], c_init, env_init)
    Ham, Ham_r, Ham_tar = model.ham()
    states, state_list = model.evolve(Ham)
    
    # fidelity
    fid = model.fid(state_list)
    
    fidm_e = max(fid[0])  # fid max
    fidm_1 = max(fid[1])
    fidm_2 = max(fid[2])
    fidm_3 = max(fid[3])
    fidm_4 = max(fid[4])
    fidm_5 = max(fid[5])
    fidm_6 = max(fid[6])
    fidm_7 = max(fid[7])
    
    pos_e = []
#    appende = pos_e.append
    pos_e.append(fid[0].index(fidm_e))  # position when a nuclear spin reaches maximal fidelity
    for i in range(0,model.N+1):
        pos_e.append(fid[i][pos_e[0]])
        
    pos_1 = []
#    append1 = pos_1.append
    pos_1.append(fid[1].index(fidm_1))
    for i in range(0,model.N+1):
        pos_1.append(fid[i][pos_1[0]])
        
    pos_2 = []
#    append2 = pos_2.append
    pos_2.append(fid[2].index(fidm_2))
    for i in range(0,model.N+1):
        pos_2.append(fid[i][pos_2[0]])
        
    pos_3 = []
#    append3 = pos_3.append
    pos_3.append(fid[3].index(fidm_3))
    for i in range(0,model.N+1):
        pos_3.append(fid[i][pos_3[0]])
        
    pos_4 = []
#    append4 = pos_4.append
    pos_4.append(fid[4].index(fidm_4))
    for i in range(0,model.N+1):
        pos_4.append(fid[i][pos_4[0]])
    
    pos_5 = []
#    append5 = pos_5.append
    pos_5.append(fid[5].index(fidm_5))
    for i in range(0,model.N+1):
        pos_5.append(fid[i][pos_5[0]])
        
    pos_6 = []
#    append6 = pos_6.append
    pos_6.append(fid[6].index(fidm_6))
    for i in range(0,model.N+1):
        pos_6.append(fid[i][pos_6[0]])
        
    pos_7 = []
#    append7 = pos_7.append
    pos_7.append(fid[7].index(fidm_7))
    for i in range(0,model.N+1):
        pos_7.append(fid[i][pos_7[0]])
    
    # plot fidelity
    count = params['omega'][0] * model.tlist
    fig = plt.figure(figsize=(8,6))
    l1, = plt.plot(count, fid[0])
    l2, = plt.plot(count, fid[1])
    l3, = plt.plot(count, fid[2])
    l4, = plt.plot(count, fid[3])
    l5, = plt.plot(count, fid[4])
    l6, = plt.plot(count, fid[5])
    l7, = plt.plot(count, fid[6])
    l8, = plt.plot(count, fid[7])
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese character
    plt.rcParams['axes.unicode_minus'] = False
    
#    plt.legend(handles=[l1, l2, l3, l4, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', r'核6', ], 
#               loc='center right', fontsize=16)
    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
               labels=[r'电子', r'核1', r'核2', r'核3', 
                       r'核4', r'核5', r'核6', '核7', ], 
               loc='center right', fontsize=16)    

    plt.xlabel(r'$\omega t$', fontsize=16)
    plt.ylabel(r'$F$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'$F - \omega t$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\data\pos_%.2f.png'%(params['omega'][0]*1e-6))
    
    # entropy
    S = []
    for i in range(0,params['N']+1):
        S.append(model.entropy(state_list[i]))
    
    # plot entropy
    fig = plt.figure(figsize=(8,6))
    l9, = plt.plot(count, S[0])
    l10, = plt.plot(count, S[1])
    l11, = plt.plot(count, S[2])
    l12, = plt.plot(count, S[3])
    l13, = plt.plot(count, S[4])
    l14, = plt.plot(count, S[5])
    l15, = plt.plot(count, S[6])
    l16, = plt.plot(count, S[7])
    
#    plt.legend(handles=[l9, l10, l11, l12, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l9, l10, l11, l12, l13, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l9, l10, l11, l12, l13, l14, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l9, l10, l11, l12, l13, l14, l15, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', r'核6', ], 
#               loc='center right', fontsize=16)
    plt.legend(handles=[l9, l10, l11, l12, l13, l14, l15, l16, ], 
               labels=[r'电子', r'核1', r'核2', r'核3', 
                       r'核4', r'核5', r'核6', '核7', ], 
               loc='center right', fontsize=16)    

    plt.xlabel(r'$\omega t$', fontsize=16)
    plt.ylabel(r'$S$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'$S - \omega t$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\data\S_%.2f.png'%(params['omega'][0]*1e-6))
    
    # entropy change
    dS = []
    for i in range(0,params['N']+1):
        S0 = S[i][0][0]
        dS.append(list(map(lambda x:x-S0, S[i])))
    
    # plot entropy change
    fig = plt.figure(figsize=(8,6))
    l17, = plt.plot(count, dS[0])
    l18, = plt.plot(count, dS[1])
    l19, = plt.plot(count, dS[2])
    l20, = plt.plot(count, dS[3])
    l21, = plt.plot(count, dS[4])
    l22, = plt.plot(count, dS[5])
    l23, = plt.plot(count, dS[6])
    l24, = plt.plot(count, dS[7])
    
#    plt.legend(handles=[l17, l18, l19, l20, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l17, l18, l19, l20, l21, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l17, l18, l19, l20, l21, l22, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l17, l18, l19, l20, l21, l22, l23, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', r'核6', ], 
#               loc='center right', fontsize=16)
    plt.legend(handles=[l17, l18, l19, l20, l21, l22, l23, l24, ], 
               labels=[r'电子', r'核1', r'核2', r'核3', 
                       r'核4', r'核5', r'核6', '核7', ], 
               loc='center right', fontsize=16)    

    plt.xlabel(r'$\omega t$', fontsize=16)
    plt.ylabel(r'$\Delta S$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'$\Delta S - \omega t$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\data\dS_%.2f.png'%(params['omega'][0]*1e-6))
    
    # relative entropy
    relS = []
    for i in range(0,params['N']+1):
        relS.append(model.entropy_rel(state_list[i]))
    
    # plot relative entropy
    fig = plt.figure(figsize=(8,6))
    l25, = plt.plot(count, S[0])
    l26, = plt.plot(count, S[1])
    l27, = plt.plot(count, S[2])
    l28, = plt.plot(count, S[3])
    l29, = plt.plot(count, S[4])
    l30, = plt.plot(count, S[5])
    l31, = plt.plot(count, S[6])
    l32, = plt.plot(count, S[7])
    
#    plt.legend(handles=[l25, l26, l27, l28, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l25, l26, l27, l28, l29, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l25, l26, l27, l28, l29, l30, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', ], 
#               loc='center right', fontsize=16)
#    plt.legend(handles=[l25, l26, l27, l28, l29, l30, l31, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', r'核6', ], 
#               loc='center right', fontsize=16)
    plt.legend(handles=[l25, l26, l27, l28, l29, l30, l31, l32, ], 
               labels=[r'电子', r'核1', r'核2', r'核3', 
                       r'核4', r'核5', r'核6', '核7', ], 
               loc='center right', fontsize=16)    

    plt.xlabel(r'$\omega t$', fontsize=16)
    plt.ylabel(r'$S_{rel}$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'$S_{rel} - \omega t$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\data\relS_%.2f.png'%(params['omega'][0]*1e-6))


elif find=='2':
    omega0_list = np.arange(0.1*1e6, 11.0*1e6+0.01*1e6, 0.01*1e6)
    
#    fidm_e, fidm_1, fidm_2, fidm_3, fidm_4 = [], [], [], [], []  # fid max
#    fidmw_e, fidmw_1, fidmw_2, fidmw_3, fidmw_4 = 2.0, 2.0, 2.0, 2.0, 2.0  # omega0 giving fid max
    
    fidm_e = []  # fid max
    fidm_1 = []
    fidm_2 = []
    fidm_3 = []
    fidm_4 = []
    fidm_5 = []
    fidm_6 = []
    fidm_7 = []
    
    fidmw_e = 0.  # omega0 giving fid max
    fidmw_1 = 0.
    fidmw_2 = 0.
    fidmw_3 = 0.
    fidmw_4 = 0.
    fidmw_5 = 0.
    fidmw_6 = 0.
    fidmw_7 = 0.
    
    zero_array = [0 for i in range(int(params['N']))]
    fidma_1 = zero_array # fidelities of all nuclei at climax
    fidma_2 = zero_array
    fidma_3 = zero_array
    fidma_4 = zero_array
    fidma_5 = zero_array
    fidma_6 = zero_array
    fidma_7 = zero_array
    
    for omega0 in omega0_list:
        model = CentralSpin(params, omega0, c_init, env_init)
        Ham, Ham_r, Ham_tar = model.ham()
        states, state_list = model.evolve(Ham)
        fid = model.fid(state_list)
        S = model.entropy(states)

        fidm_e.append(max(fid[0]))
        fidm_1.append(max(fid[1]))
        fidm_2.append(max(fid[2]))
        fidm_3.append(max(fid[3]))
        fidm_4.append(max(fid[4]))
        fidm_5.append(max(fid[5]))
        fidm_6.append(max(fid[6]))
        fidm_7.append(max(fid[7]))
        
        if len(fidm_e)>=2:
            if fidm_e[-1]>max(fidm_e[:-1]):
                fidmw_e = omega0 * 1e-6
            if fidm_1[-1]>max(fidm_1[:-1]):
                fidmw_1 = omega0 * 1e-6
                fidma_1[0] = fidm_1[-1]
                fidma_1[1] = fidm_2[-1]
                fidma_1[2] = fidm_3[-1]
                fidma_1[3] = fidm_4[-1]
                fidma_1[4] = fidm_5[-1]
                fidma_1[5] = fidm_6[-1]
                fidma_1[6] = fidm_7[-1]
            if fidm_2[-1]>max(fidm_2[:-1]):
                fidmw_2 = omega0 * 1e-6
                fidma_2[0] = fidm_1[-1]
                fidma_2[1] = fidm_2[-1]
                fidma_2[2] = fidm_3[-1]
                fidma_2[3] = fidm_4[-1]
                fidma_2[4] = fidm_5[-1]
                fidma_2[5] = fidm_6[-1]
                fidma_2[6] = fidm_7[-1]
            if fidm_3[-1]>max(fidm_3[:-1]):
                fidmw_3 = omega0 * 1e-6
                fidma_3[0] = fidm_1[-1]
                fidma_3[1] = fidm_2[-1]
                fidma_3[2] = fidm_3[-1]
                fidma_3[3] = fidm_4[-1]
                fidma_3[4] = fidm_5[-1]
                fidma_3[5] = fidm_6[-1]
                fidma_3[6] = fidm_7[-1]
            if fidm_4[-1]>max(fidm_4[:-1]):
                fidmw_4 = omega0 * 1e-6
                fidma_4[0] = fidm_1[-1]
                fidma_4[1] = fidm_2[-1]
                fidma_4[2] = fidm_3[-1]
                fidma_4[3] = fidm_4[-1]
                fidma_4[4] = fidm_5[-1]
                fidma_4[5] = fidm_6[-1]
                fidma_4[6] = fidm_7[-1]
            if fidm_5[-1]>max(fidm_5[:-1]):
                fidmw_5 = omega0 * 1e-6
                fidma_5[0] = fidm_1[-1]
                fidma_5[1] = fidm_2[-1]
                fidma_5[2] = fidm_3[-1]
                fidma_5[3] = fidm_4[-1]
                fidma_5[4] = fidm_5[-1]
                fidma_5[5] = fidm_6[-1]
                fidma_5[6] = fidm_7[-1]
            if fidm_6[-1]>max(fidm_6[:-1]):
                fidmw_6 = omega0 * 1e-6
                fidma_6[0] = fidm_1[-1]
                fidma_6[1] = fidm_2[-1]
                fidma_6[2] = fidm_3[-1]
                fidma_6[3] = fidm_4[-1]
                fidma_6[4] = fidm_5[-1]
                fidma_6[5] = fidm_6[-1]
                fidma_6[6] = fidm_7[-1]
            if fidm_7[-1]>max(fidm_7[:-1]):
                fidmw_7 = omega0 * 1e-6
                fidma_7[0] = fidm_1[-1]
                fidma_7[1] = fidm_2[-1]
                fidma_7[2] = fidm_3[-1]
                fidma_7[3] = fidm_4[-1]
                fidma_7[4] = fidm_5[-1]
                fidma_7[5] = fidm_6[-1]
                fidma_7[6] = fidm_7[-1]

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
        l6, = plt.plot(count, fid[5])
        l7, = plt.plot(count, fid[6])
        l8, = plt.plot(count, fid[7])
        
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese character
        plt.rcParams['axes.unicode_minus'] = False
        
#        plt.legend(handles=[l1, l2, l3, l4, ], 
#                   labels=[r'电子', r'核1', r'核2', r'核3', ], 
#                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, ], 
#                   labels=[r'电子', r'核1', r'核2', r'核3', 
#                           r'核4', ], 
#                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#                   labels=[r'电子', r'核1', r'核2', r'核3', 
#                           r'核4', r'核5', ], 
#                   loc='center right', fontsize=16)
#        plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#                   labels=[r'电子', r'核1', r'核2', r'核3', 
#                           r'核4', r'核5', r'核6', ], 
#                   loc='center right', fontsize=16)
        plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
                   labels=[r'电子', r'核1', r'核2', r'核3', 
                           r'核4', r'核5', r'核6', '核7', ], 
                   loc='center right', fontsize=16)        

        plt.xlabel(r'$\omega t$', fontsize=16)
        plt.ylabel(r'$F$', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(r'$ F-\omega t, \omega_0=%.2f \times 10^6 rad/s$'%(omega0*1e-6), fontsize=20)
        plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\%.2f.png'%(omega0*1e-6))  # change N

#        fig = plt.figure(figsize=(8,6))
#        plt.plot(count, S)
#        plt.xlabel(r'$\omega t$', fontsize=16)
#        plt.ylabel(r'$S$', fontsize=16)
#        plt.xticks(fontsize=14)
#        plt.yticks(fontsize=14)
#        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
#        plt.title(r'$ S-\omega t, \omega_0=%.2f \times 10^6 rad/s$'%(omega0*1e-6), fontsize=20)
#        plt.savefig(r'D:\transfer\trans_code\results_qd\grad1小_0.01步长_cn\S_%.2f.png'%(omega0*1e-6))  # change N
        
#    print("fidmw:", fidmw_e, fidmw_1, fidmw_2, fidmw_3, fidmw_4)
    
    omega0_list_ = [i*1e-6 for i in omega0_list]
    fig = plt.figure(figsize=(8,6))
    l1, = plt.plot(omega0_list_, fidm_e)
    l2, = plt.plot(omega0_list_, fidm_1)
    l3, = plt.plot(omega0_list_, fidm_2)
    l4, = plt.plot(omega0_list_, fidm_3)
    l5, = plt.plot(omega0_list_, fidm_4)
    l6, = plt.plot(omega0_list_, fidm_5)
    l7, = plt.plot(omega0_list_, fidm_6)
    l8, = plt.plot(omega0_list_, fidm_7)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese character
    plt.rcParams['axes.unicode_minus'] = False
    
#    plt.legend(handles=[l1, l2, l3, l4, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', ], 
#               loc='best', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, ],
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', ], 
#               loc='best', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', ], 
#               loc='best', fontsize=16)
#    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, ], 
#               labels=[r'电子', r'核1', r'核2', r'核3', 
#                       r'核4', r'核5', r'核6', ], 
#               loc='lower right', fontsize=16)
    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8, ], 
               labels=[r'电子', r'核1', r'核2', r'核3', 
                       r'核4', r'核5', r'核6', '核7', ], 
               loc='lower right', fontsize=16)

    plt.xlabel(r'$\omega_0 (\times 10^6 rad/s)$', fontsize=16)
    plt.ylabel(r'$F_{max}$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'$F_{max} - \omega_0$', fontsize=20)
    plt.savefig(r'D:\transfer\trans_code\results_qd\grad\N=7_新basis_0.01步长_30mus\fidmax.png')  # change N


elif find=='3':
    model = CentralSpin(params, params['omega'][0], c_init, env_init)
    Ham, Ham_r, Ham_tar = model.ham()
    fidm = fidm(Ham, Ham_tar)
    fidmu = fidmu(Ham.expm(), Ham_tar.expm(), np.power(2,int(params['N']+1)))
    print("fidm:", fidm)
    print("fidmu:", fidmu)
    
