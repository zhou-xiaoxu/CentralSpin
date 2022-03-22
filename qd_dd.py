# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 03/22/2022
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from utils.utils import tensor_power, sigmaxi, sigmayi, sigmazi


class QDSystem(object):
    """
    Quantum dot system
    Index 0 is for central spin, index 1 to N are for the 1st to the Nth
    environment spin respectively
    """
    def __init__(self, params, c_init, env_init):
        self.params = params
        self.N = int(params['N'])
        self.omega = params['omega']
        self.omegad = params['omegad']
        self.A = params['A']
        self.mu = params['mu']
        self.Bac = params['Bac']
        
        self.T = params['T']
        self.dt = params['dt']
        self.tlist = np.arange(0,self.T+self.dt,self.dt)
        
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
        sigmax0 = qt.tensor([qt.sigmax(), tensor_power(qt.qeye(2), self.N)])
        sigmay0 = qt.tensor([qt.sigmay(), tensor_power(qt.qeye(2), self.N)])
        sigmaz0 = qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), self.N)])
        tr = 1e-6
        
        # operator acting on the electron
        H_e = 0.5 * self.omega[0] * sigmaz0
        
        # operator acting on the electron in RWA
        H_e_r = 0.5 * (self.omega[0] - self.omegad) * sigmaz0

        # operator acting on each nucleus
        H_env = tensor_power(zero, self.N)
        for i in range(1,self.N+1):
            H_env += 0.5 * self.omega[i] * sigmazi(i, self.N)
        H_env = qt.tensor([qt.qeye(2), H_env])
        
        # interaction term
        H_int = tensor_power(zero, self.N+1)
        ## general form
#        for i in range(1,self.N+1):
#            H_int += 1/4 * self.A[i-1] * \
#                     (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
#                      qt.tensor([qt.sigmay(), sigmayi(i, self.N)]) + \
#                      qt.tensor([qt.sigmaz(), sigmazi(i, self.N)]))
        ## neglect zz interaction
        for i in range(1,self.N+1):
            H_int += 1/4* self.A[i-1] * \
                     (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
                      qt.tensor([qt.sigmay(), sigmayi(i, self.N)]))
        
        # interaction term in RWA
        H_int_r = tensor_power(zero, self.N+1)
        for i in range(1,self.N+1):
            H_int_r += 1/4 * self.A[i-1] * \
                      (np.cos(self.omegad * tr) * (sigmax0 * sigmaxi(i,self.N+1) + sigmay0 * sigmayi(i,self.N+1)) + \
                       np.sin(self.omegad * tr) * (sigmax0 * sigmayi(i,self.N+1) - sigmax0 * sigmayi(i,self.N+1)) + \
                       sigmaz0 * sigmazi(i,self.N+1))
        
        # control term
        H_c = 2 * self.mu * self.Bac * sigmaxi(i, self.N+1) * np.cos(self.omegad * tr)
        
        # control term in RWA
        H_c_r = tensor_power(zero, self.N+1)
        for i in range(1,self.N+1):
            H_c_r += self.mu * self.Bac * sigmaxi(i, self.N+1)    
        
        # control term only on the electron in RWA
        H_ce_r = self.mu * self.Bac * sigmax0
        
        # total Hamiltonian
        H_free = H_e + H_env + H_int
        H_c = H_e + H_env + H_int + H_c
        
        return H_free, H_c


class Evolve(object):
    
    
    def free(Ham, rho_init, N, t, dt):
        """
        Free evolution
        Args:
            N: N nuclei
        """
        tlist = np.arange(0,t+dt,dt)
#        print('tlist:', len(tlist))
        evol = qt.mesolve(Ham, rho_init, tlist, [], [])
        state_list = []
        for i in range(0,N+1):
            state_list.append([[qt.ptrace(s,i)] for s in evol.states])  # s=state
#        print('state_list':, len(state_list))
#        print('state_list:', len(state_list[0]))
        return evol.states[-1], state_list
    
    def rot0(state, N, axis, phi):
        """
        The whole system rotates angle phi around axis-x, or -y, or -z
        The system consists of 1 electron and N nuclei
        Rotate each spin one by one
        """
        spin = []
        if axis=='x':
            for i in range(0,N+1):
                spin.append(qt.rx(phi) * qt.ptrace(state,i))
            state_ = spin[0]
            for i in range(1,N+1):
                state_ = qt.tensor([state_, spin[i]])
        
        elif axis=='y':
            for i in range(0,N+1):
                spin.append(qt.ry(phi) * qt.ptrace(state,i))
            state_ = spin[0]
            for i in range(1,N+1):
                state_ = qt.tensor([state_, spin[i]])
        
        elif axis=='z':
            for i in range(0,N+1):
                spin.append(qt.rz(phi) * qt.ptrace(state,i))
            state_ = spin[0]
            for i in range(1,N+1):
                state_ = qt.tensor([state_, spin[i]])
        
        else:
            print('Undefined axis')
        
        return state_, spin
    
    def rot(state, N, axis, phi):
        """
        The whole system rotates angle phi around axis-x, or -y, or -z
        The system consists of 1 electron and N nuclei
        Rotate all the spins using tensor product of rotation operator
        """
        rx_ = tensor_power(qt.rx(phi), int(N+1))
        ry_ = tensor_power(qt.ry(phi), int(N+1))
        rz_ = tensor_power(qt.rz(phi), int(N+1))
        
        if axis=='x':
            state = rx_ * state
        elif axis=='y':
            state = ry_ * state
        elif axis=='z':
            state = rz_ * state
        else:
            print('Undefined axis')
        
        spin = []
        for i in range(0,N+1):
            spin.append(qt.ptrace(state,i))
            
        return state, spin
        
    def rect(axis, omega, deltat):
        """
        An x-, or y-, or z-pulse with time duration rather than a delta pulse
        Args:
            deltat: time duration of the pulse
        """
        if axis=='x':
            rotm = qt.rx(omega*deltat)
        elif axis=='y':
            rotm = qt.ry(omega*deltat)
        elif axis=='z':
            rotm = qt.rz(omega*deltat)
        else:
            print('Undefined axis')
            
        return rotm
    
    def gauss(T, dt, sigma):
        """
        Gaussian-shape pulse
        """
        t0 = T/2
        tlist = np.arange(0,T+dt,dt)
        pulse = []
        for t in tlist:
            pulse.append(np.exp((-(t - t0)**2) / (2 * (sigma)**2)))
            
        plt.figure(figsize=(6,8))
        plt.plot(tlist, pulse)
        plt.xlabel('$t$', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        plt.title('Gaussian-shape pulse', fontsize=16)
        plt.show()
        
        return pulse


class Calculate(object):
    
    
    def fid(N, tar, state_list):
        """
        Calculate fidelity
        """
        fid = []  # fidelity especially for spin system
        for i in range(0,N+1):
            fid.append([(s[0] * tar).tr() + \
                        2 * np.sqrt(np.linalg.det(s[0]) * np.linalg.det(tar)) \
                        for s in state_list[i]])
        
        fid2 = []  # general form of fidelity
        for i in range(0,N+1):
            fid2.append([np.square(np.absolute((tar.sqrtm() * s[0] * tar.sqrtm()).sqrtm().tr())) \
                         for s in state_list[i]])
        
        return fid2
    
    def fidr(N, tar, state):
        """
        Calculate fidelity after a delta pulse
        """
        fid = []  # fidelity especially for spin system
        for i in range(0,N+1):
            fid.append([(state[i] * tar).tr() + \
                        2 * np.sqrt(np.linalg.det(state[i]) * np.linalg.det(tar))])
        
        fid2 = []  # general form of fidelity
        for i in range(0,N+1):
            fid2.append([np.square(np.absolute((tar.sqrtm() * state[i] * tar.sqrtm()).sqrtm().tr()))])
        
        return fid2
    
    def expect(N, state_list):
        """
        Calculate expectation values of observable sigma_x, sigma_y, sigma_z
        """
        exp_x = []
        exp_y = []
        exp_z = []
        for i in range(0,N+1):
            exp_x.append([qt.expect(qt.sigmax(), [s[0] for s in state_list[i]])])
            exp_y.append([qt.expect(qt.sigmay(), [s[0] for s in state_list[i]])])
            exp_z.append([qt.expect(qt.sigmaz(), [s[0] for s in state_list[i]])])
        
        return exp_x, exp_y, exp_z
    
    def expectr(N, state):
        """
        Calculate expectation values of observable sigma_x, sigma_y, sigma_z
        after a delta pulse
        """
        exp_x = []
        exp_y = []
        exp_z = []
        for i in range(0,N+1):
            exp_x.append([qt.expect(qt.sigmax(), state[i])])
            exp_y.append([qt.expect(qt.sigmay(), state[i])])
            exp_z.append([qt.expect(qt.sigmaz(), state[i])])
        
        return exp_x, exp_y, exp_z

    def plot(count, fid, N, index):
        """
        Plot trend of fidelity
        Args:
            fid: original fidelity list
            N: N nuclei
            index: name of figure, indicating which period the figure shows
        """
#        fig, ax = plt.subplots(figsize=(8,6))
#        ax.plot(count, fid[0], label='electron')
#        for i in range(1, int(N)+1):
#            ax.plot(count, fid[i], label='nucleus %d'%i)
#        ax.legend(fontsize=16, loc='center right')
#        ax.set_xlabel(r'$\omega t$', fontsize=16)
#        ax.set_ylabel('fidelity', fontsize=16)
#        ax.set_title(r'$ F-t, N=%d $'%N, fontsize=18)
        
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
        plt.ylim((0.,1.0))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(r'$ F-t, N=%d $'%N, fontsize=20)
        plt.savefig(r'D:\transfer\trans_code\results_qd\dd\N=%d\%d.png'%(N,index))
        

params = dict()
params = {
          "N": 4,
          "omega": [8500*1e3,10000*1e3,8500*1e3,7000*1e3,5500*1e3,4000*1e3,2500*1e3,1000*1e3],
          "omegad": 1*2*np.pi,
          "A": [1*1e6,1*1e6,1*1e6,1*1e6,1.12*1e6,1.10*1e6,1.08*1e6],
          "mu": 1,
          "Bac": 1,
          "T": 40e-6,
          "dt": 40e-9,
          "option": 'U'
          }

# 2 pi-pulses around z-axis
c_init = qt.ket2dm(qt.basis(2, 0))
env_init = tensor_power(qt.ket2dm(qt.basis(2, 1)), params['N'])  # alternative
model = QDSystem(params, c_init, env_init)
Ham, Ham_r = model.ham()

endstate = []
endstate.append([model.rho_init])
state_list = []
fid = []

for i in range(0,2):

    endstate1, state_list1 = Evolve.free(Ham, endstate[-1][0], model.N, 1/4*model.T, model.dt)
    endstate.append([endstate1])
    state_list.append([state_list1])
    fid.append(Calculate.fid(model.N, model.env_tar, state_list[-1][0]))
    
    endstate2, state_list2 = Evolve.rot(endstate[-1][0], model.N, 'z', np.pi/2)
    endstate.append([endstate2])
    state_list.append([state_list2])
    fid.append(Calculate.fidr(model.N, model.env_tar, state_list[-1][0]))
    
    endstate3, state_list3 = Evolve.free(Ham, endstate[-1][0], model.N, 1/2*model.T, model.dt)
    endstate.append([endstate3])
    state_list.append([state_list3])
    fid.append(Calculate.fid(model.N, model.env_tar, state_list[-1][0]))
    
    endstate4, state_list4 = Evolve.rot(endstate[-1][0], model.N, 'z', np.pi/2)
    endstate.append([endstate4])
    state_list.append([state_list4])
    fid.append(Calculate.fidr(model.N, model.env_tar, state_list[-1][0]))
    
    endstate5, state_list5 = Evolve.free(Ham, endstate[-1][0], model.N, 1/4*model.T, model.dt)
    endstate.append([endstate5])
    state_list.append([state_list5])
    fid.append(Calculate.fid(model.N, model.env_tar, state_list[-1][0]))
    
    
    # plot fidelity
    count = params['omega'][0] * model.tlist
    
    Calculate.plot(count[:int(1/4*len(model.tlist))+1], fid[0+5*i], model.N, index=0+5*i+1)
    Calculate.plot(count[:int(1/2*len(model.tlist))+1], fid[2+5*i], model.N, index=2+5*i+1)
    Calculate.plot(count[:int(1/4*len(model.tlist))+1], fid[4+5*i], model.N, index=4+5*i+1)
    

