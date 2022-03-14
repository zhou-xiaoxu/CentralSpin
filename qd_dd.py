# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 03/14/2022
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
        self.A = params['A']
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
        # electron term in Hamiltonian
        Ham_e = self.omega[0] * qt.tensor([qt.sigmaz(), tensor_power(qt.qeye(2), self.N)])

        # environment term in Hamiltonian
        zero = qt.ket2dm(qt.Qobj(np.zeros((2, 1))))
        Ham_env = tensor_power(zero, self.N)
        for i in range(1,self.N+1):
            Ham_env += self.omega[i] * sigmazi(i, self.N)
        Ham_env = qt.tensor([qt.qeye(2), Ham_env])
        
        # interaction term in Hamiltonian
        Ham_int = qt.tensor([qt.Qobj(np.zeros((2, 2))), tensor_power(zero, self.N)])
        ## general form
#        for i in range(1,self.N+1):
#            Ham_int += self.A[i-1] * \
#                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
#                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]) + \
#                        qt.tensor([qt.sigmaz(), sigmazi(i, self.N)]))
        ## neglect zz interaction
        for i in range(1,self.N+1):
            Ham_int += self.A[i-1] * \
                       (qt.tensor([qt.sigmax(), sigmaxi(i, self.N)]) + \
                        qt.tensor([qt.sigmay(), sigmayi(i, self.N)]))
        
        # total Hamiltonian in interaction picture
        Ham = Ham_e + Ham_env + Ham_int
        
        # Hamiltonian in rotating frame
        omegar = 2*np.pi/1e-2
        tr = 2e-2
        Ham_r = qt.tensor([qt.Qobj(np.zeros((2, 2))), tensor_power(zero, self.N)])
        for i in range(1,self.N+1):
            Ham_r += self.A[i-1] * \
            (qt.tensor([qt.sigmax(), (sigmaxi(i,self.N)*np.cos(2*omegar*tr) + \
                                  sigmayi(i,self.N)*np.sin(2*omegar*tr))]) + \
             qt.tensor([qt.sigmay(), (-sigmaxi(i,self.N)*np.sin(2*omegar*tr) + \
                                   sigmayi(i,self.N)*np.cos(2*omegar*tr))]))
        
        return Ham, Ham_r


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
    
    def rot(state, N, axis, phi):
        """
        The whole system rotates angle phi around axis-x, or -y, or -z
        The system consists of 1 electron and N nuclei
        """
        spin = []
        if axis=='x':
            for i in range(0,N+1):
                spin.append(qt.rx(phi) * qt.ptrace(state,i))
#            print('spin:', spin)    
            state_ = spin[0]
            for i in range(1,N+1):
                state_ = qt.tensor([state_, spin[i]])
#            print('state_:', state_)
        
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

    def plot(count, fid, N):
        """
        Plot trend of fidelity
        Args:
            fid: original fidelity list
            N: N nuclei
        """
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(count, fid[0], label='central spin')
        for i in range(1, int(N)+1):
            ax.plot(count, fid[i], label='bath spin %d'%i)
        ax.legend(fontsize=16, loc='upper right')
        ax.set_xlabel(r'$\omega t$', fontsize=16)
        ax.set_ylabel('fidelity', fontsize=16)
        ax.set_title(r'$ F-t, N=%d $'%N, fontsize=18)
        

params = dict()
params = {
          "N": 4,
          "omega": [1e6,1e6,1e6,1e6,1e6,1e6,1e6],
          "A": [5.6*1e6,1.9*1e6,1.2*1e6,1e6,1e6,1e6,1e6],
          "Bac": 1,
          "T": 2e-6,
          "dt": 2e-9,
          "option": 'U'
          }

# 2 pi-pulses around z-axis
c_init = qt.ket2dm(qt.basis(2, 0))
env_init = tensor_power(qt.ket2dm(qt.basis(2, 1)), params['N'])  # alternative
model = QDSystem(params, c_init, env_init)
Ham, Ham_r = model.ham()

endstate1, state_list1 = Evolve.free(Ham, model.rho_init, model.N, 1/3*model.T, model.dt)
fid1 = Calculate.fid(model.N, model.env_tar, state_list1)
exp_x1, exp_y1, exp_z1 = Calculate.expect(model.N, state_list1)

endstate2, state_list2 = Evolve.rot(endstate1, model.N, 'z', np.pi)
fid2 = Calculate.fidr(model.N, model.env_tar, state_list2)
exp_x2, exp_y2, exp_z2 = Calculate.expectr(model.N, state_list2)

endstate3, state_list3 = Evolve.free(Ham, endstate2, model.N, 1/3*model.T, model.dt)
fid3 = Calculate.fid(model.N, model.env_tar, state_list3)
exp_x3, exp_y3, exp_z3 = Calculate.expect(model.N, state_list3)

endstate4, state_list4 = Evolve.rot(endstate3, model.N, 'z', np.pi)
fid4 = Calculate.fidr(model.N, model.env_tar, state_list4)
exp_x4, exp_y4, exp_z4 = Calculate.expectr(model.N, state_list4)

endstate5, state_list5 = Evolve.free(Ham, endstate4, model.N, 1/3*model.T, model.dt)
fid5 = Calculate.fid(model.N, model.env_tar, state_list5)
exp_x5, exp_y5, exp_z5 = Calculate.expect(model.N, state_list5)


# plot fidelity
count = params['omega'][0] * model.tlist

Calculate.plot(count[:int(1/3*len(model.tlist))+2], fid1, model.N)
Calculate.plot(count[:int(1/3*len(model.tlist))+2], fid3, model.N)
Calculate.plot(count[:int(1/3*len(model.tlist))+2], fid5, model.N)


# plot expectation
#fig, ax = plt.subplots(figsize=(8,6))
# sigmax
#ax.plot(count, exp_x1[0][0], label=r'$\langle \sigma_x \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_x1[i][0].T, label=r'$\langle \sigma_x \rangle$ on bath spin %d'%i)
# sigmay
#ax.plot(count, exp_y1[0][0], label=r'$\langle \sigma_y \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_y1[i][0], label=r'$\langle \sigma_y \rangle$ on bath spin %d'%i)
#ax.plot(count, exp_x1[0][0], label=r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin')
# sigmaz
#ax.plot(count, exp_z3[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    ax.plot(count, exp_z1[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#ax.plot(count, exp_z3[1][0], label=r'$\langle \sigma_z \rangle$ on each bath spin')

#ax.legend(fontsize=9, loc='upper right')
#ax.set_xlabel(r'$\omega t$', fontsize=12)
#ax.set_ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#ax.set_title(r'$ \langle \sigma_i \rangle-t, N=%d $'%params['N'], fontsize=16)


#plt.figure(figsize=(8,6))
#plt.plot(count, exp_x1[0][0], label=r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin')
#plt.plot(count, exp_z1[0][0], label=r'$\langle \sigma_z \rangle$ on central spin')
#for i in range(1,int(params['N']+1)):
#    plt.plot(count, exp_z1[i][0], label=r'$\langle \sigma_z \rangle$ on bath spin %d'%i)
#plt.xlabel(r'$t$', fontsize=12)
#plt.ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
#plt.title(r'$ \langle \sigma_i \rangle-t, N=%d $'%params['N'], fontsize=16)
