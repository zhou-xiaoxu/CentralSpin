# -*- coding: utf-8 -*-
"""
Main function of central spin model
@author: Xiaoxu Zhou
Latest update: 09/13/2021
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt

from utils.utils import tensor_power, sigmazi


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

        # fidelity
        ## fidelity as a whole
        fid_whole = [(np.trace(np.sqrt(np.sqrt(rho_init) * state * np.sqrt(rho_init))))**2 for state in evol.states]
        ## fidelity considering each environment spin
        c_list = [qt.ptrace(state,0) for state in evol.states]
#        ### N=2
#        spin1_list = [qt.ptrace(state,1) for state in evol.states]
#        spin2_list = [qt.ptrace(state,2) for state in evol.states]
##        print(cs_t)
#        fid_c = [(np.trace(np.sqrt(np.sqrt(c_init) * c * np.sqrt(c_init))))**2 for c in c_list]
#        fid_spin1 = [(np.trace(np.sqrt(np.sqrt(c_init) * spin1 * np.sqrt(c_init))))**2 for spin1 in spin1_list]
#        fid_spin2 = [(np.trace(np.sqrt(np.sqrt(c_init) * spin2 * np.sqrt(c_init))))**2 for spin2 in spin2_list]
#        fid_each = np.array([fid_c, fid_spin1, fid_spin2])
        ### N=3
        spin1_list = [qt.ptrace(state,1) for state in evol.states]
        spin2_list = [qt.ptrace(state,2) for state in evol.states]
        spin3_list = [qt.ptrace(state,3) for state in evol.states]
        fid_c = [(np.trace(np.sqrt(np.sqrt(c_init) * c * np.sqrt(c_init))))**2 for c in c_list]
        fid_spin1 = [(np.trace(np.sqrt(np.sqrt(c_init) * spin1 * np.sqrt(c_init))))**2 for spin1 in spin1_list]
        fid_spin2 = [(np.trace(np.sqrt(np.sqrt(c_init) * spin2 * np.sqrt(c_init))))**2 for spin2 in spin2_list]
        fid_spin3 = [(np.trace(np.sqrt(np.sqrt(c_init) * spin3 * np.sqrt(c_init))))**2 for spin3 in spin3_list]
        fid_each = np.array([fid_c, fid_spin1, fid_spin2, fid_spin3])
        
        return fid_whole, fid_each
 
params = dict()
params = {
          "N": 3,
          "omega": [1.,1.,1.,1.],
          "lambda": [0.1,0.1,0.1],
          "T": 1e2,
          "step": 2e3,
          "option": 'U'
          }

model = CentralSpin(params)
fid_whole, fid_each = model.nmr()
count = np.arange(0,params['step']+1,1)

# plot
## whole fidelity
plt.figure()
l1, = plt.plot(count, fid_whole)
plt.xlabel(r'time steps')
plt.ylabel(r'fidelity')
plt.legend(handles=[l1, ], labels=['N=%d'%params['N']], loc='best')
plt.title(r'Central spin model')

## each bath spin's fidelity
### N=2
#plt.figure()
#l1, = plt.plot(count, fid_each[0])
#l2, = plt.plot(count, fid_each[1])
#l3, = plt.plot(count, fid_each[2])
#plt.xlabel(r'time steps')
#plt.ylabel(r'fidelity')
#plt.legend(handles=[l1, l2, l3, ], 
#           labels=['central spin', 'bath spin 1', 'bath spin 2'], 
#           loc='best')
#plt.title(r'Central spin model, N=%d'%params['N'])

### N=3
plt.figure()
l1, = plt.plot(count, fid_each[0])
l2, = plt.plot(count, fid_each[1])
l3, = plt.plot(count, fid_each[2])
l4, = plt.plot(count, fid_each[3])
plt.xlabel(r'time steps')
plt.ylabel(r'fidelity')
plt.legend(handles=[l1, l2, l3, l4, ], 
           labels=['central spin', 'bath spin 1', 'bath spin 2', 'bath spin 3'], 
           loc='best')
plt.title(r'Central spin model, N=%d'%params['N'])

plt.show()

