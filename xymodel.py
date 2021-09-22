# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 09/18/2021
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt


params = dict()
params = {
          "energy_levels": 2,
          "omega1": 1.,
          "omega2": 1.,
          "lambda": 0.1,
          "gamma": 0.1,
          "T": 1e2,
          "step": 1e2,
          "dt":1e-2,
          "option": 'U'
          }
tlist = np.arange(0, params['T']+params['dt'], params['dt'])

# basis vectors
ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)

# Pauli matrices
sigma1 = np.array([qt.tensor([qt.sigmax(), qt.qeye(2)]), 
                   qt.tensor([qt.sigmay(), qt.qeye(2)]), 
                   qt.tensor([qt.sigmaz(), qt.qeye(2)])], dtype=qt.Qobj)
sigma2 = np.array([qt.tensor([qt.qeye(2), qt.sigmax()]), 
                   qt.tensor([qt.qeye(2), qt.sigmay()]), 
                   qt.tensor([qt.qeye(2), qt.sigmaz()])], dtype=qt.Qobj)


# 2-qubit system, XY model
init_spin1 = 1.0 * ket0 + 0.0 * ket1
init_spin2 = 0.0 * ket0 + 1.0 * ket1
init_state = qt.tensor([qt.ket2dm(init_spin1), qt.ket2dm(init_spin2)])

Ham_0 = 0.5 * params['omega1'] * sigma1[2] + 0.5 * params['omega2'] * sigma2[2] + \
        0.5 * params['lambda'] * (qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay()))
Ham = Ham_0  # without control

if params["option"] == 'U':
    cops = []
elif params['option'] == 'D':
    cops = [sigma1[2] * np.sqrt(0.5 * params['gamma']), sigma2[2] * np.sqrt(0.5 * params['gamma'])]

eops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

#evol = qt.sesolve(Ham, init_state, tlist, eops)  # yield expectation, Schroedinger eq.
#evol = qt.mesolve(Ham, init_state, tlist, cops, eops)  # yield expectation
evol = qt.mesolve(Ham, init_state, tlist, cops)  # yield states
state_list1 = [qt.ptrace(evol.states[i],0) for i in range(0,len(tlist))]
state_list2 = [qt.ptrace(evol.states[i],1) for i in range(0,len(tlist))]
e1x = [qt.expect(eops[0], [s for s in state_list1])][0]
e1y = [qt.expect(eops[1], [s for s in state_list1])][0]
e1z = [qt.expect(eops[2], [s for s in state_list1])][0]
e2x = [qt.expect(eops[0], [s for s in state_list2])][0]
e2y = [qt.expect(eops[1], [s for s in state_list2])][0]
e2z = [qt.expect(eops[2], [s for s in state_list2])][0]

plt.figure(figsize=(8,6))
#l1, = plt.plot(tlist, evol.expect[0])  # if .mesolve contains eops
#l2, = plt.plot(tlist, evol.expect[1])
#l3, = plt.plot(tlist, evol.expect[2])
#l4, = plt.plot(tlist, evol.expect[3])
#l5, = plt.plot(tlist, evol.expect[4])
#l6, = plt.plot(tlist, evol.expect[5])
#l1, = plt.plot(tlist, e1x)  # if .mesolve does not contain eops
#l2, = plt.plot(tlist, e1y)
#l3, = plt.plot(tlist, e1z)
#l4, = plt.plot(tlist, e2x)
#l5, = plt.plot(tlist, e2y)
#l6, = plt.plot(tlist, e2z)
#plt.legend(handles=[l1, l2, l3, l4, l5, l6, ], 
#           labels=['up x', 'up y', 'up z', 'down x', 'down y', 'down z'], 
#           loc='best')
l1, = plt.plot(tlist, e1x)
l2, = plt.plot(tlist, e1z)
l3, = plt.plot(tlist, e2z)
plt.legend(handles=[l1, l2, l3, ], 
           labels=[r'$\langle \sigma_x \rangle, \langle \sigma_y \rangle$ on each spin', 
                   r'$\langle \sigma_z \rangle$ on initially-up spin', 
                   r'$\langle \sigma_z \rangle$ on initially-down spin'], 
           fontsize=9)
plt.xlabel(r't', fontsize=12)
plt.ylabel(r'$\langle \sigma_i \rangle$', fontsize=12)
plt.title(r'$\langle \sigma_i \rangle$ versus $t$', fontsize=16)
plt.show()

