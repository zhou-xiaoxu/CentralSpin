# -*- coding: utf-8 -*-
"""
@author: Xiaoxu Zhou
Latest update: 09/08/2021
"""

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt


params = dict()
params = {
          "energy_levels": 2,
          "omega1": 1.,
          "omega2": 1.,
          "g": 0.15,
          "gamma": 0.1,
          "T": 1,
          "step": 1e2,
          "option": 'U',
          "pulse": {"amp": 10.,
                    "T": 1,
                    "omega_q": np.pi*10
                    }
          }
tlist = np.linspace(0, int(params['T']), int(params['step']+1))

# pulse
def pulse(t, params_d):
    """
    drive pulse
    """
    amp = params_d['amp']
    omega_q = params_d['omega_q']
    T = params_d['T']
    tau = T / 2
    sigma = T / 5
    drive_pulse = amp * np.exp(-(t-tau)**2/(sigma**2)) * np.cos(omega_q*t)
    return drive_pulse

plt.figure()
plt.plot(tlist, pulse(tlist, params['pulse']))
plt.xlabel(r'$t$')
plt.ylabel(r'$Amplitude$')
plt.title(r'Gaussian drive pulse')

# basis vectors
ket0 = qt.basis(2, 0)
ket1 = qt.basis(2, 1)
#ket00 = qt.tensor([ket0, ket0])
#ket01 = qt.tensor([ket0, ket1])
#ket10 = qt.tensor([ket1, ket0])
#ket11 = qt.tensor([ket1, ket1])

# Pauli matrices
sigma1 = np.array([qt.tensor([qt.sigmax(), qt.qeye(2)]), 
                   qt.tensor([qt.sigmay(), qt.qeye(2)]), 
                   qt.tensor([qt.sigmaz(), qt.qeye(2)])], dtype=qt.Qobj)
sigma2 = np.array([qt.tensor([qt.qeye(2), qt.sigmax()]), 
                   qt.tensor([qt.qeye(2), qt.sigmay()]), 
                   qt.tensor([qt.qeye(2), qt.sigmaz()])], dtype=qt.Qobj)

# annihilation operator
dim = int(params['energy_levels'])
a = qt.destroy(dim)


# 2-qubit system, ZZ coupling
init_spin1 = 1.0 * ket0 + 0.0 * ket1
init_spin2 = 0.0 * ket0 + 1.0 * ket1
init_state = qt.tensor([init_spin1, init_spin2])

Ham_0 = params['omega1'] * sigma1[2] + params['omega2'] * sigma2[2] + params['g'] * sigma1[2] * sigma2[2]
Ham_d = [(qt.tensor([qt.sigmaz(), a + a.dag()])) * pulse(t, params['pulse']) for t in tlist]
Ham = [Ham_0 + Ham_di for Ham_di in Ham_d]
#Ham = Ham_0  # without control

if params["option"] == 'U':
    cops = []
elif params['option'] == 'D':
    cops = [sigma1[2] * np.sqrt(0.5 * params['gamma']), sigma2[2] * np.sqrt(0.5 * params['gamma'])]

eops = [sigma1[2], sigma2[2]]

#evol = qt.sesolve(Ham, init_state, tlist, eops)  # yield expectation, Schroedinger eq.
evol = qt.mesolve(Ham, init_state, tlist, cops, eops)  # yield expectation
#evol = qt.mesolve(Ham, init_state, tlist, cops, args=params['pulse'])  # yield states
#exp1 = qt.expect(eops[0], evol.states)
#exp2 = qt.expect(eops[1], evol.states)

plt.figure()
l1, = plt.plot(tlist, evol.expect[0])  # if .mesolve contains eops
l2, = plt.plot(tlist, evol.expect[1])
#l1, = plt.plot(tlist, exp1)  # if .mesolve does not contain eops
#l2, = plt.plot(tlist, exp2)
plt.xlabel(r'time steps')
plt.ylabel(r'$\langle \sigma_z \rangle$')
plt.legend(handles=[l1, l2, ], 
           labels=['spin up initially', 'spin down initially'], 
           loc='best')
plt.title(r'$\langle \sigma_z \rangle$ versus $t$')
plt.show()

