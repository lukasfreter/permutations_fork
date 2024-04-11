#!/usr/bin/env python
"""
This file is set up to be equivalent to the qutip code written in the srsf
project.
"""
from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho
from indices import list_equivalent_elements
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
from models import setup_Dicke
from propagate import time_evolve
from expect import setup_convert_rho_nrs
import pickle


ntls = 1# number 2LS
nphot = 2# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1.0
wc = 0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 1e-02
gamma = 1e-03
gamma_phi =3e-02

# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_convert_rho_nrs(1) # conversion matrix from full to photon + single-spin RDM
initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)

t0=time()
# Note: due to the different definitions of sigma_z in this code and qutip code
# we need factors of 1/2 and 1/4 to multiply w0 and gamma_phi, such that 
# the w0 and gamma_phi we define above are equivalent in both codes
L = setup_Dicke(wc, w0/2, 0.0, 0, 0.0, 0, 0, 0, progress=False)
print(initial)
sys.exit()
L = setup_Dicke(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma, progress=False)
print('setup L in {:.1f}s'.format(time()-t0), flush=True)

n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

# PROPAGATE
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, ops, atol=1e-8, rtol=1e-8, save_states=False)
# if save_states=False, only operator expectations and initial, final density matrices are recorded
runtime=time()-t0
print('Time evolution complete in {:.0f}s'.format(runtime), flush=True)

# RESULTS
ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])
fig, axes = plt.subplots(2, figsize=(6,6))
axes[0].set_ylabel(r'$n$')
axes[0].plot(ts, ns.real)
axes[1].plot(ts, ps.real)
axes[1].set_ylabel(r'$\langle \sigma^+\sigma^-\rangle$')
fig.savefig('figures/example.png',dpi=300, bbox_inches='tight')


params = {
    'method': 'permutation',
    'N':ntls,
    'w0':w0,
    'wc':wc,
    'Delta':wc-w0,
    'gamma':gamma,
    'gamma_phi':gamma_phi,
    'kappa':kappa,
    'Omega':Omega,
    'nphot':nphot,
    'tmax':tmax,
    'dt':dt,
    'theta':0.0
    }
res = {
       't':ts,
       'e_phot_tot':ns,
       'e_excit_site':ps
       }
data = {
        'params':params,
        'results':res,
        'runtime':runtime,
        }

filename = f'results/permutation_{params["method"]}_N{ntls}_Delta{wc-w0}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}.pkl'
filename='results/perm_comp.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(data,handle)
    

