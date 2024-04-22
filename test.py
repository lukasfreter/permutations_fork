#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:32:25 2024

@author: freterl1

Test the new setup_L_block function
"""

from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
from models import setup_Dicke_block, setup_Dicke
from propagate import time_evolve, time_evolve_block, time_evolve_block1
from expect import setup_convert_rho_nrs
import pickle
import operators
    
ntls =1 #number 2LS
nphot = ntls+1# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1#1.0
wc = 1.5#0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 1e-02
gamma = 1e-03
gamma_phi =3e-02


################# BLOCK STRUCTURE ####################################
# SETUP
print(f'Number of spins {ntls}')
print('BLOCK form')
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block()       # setup mapping between compressed density matrix and block form
setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM
# Initial state
t0 = time()
initial_block = setup_rho_block(basis(nphot,0),basis(2,0))
print('setup initial state block in {:.1f}s'.format(time()-t0), flush=True)
#initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)

t0=time()
L0,L1 = setup_Dicke_block(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma)
print('setup L block in {:.1f}s'.format(time()-t0), flush=True)
sys.exit()



n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

t0=time()
resultscomp_block = time_evolve_block(L0,L1,initial_block, tmax, dt, ops, atol=1e-8, rtol=1e-8, save_states=True)
# if save_states=False, only operator expectations and initial, final density matrices are recorded
runtime=time()-t0
print('Time evolution Block complete in {:.0f}s'.format(runtime), flush=True)

#sys.exit()

############################# OLD CODE #####################################
print('-----------------')
print('Full form')

# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_convert_rho_nrs(1) # conversion matrix from full to photon + single-spin RDM

t0 = time()
initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
print('setup initial state full in {:.1f}s'.format(time()-t0), flush=True)


t0=time()
L = setup_Dicke(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma, progress=False)
print('setup L in {:.1f}s'.format(time()-t0), flush=True)

n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

# PROPAGATE
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, ops, atol=1e-8, rtol=1e-8, save_states=True)
runtime=time()-t0
print('Time evolution complete in {:.0f}s'.format(runtime), flush=True)

rho_block = resultscomp_block.rho
rho_ref= resultscomp.rho


# RESULTS
ts_block = np.array(resultscomp_block.t)
ns_block = np.array(resultscomp_block.expect[0])
ps_block = np.array(resultscomp_block.expect[1])
ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])

fig, axes = plt.subplots(2, figsize=(6,6))
axes[0].set_ylabel(r'$n$')
axes[0].plot(ts_block, ns_block.real,label='Block')
axes[0].plot(ts, ns.real, label = 'Reference')
axes[1].plot(ts_block, ps_block.real, label='Block')
axes[1].plot(ts, ps.real, label='Reference')
axes[1].set_ylabel(r'$\langle \sigma^+\sigma^-\rangle$')
plt.legend()
#fig.savefig('figures/example_block.png',dpi=300, bbox_inches='tight')
plt.show()












