#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20

@author: freterl1
@co-author: piperfw

Test the new setup_L_block function (profiling)
"""

from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
from models import setup_Dicke_block, setup_Dicke
from propagate import time_evolve, time_evolve_block
from expect import setup_convert_rho_nrs
import pickle
import operators


ntls = int(sys.argv[1]) # number 2LS
nphot = ntls+1# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1#1.0
wc = 1.5#0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 0 #1e-02
gamma = 0 #1e-03
gamma_phi =3e-02


################# BLOCK STRUCTURE ####################################
# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block()       # setup mapping between compressed density matrix and block form

setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM

# Initial state
initial_block = setup_rho_block(basis(nphot,0),basis(2,0))

#initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)

t0=time()
L0,L1 = setup_Dicke_block(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma)
print('setup L block in {:.1f}s'.format(time()-t0), flush=True)


n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

t0=time()
resultscomp_block = time_evolve_block(L0,L1,initial_block, tmax, dt, ops, atol=1e-8, rtol=1e-8, save_states=True)
# if save_states=False, only operator expectations and initial, final density matrices are recorded
runtime=time()-t0
print('Time evolution Block complete in {:.0f}s'.format(runtime), flush=True)


# RESULTS
ts_block = np.array(resultscomp_block.t)
ns_block = np.array(resultscomp_block.expect[0])
ps_block = np.array(resultscomp_block.expect[1])
results = {'ts':ts_block,'ns':ns_block,'ps':ps_block}

if len(sys.argv) > 2:
    fp = 'results/{}_N{}block.pkl'.format(sys.argv[2], ntls)
else:
    fp = f'results/N{ntls}block.pkl'

with open(fp, 'wb') as fb:
    pickle.dump(results, fb)



