#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:47:38 2024

@author: freterl1
"""

from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
from models import setup_Dicke_block, setup_Dicke, setup_Dicke_block1
from propagate import time_evolve, time_evolve_block, time_evolve_block1, time_evolve_block2
from expect import setup_convert_rho_nrs, setup_convert_rho_block_nrs
import pickle
import operators
    
ntls =5#number 2LS
nphot = ntls+1# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1.0
wc = 0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 0.011
gamma = 0.02
gamma_phi =0.03


################# BLOCK STRUCTURE ####################################
# SETUP
t0_tot = time()
print(f'Number of spins {ntls}')
print('Block form optimized')
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block(parallel=True)       # setup mapping between compressed density matrix and block form
setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM
setup_convert_rho_block_nrs(1)
#sys.exit()

# Initial state
t0 = time()
initial_block = setup_rho_block(basis(nphot,0),basis(2,0))
print('setup initial state block in {:.1f}s'.format(time()-t0), flush=True)
#initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
#sys.exit()
t0=time()
L0,L1 = setup_Dicke_block1(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma)
print('setup L block in {:.1f}s'.format(time()-t0), flush=True)


params = {
    'method': 'permutation_kirton',
    'N': ntls,
    'nphot': nphot,
    'w0': w0,
    'wc': wc,
    'Delta': wc- w0,
    'gamma': gamma,
    'gamma_phi': gamma_phi,
    'kappa': kappa,
    'Omega': Omega,
    }

data = { 'params': params,
        'L0':L0,
        'L1':L1}

fname = f'results/{params["method"]}_L.pkl'
# save results in pickle file
with open(fname, 'wb') as handle:
    pickle.dump(data,handle)









