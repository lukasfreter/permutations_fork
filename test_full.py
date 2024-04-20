#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20

@author: freterl1
@co-author: piperfw

Test the old setup_L function (profiling)
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


############################# OLD CODE #####################################
# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_convert_rho_nrs(1) # conversion matrix from full to photon + single-spin RDM

initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
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

rho_ref= resultscomp.rho


# RESULTS

ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])

results = {'ts':ts,'ns':ns,'ps':ps}

if len(sys.argv) > 2:
    fp = 'results/{}_N{}full.pkl'.format(sys.argv[2], ntls)
else:
    fp = f'results/N{ntls}full.pkl'

with open(fp, 'wb') as fb:
    pickle.dump(results, fb)










