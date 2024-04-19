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
from propagate import time_evolve
from expect import setup_convert_rho_nrs
import pickle
import operators


ntls = 1# number 2LS
nphot = ntls+1# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1#1.0
wc = 1.5#0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 0#1e-02
gamma = 0#1e-03
gamma_phi =0#3e-02

# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM
setup_mapping_block()       # setup mapping between compressed density matrix and block form

# Initial state
initial_block = setup_rho_block(basis(nphot,0),basis(2,0))

sys.exit()


initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
print(initial)
sys.exit()

t0=time()
L = setup_Dicke(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma).todense()

