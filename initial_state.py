# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:17:13 2024

@author: lukas

compare initial states of old code and supercompressed form
"""


from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
    
ntls =1#number 2LS
nphot = ntls+1# photon fock space truncation

setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block(parallel=False)       # setup mapping between compressed density matrix and block form


# Initial state

rho_phot = basis(nphot, 1) # second argument: number of photons in initial state
rho_spin = basis(2,0) # second argument: 0 for up, 1 for down
print('Spin:\n',rho_spin.todense())
print('Photon:\n',rho_phot.todense())

initial_block = setup_rho_block(rho_phot,rho_spin)
initial = setup_rho(rho_phot, rho_spin) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)

from pprint import pprint
print('Initial block:')
pprint(initial_block)
print('Initial:')
pprint(initial)