#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:01:33 2024

@author: freterl1

use this file to compare PIBS code to Peter Kirton's original code

Need to restart kernel each time the script is run due to name clash of the 
propagate modules of Peters code and PIBS code
"""

from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from basis import setup_basis, setup_rho, setup_rho_block
from indices import list_equivalent_elements, setup_mapping_block
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam
from models import setup_Dicke_block, setup_Dicke, setup_Dicke_block1
from propagate import time_evolve, time_evolve_block, time_evolve_block1, time_evolve_block2, time_evolve_block_interp
from expect import setup_convert_rho_nrs, setup_convert_rho_block_nrs
import pickle
import operators
import scipy.sparse as sp



# from util import qeye, create, destroy, sigmam, sigmap, tensor, basis

# plt.rcParams.update({'font.size': 18,
#                      'xtick.labelsize' : 18,
#                      'ytick.labelsize' : 18,
#                      'lines.linewidth' : 3,
#                      'lines.markersize': 10,
#                      'figure.figsize': (14,10),
#                      'figure.dpi': 150})
    
ntls =8#number 2LS
nphot = 5# photon fock space truncation
tmax = 200.0
dt = 0.2 # timestep

w0 = 1.0
wc = 0.65
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 1e-02
gamma = 1e-03
gamma_phi = 3e-2

atol=1e-8
rtol=1e-8



################### INITIAL PHOTON AND SPIN STATES ##################
# rotation matrix around x-axis of spin 1/2 : exp(-i*theta*Sx)=exp(-i*theta/2*sigmax) = cos(theta/2)-i*sin(theta/2)*sigmax
theta = 0
rot_x = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
rot_x_dag = np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],[1j*np.sin(theta/2), np.cos(theta/2)]])

rho_phot = basis(nphot,0)
rho_spin = sp.csr_matrix(rot_x @ basis(2,0) @ rot_x_dag)
print('Initial states:')
print('Spin:\n',rho_spin.todense())
print('Photon:\n',rho_phot.todense())



############################# PETERS CODE #####################################
print('-----------------')
print('Solve using Peter Kirtons code...')

# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
# setup_convert_rho_nrs(2) # conversion matrix from full to photon + single-spin RDM
setup_convert_rho_nrs(1) # conversion matrix from full to photon + single-spin RDM

t0 = time()
initial = setup_rho(rho_phot, rho_spin) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
print('setup initial state full in {:.1f}s'.format(time()-t0), flush=True)

t0=time()
L = setup_Dicke(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi, gamma, progress=False, parallel=True)
print('setup L in {:.1f}s'.format(time()-t0), flush=True)

n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
# sigp_sigm_ij = tensor(qeye(nphot), sigmap(), sigmam())
# a_sigp = tensor(destroy(nphot), sigmap())
# ops = [n,p,  a_sigp,sigp_sigm_ij,] # operators to calculate expectations for
ops = [n,p] # operators to calculate expectations for

# PROPAGATE
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, ops, atol=atol, rtol=rtol, save_states=False)
runtime=time()-t0
print('Time evolution complete in {:.0f}s'.format(runtime), flush=True)


# RESULTS
ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])
# sigp_sigm_ij_kirton = np.array(resultscomp.expect[3])
# a_sigp_kirton = np.array(resultscomp.expect[2])



############################# PIBS CODE #####################################
# pibs imports
module_name = 'propagate'
if module_name in sys.modules:
    print('yes')
    del sys.modules[module_name]
sys.path.insert(0, '../pibs/pibs/')
from setup import Indices, Rho, Models
from propagate import TimeEvolve
print('-----------------')
print('Solve using PIBS...')

indi = Indices(ntls,nphot, debug=True, save = False)
indi.print_elements()

rho = Rho(rho_phot, rho_spin, indi) # initial condition with zero photons and all spins up.# sys.exit()

L = Models(wc, w0,g, kappa, gamma_phi,gamma,indi, parallel=1,progress=True, debug=True,save=False, num_cpus=None)
L.setup_L_Tavis_Cummings(progress=True)


# Operators for time evolution
n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
ops = [n,p] # operators to calculate expectations for

evolve = TimeEvolve(rho, L, tmax, dt, atol=atol, rtol=rtol)
evolve.time_evolve_block_interp(ops, progress = True, expect_per_nu=True, start_block=None)
# evolve.time_evolve_chunk_parallel2(ops, chunksize=chunksize, progress=True, num_cpus=None)

ns_block = evolve.result.expect[0].real
ps_block = evolve.result.expect[1].real
ts_block = evolve.result.t

runtime = time() - t0



params = {
    'method': 'block_kirton',
    'N': ntls,
    'nphot': nphot,
    'w0': w0,
    'wc': wc,
    'Delta': wc- w0,
    'gamma': gamma,
    'gamma_phi': gamma_phi,
    'kappa': kappa,
    'Omega': Omega,
    'tmax': tmax,
    'dt': dt,
    'theta': theta
    }


fig, axes = plt.subplots(2)
axes[0].set_ylabel(r'$\langle n\rangle$')
axes[0].set_xlabel(r'$t$')
axes[0].plot(ts_block, ns_block.real,label='PIBS')
axes[0].plot(ts, ns.real, label = 'Kirton', linestyle = '--')

axes[1].plot(ts_block, ps_block.real, label='PIBS')
axes[1].plot(ts, ps.real, label='Kirton', linestyle = '--')
axes[1].set_ylabel(r'$\langle \sigma^+_i\sigma^-_i\rangle$')
axes[1].set_xlabel(r'$t$')



fig.suptitle(r'$N={N},\, \omega_c-\omega_0 = {Delta},\, g\sqrt{{N}} = {Omega}, \, \kappa={kappa},\,\gamma={gamma},\,\gamma_\phi = {gamma_phi},\,\theta={theta:.2f} $'.format(**params))

# plt.legend()
# fig.savefig('figures/example_block.png',dpi=300, bbox_inches='tight')
plt.show()








