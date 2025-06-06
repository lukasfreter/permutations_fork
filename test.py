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
from models import setup_Dicke_block, setup_Dicke, setup_Dicke_block1
from propagate import time_evolve, time_evolve_block, time_evolve_block1, time_evolve_block2, time_evolve_block_interp
from expect import setup_convert_rho_nrs, setup_convert_rho_block_nrs
import pickle
import operators
import scipy.sparse as sp

# plt.rcParams.update({'font.size': 18,
#                      'xtick.labelsize' : 18,
#                      'ytick.labelsize' : 18,
#                      'lines.linewidth' : 3,
#                      'lines.markersize': 10,
#                      'figure.figsize': (14,10),
#                      'figure.dpi': 150})
    

ntls =5#number 2LS
nphot = ntls+1# photon fock space truncation
tmax = 50.0
dt = 0.2 # timestep

w0 = 0.0
wc = 0.0
Omega = 0.4
g = Omega / np.sqrt(ntls)
kappa = 0.1
gamma = 1e-01
gamma_phi= 0.1#2e-01



################### INITIAL PHOTON AND SPIN STATES ##################
# rotation matrix around x-axis of spin 1/2 : exp(-i*theta*Sx)=exp(-i*theta/2*sigmax) = cos(theta/2)-i*sin(theta/2)*sigmax
theta = 0.0
rot_x = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
rot_x_dag = np.array([[np.cos(theta/2), 1j*np.sin(theta/2)],[1j*np.sin(theta/2), np.cos(theta/2)]])

rho_phot = basis(nphot,0)
rho_spin = sp.csr_matrix(rot_x @ basis(2,0) @ rot_x_dag)


print('Initial states:')
print('Spin:\n',rho_spin.todense())
print('Photon:\n',rho_phot.todense())


################# BLOCK STRUCTURE ####################################
# SETUP
t0_tot = time()
print(f'Number of spins {ntls}')
print('Block form optimized')
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
setup_mapping_block(parallel=False)       # setup mapping between compressed density matrix and block form
# setup_convert_rho_nrs(2)   # conversion matrix from full to photon + single-spin RDM
# setup_convert_rho_block_nrs(2)
setup_convert_rho_block_nrs(1)
#sys.exit()

# Initial state
t0 = time()
initial_block = setup_rho_block(rho_phot,rho_spin)
print('setup initial state block in {:.1f}s'.format(time()-t0), flush=True)
#initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
t0=time()
L0,L1 = setup_Dicke_block1(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi, gamma)
print('setup L block in {:.1f}s'.format(time()-t0), flush=True)

# import numpy as np
# for i in range(len(indi.mapping_block)):
#     print(i)
#     assert np.allclose(L0[i].todense(), L.L0[i].todense())


n = tensor(create(nphot)*destroy(nphot), qeye(2))
n2 = tensor(create(nphot)*destroy(nphot)*create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
# correlations:
sigp_sigm_ij = tensor(qeye(nphot), sigmap(), sigmam())
a_sigp = tensor(destroy(nphot), sigmap())



ops = [n,p,n2] # operators to calculate expectations for

# ops = [n,p, a_sigp] # operators to calculate expectations for

t0=time()
resultscomp_block = time_evolve_block_interp(L0,L1,initial_block, tmax, dt, ops, atol=1e-15, rtol=1e-15, save_states=False)
# resultscomp_block = time_evolve_block(L0,L1,initial_block, tmax, dt, ops, atol=1e-15, rtol=1e-15, save_states=False)

# if save_states=False, only operator expectations and initial, final density matrices are recorded
runtime=time()-t0_tot
print('Time evolution Block complete in {:.0f}s'.format(runtime), flush=True)


# SETUP, block structure without optimized calculate_L_line
# print('-----------------')
# print('Block form not optimized')
# setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
# list_equivalent_elements() # create mapping to/from unique spin states
# setup_mapping_block(parallel=True)       # setup mapping between compressed density matrix and block form
# setup_convert_rho_nrs(1)   # conversion matrix from full to photon + single-spin RDM


# # Initial state
# t0 = time()
# initial_block = setup_rho_block(basis(nphot,0),basis(2,0))
# print('setup initial state block in {:.1f}s'.format(time()-t0), flush=True)
# #initial = setup_rho(basis(nphot, 0), basis(2,0)) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
# #sys.exit()
# t0=time()
# L0,L1 = setup_Dicke_block(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi/4, gamma)
# print('setup L block in {:.1f}s'.format(time()-t0), flush=True)

# n = tensor(create(nphot)*destroy(nphot), qeye(2))
# p = tensor(qeye(nphot), sigmap()*sigmam())
# ops = [n,p] # operators to calculate expectations for

# t0=time()
# resultscomp_block = time_evolve_block1(L0,L1,initial_block, tmax, dt, ops, atol=1e-8, rtol=1e-8, save_states=True)
# # if save_states=False, only operator expectations and initial, final density matrices are recorded
# runtime=time()-t0
# print('Time evolution Block complete in {:.0f}s'.format(runtime), flush=True)



# RESULTS
ts_block = np.array(resultscomp_block.t)
ns_block = np.array(resultscomp_block.expect[0])
ps_block = np.array(resultscomp_block.expect[1])
n2_block = np.array(resultscomp_block.expect[2])
g2_block = (n2_block[1:] - ns_block[1:]) / ns_block[1:]**2
# sigp_sigm_ij_block = np.array(resultscomp_block.expect[3])
# a_sigp_block = np.array(resultscomp_block.expect[2])


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

res = {
    't':ts_block,
    'e_phot_tot': ns_block,
    'e_excit_site': ps_block,  
    'e_phot_n2': n2_block
        }
data = {
        'params': params,
        'results': res,
        'runtime': runtime}

fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}.pkl'
fname = f'results/{params["method"]}.pkl'
# save results in pickle file
# with open(fname, 'wb') as handle:
#     pickle.dump(data,handle)
# print('Saved as ',fname)


# sys.exit()

############################# OLD CODE #####################################
print('-----------------')
print('Full form')

# SETUP
setup_basis(ntls, 2,nphot) # defines global variables for backend ('2' for two-level system)
list_equivalent_elements() # create mapping to/from unique spin states
# setup_convert_rho_nrs(2) # conversion matrix from full to photon + single-spin RDM
setup_convert_rho_nrs(1) # conversion matrix from full to photon + single-spin RDM

# rho_spin[0,1] = 0
# rho_spin[1,0] = 0
t0 = time()
initial = setup_rho(rho_phot, rho_spin) # initial state in compressed representation, 0 photons, spin UP (N.B. TLS vs Pauli ordering of states)
print('setup initial state full in {:.1f}s'.format(time()-t0), flush=True)


t0=time()
L = setup_Dicke(wc, w0/2, 0.0, g, 0.0, kappa, gamma_phi, gamma, progress=False, parallel=True)
print('setup L in {:.1f}s'.format(time()-t0), flush=True)

n = tensor(create(nphot)*destroy(nphot), qeye(2))
p = tensor(qeye(nphot), sigmap()*sigmam())
sigp_sigm_ij = tensor(qeye(nphot), sigmap(), sigmam())
a_sigp = tensor(destroy(nphot), sigmap())
ops = [n,p, n2] # operators to calculate expectations for
# ops = [n,p, a_sigp] # operators to calculate expectations for

# PROPAGATE
t0=time()
resultscomp = time_evolve(L, initial, tmax, dt, ops, atol=1e-15, rtol=1e-15, save_states=False)
runtime=time()-t0
print('Time evolution complete in {:.0f}s'.format(runtime), flush=True)

# rho_block = resultscomp_block.rho
# rho_ref= resultscomp.rho



# RESULTS
ts = np.array(resultscomp.t)
ns = np.array(resultscomp.expect[0])
ps = np.array(resultscomp.expect[1])
n2s = np.array(resultscomp.expect[2])
g2_perm = (n2s[1:] - ns[1:]) / ns[1:]**2

# sigp_sigm_ij_kirton = np.array(resultscomp.expect[3])

# a_sigp_kirton = np.array(resultscomp.expect[2])


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
    'tmax': tmax,
    'dt': dt,
    'theta': theta
    }

res = {
    't':ts,
    'e_phot_tot': ns,
    'e_excit_site': ps,  
    'e_phot_n2': n2s
        }
data = {
        'params': params,
        'results': res,
        'runtime': runtime}

fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}.pkl'
fname = f'results/{params["method"]}.pkl'
# save results in pickle file
# with open(fname, 'wb') as handle:
#     pickle.dump(data,handle)
# print('Saved as ',fname)


fig,ax = plt.subplots()
ax.plot(ts_block[1:], g2_block, label='block')
ax.plot(ts[1:], g2_perm, label='perm', ls='--')
plt.show()



sys.exit()
###############################
# DIFFERENCES
##############################

# fig, ax = plt.subplots(3, figsize=(8,10))
# fig.suptitle(r'$N={N},\, \omega_c-\omega_0 = {Delta},\, g\sqrt{{N}} = {Omega}, \, \kappa={kappa},\,\gamma={gamma},\,\gamma_\phi = {gamma_phi},\,\theta={theta:.2f} $'.format(**params))

# ax[0].plot(ts_block, (ns_block - ns))
# ax[1].plot(ts_block, (ps_block - ps))
# # ax[2].plot(ts_block[1:], g2_block)
# # ax[2].plot(ts[1:], g2_perm)

# ax[2].plot(ts_block, ns_block, label='block')
# ax[2].plot(ts, ns, ls='--', label='permutation')
# # ax[2].plot(ts_block, ps_block, label='block')
# # ax[2].plot(ts, ps, ls='--', label='permutation')
# ax[2].legend()

# ax[0].set_ylabel('difference n')
# ax[1].set_ylabel('difference spin')
# ax[2].set_ylabel('g2(0)')
# for a in ax:
#     a.set_xlabel('t')
# plt.tight_layout()
# plt.show()
# sys.exit()







fig, axes = plt.subplots(3,2)
axes[0,0].set_ylabel(r'$n$')
axes[0,0].plot(ts_block, ns_block.real,label='Block',linestyle = '-', color='tab:blue')
axes[0,0].plot(ts, ns.real, label = 'Reference', linestyle = '--', color='tab:orange')
axes[0,1].plot(ts_block, ps_block.real, label='Block',linestyle = '-', color='tab:blue')
axes[0,1].plot(ts, ps.real, label='Reference', linestyle = '--', color='tab:orange')
axes[0,1].set_ylabel(r'$\langle \sigma^+_i\sigma^-_i\rangle$')

axes[1,0].plot(ts_block, a_sigp_block.real, label='Block', linestyle = '-', color='tab:blue')
axes[1,0].plot(ts, a_sigp_kirton.real, label='Reference', linestyle = '--', color='tab:orange')
axes[1,0].set_ylabel(r'RE$\langle a\sigma^+_i\rangle$')

axes[1,1].plot(ts_block, a_sigp_block.imag, label='Block', linestyle = '-', color='tab:blue')
axes[1,1].plot(ts, a_sigp_kirton.imag, label='Reference', linestyle = '--', color='tab:orange')
axes[1,1].set_ylabel(r'IM$\langle a\sigma^+_i\rangle$')

axes[2,0].plot(ts_block, sigp_sigm_ij_block.real, label='Block', linestyle = '-', color='tab:blue')
axes[2,0].plot(ts, sigp_sigm_ij_kirton.real, label='Kirton', linestyle = '--', color='tab:orange')
axes[2,0].set_ylabel(r'$\langle \sigma^+_i\sigma^-_j\rangle$')
axes[2,0].legend()

fig.suptitle(r'$N={N},\, \omega_c-\omega_0 = {Delta},\, g\sqrt{{N}} = {Omega}, \, \kappa={kappa},\,\gamma={gamma},\,\gamma_\phi = {gamma_phi},\,\theta={theta:.2f} $'.format(**params))

# plt.legend()
fig.savefig('figures/example_block.png',dpi=300, bbox_inches='tight')
plt.show()
sys.exit()

# # store results

# res = {
#     't':ts_block,
#     'e_phot_tot': ns_block,
#     'e_excit_site': ps_block,    
#        }
# data = {
#         'params': params,
#         'results': res,
#         'runtime': runtime}

# fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}.pkl'
# fname = f'results/{params["method"]}.pkl'
# # save results in pickle file
# with open(fname, 'wb') as handle:
#     pickle.dump(data,handle)
    
# # store results
# params = {
#     'method': 'reference_block_kirton',
#     'N': ntls,
#     'nphot': nphot,
#     'w0': w0,
#     'wc': wc,
#     'Delta': wc- w0,
#     'gamma': gamma,
#     'gamma_phi': gamma_phi,
#     'kappa': kappa,
#     'Omega': Omega,
#     'tmax': tmax,
#     'dt': dt,
#     'theta': 0.0
#     }
# res = {
#     't':ts,
#     'e_phot_tot': ns,
#     'e_excit_site': ps,    
#        }
# data = {
#         'params': params,
#         'results': res,
#         'runtime': runtime}

# fname = f'results/{params["method"]}_N{ntls}_Delta{params["Delta"]}_Omega{Omega}_kappa{kappa}_gamma{gamma}_gammaphi{gamma_phi}.pkl'
# fname = f'results/{params["method"]}.pkl'
# # save results in pickle file
# with open(fname, 'wb') as handle:
#     pickle.dump(data,handle)




# # # get relative deviation
# # dev_n = (ns - ns_block) / ns
# # dev_ps = (ps - ps_block) / ps
# # fig, axes = plt.subplots(1, figsize=(6,3))
# # axes.set_xlabel(r'$t$')
# # axes.plot(ts_block, dev_n,label='rel. deviation n')
# # axes.plot(ts_block, dev_ps, label = 'rel. deviation exc. number')
# # plt.legend()
# # #fig.savefig('figures/example_block.png',dpi=300, bbox_inches='tight')
# # plt.show()












