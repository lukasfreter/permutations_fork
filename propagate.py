class Results:
    def __init__(self):
        self.rho= []
        self.t = []
        self.expect = []

class Progress:
    def __init__(self, total, description='', start_step=0):
        self.description = description
        self.step = start_step
        self.end = total-1
        self.percent = self.calc_percent()
        self.started = False

    def calc_percent(self):
        return int(100*self.step/self.end)

    def update(self, step=None):
        # print a description at the start of the calculation
        if not self.started:
            print('{}{:4d}%'.format(self.description, self.percent), end='', flush=True)
            self.started = True
            return
        # progress one step or to the specified step
        if step is None:
            self.step += 1
        else:
            self.step = step
        percent = self.calc_percent()
        # only waste time printing if % has actually increased one integer
        if percent > self.percent:
            print('\b\b\b\b\b{:4d}%'.format(percent), end='', flush=True)
            self.percent = percent
        if self.step == self.end:
            print('', flush=True)

def time_evolve(L, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5,
                progress=False, save_states=None):
    """Time evolve matrix L from initial condition initial with step dt to tend
    Default behaviour is to record compressed state matrices at each timestep if
    expect_oper is None, and to record expectations of operators in expect_oper
    (and not states) if expect_oper is not None. Use save_states to override this,
    i.e., save_states==True to always record states, save_states=False to never
    save states.

    expect_oper should be a list of operators that each either act on the photon
    (dim_lp Z dim_lp), the photon and one spin (dim_lp*dim_ls X dim_lp*dim_ls), the
    photon and two spins... etc. setup_convert_rho_nrs(X) must have been run with
    X = 0, 1, 2,... prior to the calculation.

    progress==True writes progress in % for the time evolution
    """
    from scipy.integrate import ode
    from numpy import zeros, array
    from expect import expect_comp
    
    #L=L.todense()
    
    t0 = 0
    r = ode(_intfunc).set_integrator('zvode', method='bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial, t0).set_f_params(L)
    output = Results()
    # Record initial values
    output.t.append(r.t)
    output.rho.append(initial)
    ntimes = int(tend/dt)+1
    if progress:
        bar = Progress(ntimes, description='Time evolution under L...', start_step=1)
    if save_states is None:
        save_states = True if expect_oper is None else False
    if not save_states and expect_oper is None:
        print('Warning: Not recording states or any observables. Only initial and final'\
                ' compressed state will be returned.')
    
    if expect_oper == None:
        while r.successful() and r.t < tend:
            rho = r.integrate(r.t+dt)
            if save_states:
                output.rho.append(rho)
            output.t.append(r.t)
            if progress:
                bar.update()
        return output
    else:
        output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
        output.expect[:,0] = array(expect_comp([initial], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            output.expect[:,n_t] = array(expect_comp([rho], expect_oper)).flatten()
            output.t.append(r.t)
            if save_states:
                output.rho.append(rho)
            n_t += 1
            if progress:
                bar.update()
        if not save_states:
            output.rho.append(rho) # record final state in this case (otherwise already recorded)
        return output
    
    
def time_evolve_block2(L0,L1, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5,
                progress=False, save_states=None):
    """Time evolve initial state using L0, L1 block structure Liouvillian matrices.
    This only works for weak U(1) symmetry.

    expect_oper should be a list of operators that each either act on the photon
    (dim_lp Z dim_lp), the photon and one spin (dim_lp*dim_ls X dim_lp*dim_ls), the
    photon and two spins... etc. setup_convert_rho_nrs(X) must have been run with
    X = 0, 1, 2,... prior to the calculation.

    progress==True writes progress in % for the time evolution
    """
    from scipy.integrate import ode
    from numpy import zeros, array
    from expect import expect_comp, expect_comp_block
    from indices import mapping_block
    from basis import ldim_p
    from indices import indices_elements
    
    dim_rho_compressed = ldim_p**2 * len(indices_elements)
    num_blocks = len(mapping_block)
    t0 = 0
    ntimes = int(tend/dt)+1
    output_nu2 = []
    for i in range(num_blocks):
        output_nu2.append(Results()) # create output for each block
        
    
    # first calculate block nu_max
    r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial[num_blocks-1],t0).set_f_params(L0[num_blocks-1])
    #Record initial values
    output_nu2[num_blocks-1].t.append(r.t)
    output_nu2[num_blocks-1].rho.append(rho_block_to_compressed(initial[num_blocks-1], num_blocks-1))

    
    if progress:
        bar = Progress(num_blocks*(ntimes-1), description='Time evolution under L...', start_step=1)
    if save_states is None:
        save_states = True if expect_oper is None else False
    if not save_states and expect_oper is None:
        print('Warning: Not recording states or any observables. Only initial and final'\
                ' compressed state will be returned.')
  
    if expect_oper == None:
        while r.successful() and r.t < tend:
            rho = r.integrate(r.t+dt)
            if save_states:
                output_nu2[num_blocks-1].rho.append(rho)
            output_nu2[num_blocks-1].t.append(r.t)
            if progress:
                bar.update()
    
    else:
        output_nu2[num_blocks-1].expect = zeros((len(expect_oper), ntimes), dtype=complex)
        output_nu2[num_blocks-1].expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[num_blocks-1], num_blocks-1)], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            output_nu2[num_blocks-1].expect[:,n_t] = \
                    array(expect_comp_block([rho], num_blocks-1, expect_oper)).flatten()
            output_nu2[num_blocks-1].t.append(r.t)
            if save_states:
                output_nu2[num_blocks-1].rho.append(rho_block_to_compressed(rho, num_blocks-1))
            n_t += 1
            if progress:
                bar.update()
        if not save_states:
            output_nu2[num_blocks-1].rho.append(rho_block_to_compressed(rho, num_blocks-1)) # record final state in this case (otherwise already recorded)
    
    # INCLUDE CHECK IF COUPLING TO DIFFERENT NU IS ZERO
    
    for nu in range(num_blocks-2, -1,-1):
        r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(initial[nu],t0).set_f_params(L0[nu])
        #Record initial values
        output_nu2[nu].t.append(r.t)
        output_nu2[nu].rho.append(rho_block_to_compressed(initial[nu],nu))
        
        if expect_oper == None:
            while r.successful() and r.t < tend:
                rho = r.integrate(r.t+dt)
                if save_states:
                    output_nu2[nu].rho.append(rho_block_to_compressed(rho,nu))
                output_nu2[nu].t.append(r.t)
                if progress:
                    bar.update()
        
        else:
            output_nu2[nu].expect = zeros((len(expect_oper), ntimes), dtype=complex)
            output_nu2[nu].expect[:,0] = array(expect_comp_block([initial[nu]], nu, expect_oper)).flatten()
            n_t=1
            while r.successful() and n_t<ntimes:
                rho = r.integrate(r.t+dt)
                output_nu2[nu].expect[:,n_t] = array(expect_comp_block([rho], nu, expect_oper)).flatten()
                output_nu2[nu].t.append(r.t)
                if save_states:
                    output_nu2[nu].rho.append(rho_block_to_compressed(rho,nu))
                n_t += 1
                if progress:
                    bar.update()
            if not save_states:
                output_nu2[nu].rho.append(rho) # record final state in this case (otherwise already recorded)

        
    
    return output_nu2[num_blocks-1]
    
    


def time_evolve_block(L0,L1, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5,
                progress=False, save_states=None):
    """Time evolve initial state using L0, L1 block structure Liouvillian matrices.
    This only works for weak U(1) symmetry.

    expect_oper should be a list of operators that each either act on the photon
    (dim_lp Z dim_lp), the photon and one spin (dim_lp*dim_ls X dim_lp*dim_ls), the
    photon and two spins... etc. setup_convert_rho_nrs(X) must have been run with
    X = 0, 1, 2,... prior to the calculation.

    progress==True writes progress in % for the time evolution
    """
    from scipy.integrate import ode
    from numpy import zeros, array
    from expect import expect_comp, expect_comp_block
    from indices import mapping_block
    from basis import ldim_p
    from indices import indices_elements
    
    dim_rho_compressed = ldim_p**2 * len(indices_elements)
    num_blocks = len(mapping_block)
    t0 = 0
    ntimes = int(tend/dt)+1
        
    output = Results()
    rhos= [[] for _ in range(num_blocks)] # store all rho for feed forward
    
    # first calculate block nu_max
    nu = num_blocks - 1
    r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial[nu],t0).set_f_params(L0[nu])
    #Record initial values
    output.t.append(r.t)
    rhos[nu].append(initial[nu])
    
    if progress:
        bar = Progress(ntimes, description='Time evolution under L...', start_step=1)
    if save_states is None:
        save_states = True if expect_oper is None else False
    if not save_states and expect_oper is None:
        print('Warning: Not recording states or any observables. Only initial and final'\
                ' compressed state will be returned.')
            
    
    # FOR LATER: what if expect_oper = []
    
    # if expect_oper == None:
    #     while r.successful() and r.t < tend:
    #         rho = r.integrate(r.t+dt)
    #         if save_states:
    #             output_nu[nu].rho.append(rho)
    #         output_nu[nu].t.append(r.t)
    #         if progress:
    #             bar.update()
    
    # else:
    #output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
    #output.expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[nu], nu)], expect_oper)).flatten()
    n_t=1
    while r.successful() and n_t<ntimes:
        rho = r.integrate(r.t+dt)
      #  output.expect[:,n_t] = array(expect_comp([rho_block_to_compressed(rho,nu)], expect_oper)).flatten()
        output.t.append(r.t)
        rhos[nu].append(rho)
        n_t += 1
        if progress:
            bar.update()
        #if not save_states:
        #    output_nu[nu].rho.append(rho) # record final state in this case (otherwise already recorded)
    
    
    # # INCLUDE CHECK IF COUPLING TO DIFFERENT NU IS ZERO
    
    # Now, do the feed forward for all other blocks. Need different integration function,
    # that for this -> _intfunc_block
    
    for nu in range(num_blocks-2, -1,-1):
        r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(initial[nu],t0).set_f_params(L0[nu], L1[nu], rhos[nu+1][0])
        #Record initial values
        rhos[nu].append(initial[nu])
        
        
        # FOR LATER
        # if expect_oper == None:
        #     while r.successful() and r.t < tend:
        #         rho = r.integrate(r.t+dt)
        #         if save_states:
        #             output_nu[nu].rho.append(rho)
        #         output_nu[nu].t.append(r.t)
        #         if progress:
        #             bar.update()
        
        # else:
            #output_nu[nu].expect = zeros((len(expect_oper), ntimes), dtype=complex)
        #output_nu[nu].expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[nu],nu)], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            #output_nu[nu].expect[:,n_t] = array(expect_comp([rho_block_to_compressed(rho,nu)], expect_oper)).flatten()
            #output_nu[nu].t.append(r.t)
            #if save_states:
            rhos[nu].append(rho)
            # update integrator:
            r = ode(_intfunc_block).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol).set_initial_value(r.y,r.t).set_f_params(L0[nu],L1[nu],rhos[nu+1][n_t])
            n_t += 1

            if progress:
                bar.update()
            
        # if not save_states:
            # output_nu[nu].rho.append(rho) # record final state in this case (otherwise already recorded)
    
    # Now with all rhos, I can calculate the expectation values:
    output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
    # for t_idx in range(ntimes):
    #     # build density matrix out of blocks
    #     rho_compressed = zeros(dim_rho_compressed, dtype=complex)
    #     for nu in range(num_blocks):
    #         rho_compressed[mapping_block[nu]] = rhos[nu][t_idx]
    #     output.expect[:,t_idx] = array(expect_comp([rho_compressed], expect_oper)).flatten()
            
    for t_idx in range(ntimes):
        for nu in range(num_blocks):
            output.expect[:,t_idx] = output.expect[:,t_idx] +  array(expect_comp_block([rhos[nu][t_idx]],nu, expect_oper)).flatten()

        
    
    return output
    
    
    
    
def time_evolve_block1(L0,L1, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5,
                progress=False, save_states=None):
    """ Same as time_evolve_block, but only for the nu_max block. This is useful
    if one knows that there is no coupling to other blocks."""
    from scipy.integrate import ode
    from numpy import zeros, array
    from expect import expect_comp
    from indices import mapping_block
    from basis import ldim_p
    from indices import indices_elements
    
    dim_rho_compressed = ldim_p**2 * len(indices_elements)
    num_blocks = len(mapping_block)
    #print(num_blocks)
    t0 = 0
    ntimes = int(tend/dt)+1
    output= Results()
        
    
    # first calculate block nu_max
    nu = num_blocks - 1
    r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial[nu],t0).set_f_params(L0[nu])
    #Record initial values
    output.t.append(r.t)
    output.rho.append(initial[nu])

    
    if progress:
        bar = Progress(num_blocks*(ntimes-1), description='Time evolution under L...', start_step=1)
    if save_states is None:
        save_states = True if expect_oper is None else False
    if not save_states and expect_oper is None:
        print('Warning: Not recording states or any observables. Only initial and final'\
                ' compressed state will be returned.')
  
    if expect_oper == None:
        while r.successful() and r.t < tend:
            rho = r.integrate(r.t+dt)
            if save_states:
                output.rho.append(rho)
            output.t.append(r.t)
            if progress:
                bar.update()
        return output
    else:
        output.expect = zeros((len(expect_oper), ntimes), dtype=complex)
        output.expect[:,0] = array(expect_comp([rho_block_to_compressed(initial[nu], nu)], expect_oper)).flatten()
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            output.expect[:,n_t] = array(expect_comp([rho_block_to_compressed(rho,nu)], expect_oper)).flatten()
            output.t.append(r.t)
            if save_states:
                output.rho.append(rho)
            n_t += 1
            if progress:
                bar.update()
        if not save_states:
            output.rho.append(rho) # record final state in this case (otherwise already recorded)
    
        return output
    
    
    
    
    
    
    
def time_evolve_block_interp(L0,L1, initial, tend, dt, expect_oper=None, atol=1e-5, rtol=1e-5,
                progress=False, save_states=None):
    """ Time evolution of the block structure without resetting the solver at each step.
    Do so by interpolating feedforward.
    
    method: 'bdf' = stiff, 'adams' = non-stiff
    
    """
    from scipy.integrate import ode
    from numpy import zeros, array
    from expect import expect_comp_block
    from indices import mapping_block
    from basis import ldim_p
    from indices import indices_elements
    from time import time
    from scipy.interpolate import interp1d
    
    print('Starting time evolution serial block (interpolation)...')
    tstart = time()
           
    # store number of elements in each block
    num_blocks = len(mapping_block)
    blocksizes = [len(mapping_block[nu]) for nu in range(num_blocks)]
    nu_max = num_blocks -1
    t0 = 0
    ntimes = round(tend/dt)+1
    
    if progress:
        bar = Progress(2*(ntimes-1)*num_blocks, description='Time evolution under L...', start_step=1)
    if save_states is None:
        save_states = True if expect_oper is None else False
    if not save_states and expect_oper is None:
        print('Warning: Not recording states or any observables. Only initial and final'\
                ' compressed state will be returned.')
            

    # set up results:
    result = Results()
    result.t = zeros(ntimes)
    if save_states:
        result.rho = [zeros((blocksizes[nu], ntimes), dtype=complex) for nu in range(num_blocks)]
    else: # only record initial and final states
        result.rho = [zeros((blocksizes[nu], 2), dtype=complex) for nu in range(num_blocks)]
        
    if expect_oper is not None:
        result.expect = zeros((len(expect_oper), ntimes), dtype=complex)
    
    #Record initial values
    result.t[0] = t0            
    
    # first calculate block nu_max. Setup integrator
    r = ode(_intfunc).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
    r.set_initial_value(initial[nu_max],t0).set_f_params(L0[nu_max])
    
    # temporary variable to store states
    #rhos = [ np.zeros((len(self.indices.mapping_block[i]), ntimes), dtype=complex) for i in range(num_blocks)]
    #rhos[nu][:,0] = self.rho.initial[nu]

    rho_nu = zeros((blocksizes[nu_max], ntimes), dtype = complex)
    rho_nu[:,0] = initial[nu_max] 
    
    # using the exact solver times for the feedforward interpolation instead of using linearly spaced time array makes a (small) difference
    solver_times = zeros(ntimes)
    solver_times[0] = t0

    n_t=1
    while r.successful() and n_t<ntimes:
        rho = r.integrate(r.t+dt)
        result.t[n_t] = r.t
        solver_times[n_t] = r.t
        #rhos[nu][:,n_t] = rho
        rho_nu[:,n_t] = rho
        n_t += 1
        
        if progress:
            bar.update()
            
    # calculate nu_max part of expectation values
    if expect_oper is not None:
        # if progress:
        #     bar = Progress(ntimes, description='Calculating expectation values...', start_step=1)
            
        for t_idx in range(ntimes):
            #self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
            result.expect[:,t_idx] +=  array(expect_comp_block([rho_nu[:,t_idx]],nu_max, expect_oper)).flatten()
            if progress:
                bar.update()
    if save_states:
        result.rho[nu_max] = rho_nu
    else: # store initial and final states
        result.rho[nu_max][:,0] = rho_nu[:,0]
        result.rho[nu_max][:,1] = rho_nu[:,-1] 
        
    #self.result.t = np.arange(t0, self.tend+self.dt,self.dt)

    
    # Now, do the feed forward for all other blocks. Need different integration function, _intfunc_block_interp
    for nu in range(num_blocks-2, -1,-1):           
        #rho_interp = interp1d(self.result.t, rhos[nu+1], bounds_error=False, fill_value="extrapolate") # extrapolate results from previous block
        rho_interp = interp1d(solver_times, rho_nu, bounds_error=False, fill_value="extrapolate") # interpolate results from previous block, rho_nu                  
                   
        r = ode(_intfunc_block_interp).set_integrator('zvode', method = 'bdf', atol=atol, rtol=rtol)
        r.set_initial_value(initial[nu],t0).set_f_params(L0[nu], L1[nu], rho_interp)
        
        #Record initial value
        #rhos[nu][:,0] = (self.rho.initial[nu])
        # Update rho_nu variable for current block
        rho_nu = zeros((blocksizes[nu], ntimes), dtype=complex)
        rho_nu[:,0] = initial[nu]
        solver_times[0] = t0
        
        # integrate
        n_t=1
        while r.successful() and n_t<ntimes:
            rho = r.integrate(r.t+dt)
            #rhos[nu][:,n_t] = rho
            solver_times[n_t] = r.t
            rho_nu[:,n_t] = rho
            n_t += 1

            if progress:
                bar.update()
                
        # calculate contribution of block nu to expectation values
        if expect_oper is not None:
            for t_idx in range(ntimes):
                #self.result.expect[:,t_idx] +=  np.array(self.expect_comp_block([rhos[nu][:,t_idx]],nu, expect_oper)).flatten()
                result.expect[:,t_idx] +=  array(expect_comp_block([rho_nu[:,t_idx]],nu, expect_oper)).flatten()
                if progress:
                    bar.update()
        if save_states:
            result.rho[nu] = rho_nu
        else:
            result.rho[nu][:,0] = rho_nu[:,0]
            result.rho[nu][:,1] = rho_nu[:,-1] 

    elapsed = time()-tstart
    print(f'Complete {elapsed:.0f}s', flush=True)
    return result
    
    
    



def rho_block_to_compressed(rho, nu):
    """ Calculate the compressed density matrix from the block density matrix"""
    from indices import mapping_block
    from basis import ldim_p
    from indices import indices_elements
    from numpy import zeros
    
    dim_rho_compressed = ldim_p**2 * len(indices_elements)
    num_blocks = len(mapping_block)
    rho_compressed = zeros(dim_rho_compressed,dtype='complex')
    rho_compressed[mapping_block[nu]] = rho
    
    return rho_compressed
    

def _intfunc(t, y, L):
    return (L.dot(y))

def _intfunc_block(t,y, L0, L1, y1):
    """ For blocks in block structure that couple do different excitation, described
    by L1 and y1"""    
    return(L0.dot(y) + L1.dot(y1))

def _intfunc_block_interp(t,y,L0,L1,y1_func):
    """ Same as _intfunc_block, but where y1 is given as a function of time"""
    return (L0.dot(y) + L1.dot(y1_func(t)))

def steady(L, init=None, maxit=1e6, tol=None):
    
    """calculate steady state of L using sparse eignevalue solver"""

    rho = find_gap(L, init, maxit, tol, return_ss=True)   

    return rho
    
def find_gap(L, init=None, maxit=1e6, tol=None, return_ss=False, k=10):
    """Calculate smallest set of k eigenvalues of L"""
    
    from numpy import sort
    from scipy.sparse.linalg import eigs
    from operators import tensor, qeye
    from basis import ldim_s, ldim_p
    from expect import expect_comp
    import gc
    
    if tol is None:
        tol = 1e-8
    
    gc.collect()
    # pfw: use ARPACK shift-invert mode to find eigenvalues near 0
    val, rho = eigs(L, k=k, sigma=0, which = 'LM', maxiter=maxit, v0=init, tol=tol)
    # N.B. unreliable to find mutliple eigenvalues, see https://github.com/scipy/scipy/issues/13571
    gc.collect()

    #shift any spurious positive eignevalues out of the way
    for count in range(k):
        if val[count]>1e-10:
            val[count]=-5.0

    sort_perm = val.argsort()
    val = val[sort_perm]

    rho= rho[:, sort_perm]
    
    #calculate steady state and normalise
    if (return_ss):
        rho = rho[:,k-1]
        rho = rho/expect_comp([rho], [tensor(qeye(ldim_p), qeye(ldim_s))])
        rho = rho[0,:]
        return rho
    
    else:
        return val

#calculate spectrum of <op1(t)op2(0)> using initial state op2*rho
#op2 should be a matrix in the full permutation symmetric space
#op1 should be an operator 
def spectrum(L, rho, op1, op2, tlist, ncores):
    
    import scipy.fftpack
    import numpy as np
    
    N = len(tlist)
    dt = tlist[1] - tlist[0]
    
    corr = corr_func(L, rho, op1, op2, tlist[-1], dt, ncores)
    

    F = scipy.fftpack.fft(corr)

    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)

    # select only indices for elements that corresponds
    # to positive frequencies
    indices = np.where(f > 0.0)

    omlist  = 2 * np.pi * f[indices]
    spec =  2 * dt * np.real(F[indices])
    
    

    return spec, omlist


def corr_func(L, rho, op1, op2, tend, dt, ncores, op2_big=False):
    
    from basis import setup_op
    if not op2_big:
        op2 = setup_op(op2, ncores)
    init =  op2.dot(rho) # need to define this in expec
    corr = time_evolve(L, init, tend, dt, [op1])
    return corr.expect[0]
