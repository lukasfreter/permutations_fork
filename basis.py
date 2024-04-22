ldim_s = []
ldim_p = []
nspins = []

def setup_basis(ns, ls, lp):
    """Define global variables"""
    
    from indices import list_equivalent_elements
    from expect import setup_convert_rho

    # basic input checks
    assert isinstance(ls, int) and ls > 0, "Photon dimension must be positive int"
    assert isinstance(lp, int) and lp > 1, "Spin dimension must be int greater than 1"
    assert isinstance(ns, int) and ns > 0, "Number of spins must be positive int"
    
    # set global variables
    global ldim_s, ldim_p, nspins
    ldim_s = ls
    ldim_p = lp
    nspins = ns
    
def setup_L(H, c_ops, num_threads, progress=False, parallel=False):
    
    """Generate generic Liouvillian for Hamiltonian H and 
    collapse operators c_ops. Use num_threads processors for 
    parallelisation.
    Note c_ops must be a list (even with only one element)"""
    
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    
    num_elements = len(indices_elements)
    n_cops = len(c_ops)
    
    #precalculate Xdag*X and Xdag
    c_ops_2 = []
    c_ops_dag = []
    for count in range(n_cops):
        c_ops_2.append((c_ops[count].T.conj()*c_ops[count]).todense())
        c_ops_dag.append((c_ops[count].T.conj()).todense())
        c_ops[count] = c_ops[count].todense()
        
    Hfull = H.todense()
    #Hfull[0,3]=0.8
    #print(Hfull)
    ############ DELETE THIS DIRECTLY AFTER DEBUGGING#################
    # import numpy as np
    #Hfull = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    # Hfull = np.array([[1,2],[3,4]])
    #print(Hfull)
    ########################################################
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, c_ops, c_ops_2, c_ops_dag, ldim_p*ldim_p*num_elements))
        
    #print(len(arglist))
    #parallel version
    if parallel:
        if num_threads == None:
            pool = Pool()
        else:
            pool = Pool(num_threads)
        #find all the rows of L
        L_lines = []
        if progress:
            print('Constructing Liouvillian L...')
            try:
                import tqdm
                for line in tqdm.tqdm(pool.imap(calculate_L_fixed, arglist), total=len(arglist)):
                    L_lines.append(line)
            except:
                print('Package tqdm required for progress bar in parallel version')
                pass
        if len(L_lines) == 0:
            L_lines = pool.imap(calculate_L_fixed, arglist)
        pool.close()
        #combine into a big matrix                    
        L = vstack(L_lines)
        return L
    
    if progress:
        from propagate import Progress
        bar = Progress(ldim_p**2 * num_elements, description='Constructing Liouvillian L...')

    #serial version
    L_lines = []
    for count in range(ldim_p*ldim_p*len(indices_elements)):
        #print(f'Element: {arglist[count][0]}')
        L_lines.append(calculate_L_fixed(arglist[count]))
        if progress:
            bar.update()
    
    #combine into a big matrix                    
    L = vstack(L_lines)
    
    return L
    
def calculate_L_fixed(args):
    return calculate_L_line(*args)

#@profile
def calculate_L_line(element, H, c_ops, c_ops_2, c_ops_dag, length):
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import zeros, concatenate, copy
    from scipy.sparse import lil_matrix, csr_matrix
    
    n_cops = len(c_ops)
    
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    tol = 1e-10
   # print(f'\nIn calculate Line: left={left}, right={right}')
    L_line = zeros((1, length), dtype = complex)
    

        
    for count_phot in range(ldim_p):
       # print(f'count phot: {count_phot}')
        for count_s in range(ldim_s):
            #print(f'count s: {count_s}')
            for count_ns in range(nspins):
                #print(f'count ns: {count_ns}')
                #print(f'\ncount_phot={count_phot}, count_s={count_s}, count_ns={count_ns}')
                #keep track of if we have done the n1/n2 calculations
                n1_calc = False
                n2_calc = False
                    
                #calculate appropriate matrix elements of H
                Hin = get_element(H, [left[0], left[count_ns+1]], [count_phot, count_s])
               # print(f'Hin: {Hin}')
                #print(f'ELEMENT left: {ldim_s*left[0] + left[count_ns+1]}, {ldim_s*count_phot + count_s}')
                #only bother if H is non-zero
                if abs(Hin)>tol:
                    
                    #print('\n')
                    #work out which elements of rho this couples to
                    #note the resolution of identity here is small because H only acts between photon and one spin
                    n1_element = copy(left)
                    n1_element[0] = count_phot
                    n1_element[count_ns+1] = count_s
                    n1_calc = True
                    
                    #get the indices of the equivalent element to the one which couples
                    spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                    rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                   # print(f'Hin={Hin},spinnj={spinnj},rhonj={rhonj}')

                    
                    #increment L
                    L_line[0, rhonj] = L_line[0, rhonj] -1j * Hin
                    
                #same for other part of commutator
                Hnj = get_element(H, [count_phot, count_s], [right[0], right[count_ns+1]])
               # print(f'Hnj: {Hnj}')    
                if abs(Hnj)>tol:
                    n2_element = copy(right)
                    n2_element[0] = count_phot
                    n2_element[count_ns+1] = count_s
                    n2_calc = True
                    
                    spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
                    rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
                   # print(f'Hnj={Hnj},spinin={spinin},rhoin={rhoin}')
                    
                    L_line[0, rhoin] = L_line[0, rhoin] + 1j * Hnj
                    
                for count_cop in range(n_cops):
                        
                    #Do the same as above for each collapse operator
                    Xin = get_element(c_ops_2[count_cop], [left[0], left[count_ns+1]], [count_phot, count_s])
                    if abs(Xin)>tol:
                        if not(n1_calc):
                            n1_element = copy(left)
                            n1_element[0] = count_phot
                            n1_element[count_ns+1] = count_s
                            n1_calc = True
                                
                            spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                            rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                            
                        L_line[0, rhonj] = L_line[0, rhonj] - 0.5*Xin
                        
                    Xnj = get_element(c_ops_2[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
                    if abs(Xnj)>tol:
                        if not(n2_calc):
                            n2_element = copy(right)
                            n2_element[0] = count_phot
                            n2_element[count_ns+1] = count_s
                            n2_calc = True
                    
                            spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
                            rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
                        L_line[0, rhoin] = L_line[0, rhoin] - 0.5*Xnj
                        
                    Xdagnj = get_element(c_ops_dag[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
                    #only need to calculate if Xdag is non-zero
                    if abs(Xdagnj)>tol:
                        for count_phot2 in range(ldim_p):
                            for count_s2 in range(ldim_s):
                                #The term XpXdag requires two resolutions of unity
                                Xim = get_element(c_ops[count_cop], [left[0], left[count_ns+1]], [count_phot2, count_s2])
                                if abs(Xim)>tol:
                                    m1_element = copy(left)
                                    m1_element[0] = count_phot2
                                    m1_element[count_ns+1] = count_s2
                                        
                                    if not(n2_calc):
                                        n2_element = copy(right)
                                        n2_element[0] = count_phot
                                        n2_element[count_ns+1] = count_s
                                        n2_calc = True
                                            
                                    spinmn = indices_elements_inv[get_equivalent_dm_tuple(concatenate((m1_element[1:], n2_element[1:])))]
                                    rhomn = (length//ldim_p)*m1_element[0] + length//(ldim_p*ldim_p)*n2_element[0] + spinmn
                                    L_line[0, rhomn] = L_line[0, rhomn] + Xim*Xdagnj 

    L_line = csr_matrix(L_line)
    return L_line



def setup_L_block(H, c_ops,num_threads, progress=False, parallel=False):
    
    """ Generate Liouvillian for Dicke Hamiltonian 
        H = omega*adag*a + sum_n{ omega0*sz_n  + g*(a*sp_n + adag*sm_n) }
        with collective loss c_ops = kappa L[a]  and individual loss
        c_ops = sum_n { gam_phi L[sigmaz_n] + gam_dn L[sigmam_n] }
        in Block structure. Blocks are denoted by their total excitation
        nu = n_phot + n_up (n_up is number of excited spins)
        Block labeled nu only couples to itself and to the block nu+1 
        (due to weak U(1) symmetry)
        
        For now: use the same function as in setup_L, but in the end use the 
        mapping_block to choose the elements important for us. This allows us
        to use the existing code as much as possible, but might be a bit inefficient.
        
        For later, a better way might be to loop through all nu, then through all
        spin elements, and from there calculate the photon number needed to satisfy
        the total excitation nu.

    """
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple, mapping_block
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    import sys
    
    num_elements = len(indices_elements)
    num_blocks = len(mapping_block)
    n_cops = len(c_ops)
    
    #precalculate Xdag*X and Xdag
    c_ops_2 = []
    c_ops_dag = []
    for count in range(n_cops):
        c_ops_2.append((c_ops[count].T.conj()*c_ops[count]).todense())
        c_ops_dag.append((c_ops[count].T.conj()).todense())
        c_ops[count] = c_ops[count].todense()
        
    Hfull = H.todense()
    #Hfull[0,3]=0.8
    #print(Hfull)
    ############ DELETE THIS DIRECTLY AFTER DEBUGGING#################
    # import numpy as np
    #Hfull = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    # Hfull = np.array([[1,2],[3,4]])
    #print(Hfull)
    ########################################################
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, c_ops, c_ops_2, c_ops_dag, ldim_p*ldim_p*num_elements))
    #parallel version
    # if parallel:
    #     if num_threads == None:
    #         pool = Pool()
    #     else:
    #         pool = Pool(num_threads)
    #     #find all the rows of L
    #     L_lines = []
    #     if progress:
    #         print('Constructing Liouvillian L...')
    #         try:
    #             import tqdm
    #             for line in tqdm.tqdm(pool.imap(calculate_L_fixed, arglist), total=len(arglist)):
    #                 L_lines.append(line)
    #         except:
    #             print('Package tqdm required for progress bar in parallel version')
    #             pass
    #     if len(L_lines) == 0:
    #         L_lines = pool.imap(calculate_L_fixed, arglist)
    #     pool.close()
    #     #combine into a big matrix                    
    #     L = vstack(L_lines)
    #     return L
    
    if progress:
        from propagate import Progress
        total_points = sum([len(block) for block in mapping_block])
        bar = Progress(total_points, description='Constructing Liouvillian L...')

    #serial version
    L0 = [] # couples nu to nu 
    L1 = [] # couples nu to nu+1
    # Loop through all elements listed in mapping_block, grouped by excitation nu
    for nu in range(num_blocks):
        current_block = len(mapping_block[nu])
        line_block_nu = []          # L-line that couples to same excitation number
        line_block_nup = []         # L-line that couples to excitation number plus 1
        
        # For each nu, calculate part of the liouvillian that couples
        # to same nu, stored in L0, and part of the liouvillian that couples to 
        # nu+1, stored in L1. L0 and L1 are different for each nu.
        for count in range(current_block):
            idx = mapping_block[nu][count]  # this is the index of the current element in the conventional representation
            #print(idx)
            #print(f'Element: {arglist[idx][0]}')
            line = calculate_L_line(*arglist[idx]) # calculate the whole line of liouvillian for this element
            #line = csr_matrix([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
            # first index: row. Since calculate_L_fixed returns a matrix, the row index must be chosen as 0
            line_block_nu.append(csr_matrix(line[[0]*current_block,mapping_block[nu]]))  # get the elements that couple to the same nu
                       
            # calculate L1 part, that couples to nu+1 only if nu_max has not been reached
            if nu < num_blocks-1:
                next_block = len(mapping_block[nu+1])
                line_block_nup.append(csr_matrix(line[[0]*next_block, mapping_block[nu+1]]))
            if progress:
                bar.update()
        # append block matrix to L0
        L0.append(vstack(line_block_nu)) 

        if nu < num_blocks -1:
            L1.append(vstack(line_block_nup))

                   
    return L0,L1



def setup_L_block1(H, c_ops,num_threads, progress=False, parallel=False):
    
    """ Same as setup_L_block but optimized version of calculate_Line_block
    is used to only calculate L elements actually needed.

    """
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple, mapping_block
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    import sys
    
    num_elements = len(indices_elements)
    num_blocks = len(mapping_block)
    n_cops = len(c_ops)
    
    #precalculate Xdag*X and Xdag
    c_ops_2 = []
    c_ops_dag = []
    for count in range(n_cops):
        c_ops_2.append((c_ops[count].T.conj()*c_ops[count]).todense())
        c_ops_dag.append((c_ops[count].T.conj()).todense())
        c_ops[count] = c_ops[count].todense()
        
    Hfull = H.todense()
    #Hfull[0,3]=0.8
    #print(Hfull)
    ############ DELETE THIS DIRECTLY AFTER DEBUGGING#################
    # import numpy as np
    #Hfull = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    # Hfull = np.array([[1,2],[3,4]])
    #print(Hfull)
    ########################################################
    
    # can be optimized; we do not need all elements, only the ones in block 
    # structure
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, c_ops, c_ops_2, c_ops_dag, ldim_p*ldim_p*num_elements))
    #parallel version
    # if parallel:
    #     if num_threads == None:
    #         pool = Pool()
    #     else:
    #         pool = Pool(num_threads)
    #     #find all the rows of L
    #     L_lines = []
    #     if progress:
    #         print('Constructing Liouvillian L...')
    #         try:
    #             import tqdm
    #             for line in tqdm.tqdm(pool.imap(calculate_L_fixed, arglist), total=len(arglist)):
    #                 L_lines.append(line)
    #         except:
    #             print('Package tqdm required for progress bar in parallel version')
    #             pass
    #     if len(L_lines) == 0:
    #         L_lines = pool.imap(calculate_L_fixed, arglist)
    #     pool.close()
    #     #combine into a big matrix                    
    #     L = vstack(L_lines)
    #     return L
    
    if progress:
        from propagate import Progress
        total_points = sum([len(block) for block in mapping_block])
        bar = Progress(total_points, description='Constructing Liouvillian L...')

    #serial version
    L0 = [] # couples nu to nu 
    L1 = [] # couples nu to nu+1
    # Loop through all elements listed in mapping_block, grouped by excitation nu
    for nu in range(num_blocks):
        current_block = len(mapping_block[nu])
        line_block_nu = []          # L-line that couples to same excitation number
        line_block_nup = []         # L-line that couples to excitation number plus 1
        
        # For each nu, calculate part of the liouvillian that couples
        # to same nu, stored in L0, and part of the liouvillian that couples to 
        # nu+1, stored in L1. L0 and L1 are different for each nu.
        for count in range(current_block):
            idx = mapping_block[nu][count]  # this is the index of the current element in the conventional representation
            #print(idx)
            #print(f'Element: {arglist[idx][0]}')
            line = calculate_L_line_block(*arglist[idx]) # calculate the whole line of liouvillian for this element
            #line = csr_matrix([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
            # first index: row. Since calculate_L_fixed returns a matrix, the row index must be chosen as 0
        #     line_block_nu.append(csr_matrix(line[[0]*current_block,mapping_block[nu]]))  # get the elements that couple to the same nu
                       
        #     # calculate L1 part, that couples to nu+1 only if nu_max has not been reached
        #     if nu < num_blocks-1:
        #         next_block = len(mapping_block[nu+1])
        #         line_block_nup.append(csr_matrix(line[[0]*next_block, mapping_block[nu+1]]))
        #     if progress:
        #         bar.update()
        # # append block matrix to L0
        # L0.append(vstack(line_block_nu)) 

        # if nu < num_blocks -1:
        #     L1.append(vstack(line_block_nup))
    sys.exit()

                   
    return L0,L1


def calculate_L_line_block(element, H, c_ops, c_ops_2, c_ops_dag, length):
    """ Same as calculate_L_line, but only calculate block terms that are needed."""
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple, mapping_block, elements_block
    from numpy import zeros, concatenate, copy, mod
    from scipy.sparse import lil_matrix, csr_matrix
    
    tol = 1e-10
    n_cops = len(c_ops)
    num_blocks = len(mapping_block)
    num_elements = len(indices_elements)
    
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    
    # calculate nu from left (since left and right must have same excitation number, it does not matter if we choose left or right)
    nu_element = left[0] + num_blocks - 1 - sum(left[1:]) #left[0] is number of photons
    # setup L0 and L1 lines
    L0_line = zeros((1, len(mapping_block[nu_element])), dtype = complex)
    if nu_element < num_blocks-1:
        L1_line = zeros((1,len(mapping_block[nu_element+1])), dtype=complex)
    else:
        L1_line = []    
    
    # Calculate L0 elements -> in block nu_element
    for count in range(len(mapping_block[nu_element])):
        idx = mapping_block[nu_element][count] # current index
        
        # from this index, get spin element, n_left and n_right. Formula: idx = (n_left*ldim_p + n_right)*len(indices_elements) + element_idx
        element_idx = mod(idx,num_elements)
        idx1 = int((idx-element_idx)/num_elements)
        n_right = mod(idx1, ldim_p)
        n_left = int((idx1 - n_right)/ldim_p)
        
        element_left = indices_elements[element_idx][0:nspins]
        element_right = indices_elements[element_idx][nspins:2*nspins]
        
        # task: how to get coupling element between element (input of function) and element determined by count
        
        # x = indices_elements[element_idx]
        # el = concatenate(([n_left], x[0:nspins], [n_right],x[nspins:2*nspins]))
        # print(idx, el)
        
        # calculate commutator term
        # Hin = get_element(H, [left[0], left[count_ns+1]], [n_right, count_s])
        # n1_calc = False
        # n2_calc = False
        
        # if abs(Hin)>tol:
        #     #work out which elements of rho this couples to
        #     #note the resolution of identity here is small because H only acts between photon and one spin
        #     n1_element = copy(left)
        #     n1_element[0] = n_right
        #     n1_element[count_ns+1] = count_s
        #     n1_calc = True
            
        #     #get the indices of the equivalent element to the one which couples
        #     spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
        #     rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
        #     # print(f'Hin={Hin},spinnj={spinnj},rhonj={rhonj}')

            
        #     #increment L
        #     L_line[0, rhonj] = L_line[0, rhonj] -1j * Hin
            
        # #same for other part of commutator
        # Hnj = get_element(H, [count_phot, count_s], [right[0], right[count_ns+1]])
        # # print(f'Hnj: {Hnj}')    
        # if abs(Hnj)>tol:
        #     n2_element = copy(right)
        #     n2_element[0] = count_phot
        #     n2_element[count_ns+1] = count_s
        #     n2_calc = True
            
        #     spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
        #     rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
        #     # print(f'Hnj={Hnj},spinin={spinin},rhoin={rhoin}')
            
        #     L_line[0, rhoin] = L_line[0, rhoin] + 1j * Hnj
    
    
    # for count_phot in range(ldim_p):
    #     # print(f'count phot: {count_phot}')
    #     for count_s in range(ldim_s):
    #         #print(f'count s: {count_s}')
    #         for count_ns in range(nspins):
    #             #print(f'count ns: {count_ns}')
    #             #print(f'\ncount_phot={count_phot}, count_s={count_s}, count_ns={count_ns}')
    #             #keep track of if we have done the n1/n2 calculations
    #             n1_calc = False
    #             n2_calc = False
                    
    #             #calculate appropriate matrix elements of H
    #             Hin = get_element(H, [left[0], left[count_ns+1]], [count_phot, count_s])
    #             # print(f'Hin: {Hin}')
    #             #print(f'ELEMENT left: {ldim_s*left[0] + left[count_ns+1]}, {ldim_s*count_phot + count_s}')
    #             #only bother if H is non-zero
    #             if abs(Hin)>tol:
                    
    #                 #print('\n')
    #                 #work out which elements of rho this couples to
    #                 #note the resolution of identity here is small because H only acts between photon and one spin
    #                 n1_element = copy(left)
    #                 n1_element[0] = count_phot
    #                 n1_element[count_ns+1] = count_s
    #                 n1_calc = True
                    
    #                 #get the indices of the equivalent element to the one which couples
    #                 spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
    #                 rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
    #                 # print(f'Hin={Hin},spinnj={spinnj},rhonj={rhonj}')

                    
    #                 #increment L
    #                 L_line[0, rhonj] = L_line[0, rhonj] -1j * Hin
                    
    #             #same for other part of commutator
    #             Hnj = get_element(H, [count_phot, count_s], [right[0], right[count_ns+1]])
    #             # print(f'Hnj: {Hnj}')    
    #             if abs(Hnj)>tol:
    #                 n2_element = copy(right)
    #                 n2_element[0] = count_phot
    #                 n2_element[count_ns+1] = count_s
    #                 n2_calc = True
                    
    #                 spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
    #                 rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
    #                 # print(f'Hnj={Hnj},spinin={spinin},rhoin={rhoin}')
                    
    #                 L_line[0, rhoin] = L_line[0, rhoin] + 1j * Hnj
                    
    #             for count_cop in range(n_cops):
                        
    #                 #Do the same as above for each collapse operator
    #                 Xin = get_element(c_ops_2[count_cop], [left[0], left[count_ns+1]], [count_phot, count_s])
    #                 if abs(Xin)>tol:
    #                     if not(n1_calc):
    #                         n1_element = copy(left)
    #                         n1_element[0] = count_phot
    #                         n1_element[count_ns+1] = count_s
    #                         n1_calc = True
                                
    #                         spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
    #                         rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                            
    #                     L_line[0, rhonj] = L_line[0, rhonj] - 0.5*Xin
                        
    #                 Xnj = get_element(c_ops_2[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
    #                 if abs(Xnj)>tol:
    #                     if not(n2_calc):
    #                         n2_element = copy(right)
    #                         n2_element[0] = count_phot
    #                         n2_element[count_ns+1] = count_s
    #                         n2_calc = True
                    
    #                         spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
    #                         rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
    #                     L_line[0, rhoin] = L_line[0, rhoin] - 0.5*Xnj
                        
    #                 Xdagnj = get_element(c_ops_dag[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
    #                 #only need to calculate if Xdag is non-zero
    #                 if abs(Xdagnj)>tol:
    #                     for count_phot2 in range(ldim_p):
    #                         for count_s2 in range(ldim_s):
    #                             #The term XpXdag requires two resolutions of unity
    #                             Xim = get_element(c_ops[count_cop], [left[0], left[count_ns+1]], [count_phot2, count_s2])
    #                             if abs(Xim)>tol:
    #                                 m1_element = copy(left)
    #                                 m1_element[0] = count_phot2
    #                                 m1_element[count_ns+1] = count_s2
                                        
    #                                 if not(n2_calc):
    #                                     n2_element = copy(right)
    #                                     n2_element[0] = count_phot
    #                                     n2_element[count_ns+1] = count_s
    #                                     n2_calc = True
                                            
    #                                 spinmn = indices_elements_inv[get_equivalent_dm_tuple(concatenate((m1_element[1:], n2_element[1:])))]
    #                                 rhomn = (length//ldim_p)*m1_element[0] + length//(ldim_p*ldim_p)*n2_element[0] + spinmn
    #                                 L_line[0, rhomn] = L_line[0, rhomn] + Xim*Xdagnj 

    # L_line = csr_matrix(L_line)
    #return csr_matrix([1]),csr_matrix([1,2,3,4,5])
        
        
 
    
    
def setup_op(H, num_threads):
    
    """Generate generic Liouvillian for Hamiltonian H and 
    collapse operators c_ops. Use num_threads processors for 
    parallelisation.
    Note c_ops must be a list (even with only one element)"""
    
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    
    num_elements = len(indices_elements)
        
    Hfull = H.todense()
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, ldim_p*ldim_p*num_elements))
        
    
    #allocate a pool of threads
    if num_threads == None:
        pool = Pool()
    else:
        pool = Pool(num_threads)
    #find all the rows of L
    L_lines = pool.map(calculate_op_fixed, arglist)
    
    pool.close()
    
    #uncomment for serial version
    #L_lines = []
    #for count in range(ldim_p*ldim_p*len(indices_elements)):
    #    L_lines.append(calculate_L_fixed(arglist[count]))
    
    #combine into a big matrix                    
    L = vstack(L_lines)
    
    return L
    
def calculate_op_fixed(args):
    return calculate_op_line(*args)
    
def calculate_op_line(element, H, length):
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import zeros, concatenate, copy
    from scipy.sparse import lil_matrix, csr_matrix
    
       
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    tol = 1e-10
    
    L_line = zeros((1, length), dtype = complex)

        
    for count_phot in range(ldim_p):
        for count_s in range(ldim_s):
            for count_ns in range(nspins):
                    
                #keep track of if we have done the n1/n2 calculations
                n1_calc = False
                n2_calc = False
                    
                #calculate appropriate matrix elements of H
                Hin = get_element(H, [left[0], left[count_ns+1]], [count_phot, count_s])
                    
                #only bother if H is non-zero
                if abs(Hin)>tol:
                    #work out which elements of rho this couples to
                    #note the resolution of identity here is small because H only acts between photon and one spin
                    n1_element = copy(left)
                    n1_element[0] = count_phot
                    n1_element[count_ns+1] = count_s
                    n1_calc = True
                    
                    #get the indices of the equivalent element to the one which couples
                    spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                    rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                    
                    #increment L
                    L_line[0, rhonj] = L_line[0, rhonj] + Hin
                    
                    
                
    L_line = csr_matrix(L_line)
    return L_line


def setup_rho(rho_p, rho_s):
    
    """Calculate the compressed representation of the state 
    with photon in state rho_p and all spins in state rho_s"""
    
    from indices import indices_elements
    from numpy import zeros
    
        
    num_elements = len(indices_elements)
    
    rho_vec = zeros(ldim_p*ldim_p*num_elements, dtype = complex)
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                element = indices_elements[count]
                element_index = ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                left = element[0:nspins]
                right = element[nspins:2*nspins]
        
                rho_vec[element_index] = rho_p[count_p1, count_p2]
                for count_ns in range(nspins):
                    rho_vec[element_index] *= rho_s[left[count_ns], right[count_ns]] 
    return rho_vec


def setup_rho_block(rho_p, rho_s):
    """ Same as 'setup_rho', but in block format, i.e. elements are grouped 
    according to their excitation number. To do this, use the mapping_block list
    """
    
    from indices import indices_elements, mapping_block
    from numpy import zeros
    
    num_elements = len(indices_elements)
    blocks = len(mapping_block)
    
    rho_vec = zeros(ldim_p*ldim_p*num_elements, dtype = complex)    
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                element = indices_elements[count]
                element_index = ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                left = element[0:nspins]
                right = element[nspins:2*nspins]
                rho_vec[element_index] = rho_p[count_p1, count_p2]
                for count_ns in range(nspins):
                    rho_vec[element_index] *= rho_s[left[count_ns], right[count_ns]]
                    
    # Now use the mapping list to get the desired block structure from the whole rho_vec:
    rho_vec_block = []
    for count in range(blocks):
        rho_vec_block.append(rho_vec[mapping_block[count]])
    #print(rho_vec_block)
    
    return rho_vec_block
    
    
    
    """ come back to this idea later maybe, it might be more efficient. For now,
    Let us copy the setup_rho and just take the elements we need."""
    # from indices import indices_elements
    
    # num_elements = len(indices_elements)
    
    # rho_vec = []
    # nu_max = nspins # maximum excitation number IF initial state is all spins up and zero photons
    
    # for nu in range(nu_max, -1, -1): # from nu_max to nu=0
    #     print(nu)
    #     for count in range(num_elements):
    #         element = indices_elements[count]
    #         left = element[:nspins]
    #         right = element[nspins:]
            
    #         # spin excitation number equals number of ones in left/right
    #         m_left = sum(left)
    #         m_right = sum(right)
    #         if m_left > nu or m_right > nu: # do not consider if spin excitation is larger than total excitation
    #             continue 
            
    #         # photon excitation: nu-m
    #         n_left = nu - m_left
    #         n_right = nu - m_right
            
    #         print(f'Element: left: ({n_left},{m_left}), right: ({n_right},{m_right})')    
    #         element_index = ldim_p*num_elements*n_left + num_elements*n_right + count
    #         print(f'Index: {element_index}\n')

            


                 
def get_element(H, left, right):
    global ldim_s
    return H[ldim_s*left[0] + left[1], ldim_s*right[0] + right[1]]
    
    


