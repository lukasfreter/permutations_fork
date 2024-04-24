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
                    if rhonj == 17:
                        print('rhonj')
                        print(f'left: {left[0]},{left[count_ns+1]}, right: {count_phot}, {count_s}')
                        print(Hin)
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
                    if rhoin == 17:
                        print('rhoin')
                        print(count_phot, count_s, count_ns)
                        print(Hnj)
                    
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
            if count == 4:
                print('kek')
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
            if count == 2:
                print('kek')
            line0, line1 = calculate_L_line_block1(*arglist[idx]) # calculate the whole line of liouvillian for this element
            line_block_nu.append(line0)  # get the elements that couple to the same nu
                       
        #     # calculate L1 part, that couples to nu+1 only if nu_max has not been reached
        #     if nu < num_blocks-1:
        #         next_block = len(mapping_block[nu+1])
        #         line_block_nup.append(csr_matrix(line[[0]*next_block, mapping_block[nu+1]]))
        #     if progress:
        #         bar.update()
        # # append block matrix to L0
        L0.append(vstack(line_block_nu)) 

        # if nu < num_blocks -1:
        #     L1.append(vstack(line_block_nup))
   # sys.exit()
    return L0,L1

def calculate_L_line_block1(element, H, c_ops, c_ops_2, c_ops_dag, length):
    """ Same as calculate_L_line, but only calculate block terms that are needed."""
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple, mapping_block, elements_block
    from numpy import zeros, concatenate, copy, mod
    from scipy.sparse import lil_matrix, csr_matrix
    
    tol = 1e-10
    n_cops = len(c_ops)
    num_blocks = len(mapping_block)
    num_elements = len(indices_elements)
    
    # These are the density matrix indices for the element, of which we want
    # to calculate the time derivative.
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
    # loop through all elements in the current block with same excitation, to build L0
    for count in range(len(mapping_block[nu_element])):
        idx = mapping_block[nu_element][count] # current index
        
        # from this index, get spin element, n_left and n_right. Formula: idx = (n_left*ldim_p + n_right)*len(indices_elements) + element_idx
        element_idx = mod(idx,num_elements)
        idx1 = int((idx-element_idx)/num_elements)
        n_right = mod(idx1, ldim_p)
        n_left = int((idx1 - n_right)/ldim_p)
        
        # elements which differ in photon number by 2 will never couple:
        if abs(n_left - left[0]) > 1 or abs(n_right - right[0]) > 1:
            continue
        
        # these are the density matrix indices of the element, which possibly contributes
        # to the time derivative of the element labeled with "left" and "right" above.
        element_left = indices_elements[element_idx][0:nspins]
        element_right = indices_elements[element_idx][nspins:2*nspins]
        
        left_to_couple = concatenate(([n_left], element_left))
        right_to_couple = concatenate(([n_right], element_right))
        
        # Now that the coupled to element is determined, calculate the commutator part of L
        # d/dt rho_nmn'm' = -i (H_nmij rho_ijn'm' - rho_nmij H_ijn'm')
        for count_ns in range(nspins): # go through the spins one by one
            # First part of commutator: check if right_to_couple is compatible with right
            if(states_compatible(right, right_to_couple)):
                # calculate Hnmij, labelled as Hin
                Hin = get_element(H,[left[0], left[1+count_ns]],[left_to_couple[0], left_to_couple[1+count_ns]])
                deg = degeneracy_outer_invariant(left[1:], right[1:], left_to_couple[1:])
                L0_line[0,count] = L0_line[0,count] - 1j*Hin*deg
            
            # second part of commutator
            if(states_compatible(left, left_to_couple)):
                # calculate H_ijn'm' labelled as Hnj
                Hnj = get_element(H, [right_to_couple[0], right_to_couple[1+count_ns]],[right[0], right[1+count_ns]])
                deg = degeneracy_outer_invariant(left[1:], right[1:], right_to_couple[1:])
                L0_line[0,count] = L0_line[0,count] + 1j*Hnj*deg
                

        
        # for count_ns in range(nspins): # go through the spins one by one
        #     # first check if the left/right elements match and calculate proper commutator parts
        #     if (left_to_couple == left).all() and (right_to_couple == right).all():
        #         # this case is if an element couples to itself. Then both parts of the commutator matter.
        #         Hin = get_element(H,[left[0], left[1+count_ns]],[left[0], left[1+count_ns]])
        #         Hnj = get_element(H, [right[0], right[1+count_ns]],[right[0], right[1+count_ns]])
        #         L0_line[0,count] = L0_line[0,count] -1j * Hin + 1j*Hnj
                
        #     elif (left_to_couple == left).all():
        #         # Left side matches: i.e. d/dt rho_ij = -i(Hin*rhonj - rhoin*Hnj) = -i(-rhoin*Hnj) because index i matches
        #         # get all elements that contribute under permutation of the right index, that leave the element invariant.
        #         Hnj = get_element(H, [right_to_couple[0], right_to_couple[1+count_ns]],[right[0], right[1+count_ns]])
        #         L0_line[0, count] = L0_line[0, count] + 1j * Hnj
                    
        #     elif (right_to_couple == right).all():
        #         Hin = get_element(H, [left[0], left[1+count_ns]],[left_to_couple[0],left_to_couple[1+count_ns]])
        #         L0_line[0, count] = L0_line[0, count] -1j * Hin
        #     else:
        #         # if there is no matching left or right parts, then try to permute them to equivalent elements
        #         right_to_couple_perm = get_permutation_equivalent(left, left_to_couple, right_to_couple)
        #         if len(right_to_couple_perm) != 0:
        #             Hnj = get_element(H, [right_to_couple_perm[0], right_to_couple_perm[1+count_ns]],[right[0], right[1+count_ns]])
        #             L0_line[0, count] = L0_line[0, count] + 1j * Hnj
        #         else:
        #             left_to_couple_perm = get_permutation_equivalent(right, right_to_couple, left_to_couple)
        #             if len(left_to_couple_perm) != 0:
        #                 Hin = get_element(H, [left[0], left[1+count_ns]],[left_to_couple_perm[0],left_to_couple_perm[1+count_ns]])
        #                 L0_line[0, count] = L0_line[0, count] -1j * Hin
                
            
    L0_line = csr_matrix(L0_line)
    return L0_line, L1_line

def states_compatible(state1, state2):
    """ checks, if state1 and state2 are equivalent up to permutation of spins"""
    if state1[0] != state2[0]:
        return False
    
    from numpy import where, setdiff1d, intersect1d
    
    spins1 = state1[1:]
    spins2 = state2[1:]
    if(sum(state1) != sum(state2)):
        return False
    
    return True
    # # check indices, where the arrays have ones
    # idx1 = where(spins1 == 1)[0]
    # idx2 = where(spins2 == 1)[0]
    
    # # find common indices and remove them, because they are already in order
    # common_elements = intersect1d(idx1,idx2)
    # idx_ones1 = setdiff1d(idx1, common_elements)
    # idx_ones2 = setdiff1d(idx2, common_elements)
            
    # return [idx_ones1, idx_ones2]
    
def degeneracy_outer_invariant(outer1, outer2, inner):
    """ calculate how many distinct permutations there are of the spins (outer1, inner)
    and (inner, outer2), which leave outer1 and outer2 invariant. """
    
    from itertools import permutations
    from numpy import array
    perms = [inner]
    for p in permutations(range(len(inner))):
        inner_cp = array([inner[i] for i in p])
        if(any(all(existing_list == inner_cp) for existing_list in perms)):
            continue
        
        outer1_cp = array([outer1[i] for i in p])
        outer2_cp = array([outer2[i] for i in p])
        
        print(outer1_cp, inner_cp, outer2_cp)
        
        if (all(outer1_cp == outer1) and all(outer2_cp == outer2)):
            perms.append(inner_cp)

    return len(perms)
    
    


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
    
    # These are the density matrix indices for the element, of which we want
    # to calculate the time derivative.
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

    
    # loop through all elements in the corrent block with same excitation, to build L0
    for count in range(len(mapping_block[nu_element])):
        idx = mapping_block[nu_element][count] # current index
        
        # from this index, get spin element, n_left and n_right. Formula: idx = (n_left*ldim_p + n_right)*len(indices_elements) + element_idx
        element_idx = mod(idx,num_elements)
        idx1 = int((idx-element_idx)/num_elements)
        n_right = mod(idx1, ldim_p)
        n_left = int((idx1 - n_right)/ldim_p)
        
        # elements which differ in photon number by 2 will never couple:
        if abs(n_left - left[0]) > 1 or abs(n_right - right[0]) > 1:
            continue
        
        # these are the density matrix indices of the element, which contributes
        # to the time derivative of the element labeled with "left" and "right" above.
        element_left = indices_elements[element_idx][0:nspins]
        element_right = indices_elements[element_idx][nspins:2*nspins]
        
        left_to_couple = concatenate(([n_left], element_left))
        right_to_couple = concatenate(([n_right], element_right))
        
        if count == 3:
            print(1)

        
        # Now that the coupled to element is determined, calculate the commutator part of L
        
        for count_ns in range(nspins): # go through the spins one by one
            # first check if the left/right elements match and calculate proper commutator parts
            if (left_to_couple == left).all() and (right_to_couple == right).all():
                # this case is if an element couples to itself. Then both parts of the commutator matter.
                Hin = get_element(H,[left[0], left[1+count_ns]],[left[0], left[1+count_ns]])
                Hnj = get_element(H, [right[0], right[1+count_ns]],[right[0], right[1+count_ns]])
                
                # degin = degeneracy(copy(right[1:]), copy(left[1:]))
                # degnj = degeneracy(copy(left[1:]), copy(right[1:])) 
                
                L0_line[0,count] = L0_line[0,count] -1j * Hin + 1j*Hnj
                
            elif (left_to_couple == left).all():
                # Left side matches: i.e. d/dt rho_ij = -i(Hin*rhonj - rhoin*Hnj) = -i(-rhoin*Hnj) because index i matches
                # get all elements that contribute under permutation of the right index, that leave the element invariant.
                # degnj = degeneracy(copy(left_to_couple[1:]), copy(right_to_couple[1:]))
                
                Hnj = get_element(H, [right_to_couple[0], right_to_couple[1+count_ns]],[right[0], right[1+count_ns]])
                L0_line[0, count] = L0_line[0, count] + 1j * Hnj
                    
            elif (right_to_couple == right).all():
                # degin = degeneracy(copy(right_to_couple[1:]), copy(left_to_couple[1:]))
                Hin = get_element(H, [left[0], left[1+count_ns]],[left_to_couple[0],left_to_couple[1+count_ns]])
                L0_line[0, count] = L0_line[0, count] -1j * Hin
            else:
                # if there is no matching left or right parts, then try to permute them to equivalent elements
                right_to_couple_perm = get_permutation_equivalent(left, left_to_couple, right_to_couple)
                if len(right_to_couple_perm) != 0:
                    # degnj = degeneracy(copy(left_to_couple[1:]), copy(right_to_couple[1:]))
                    Hnj = get_element(H, [right_to_couple_perm[0], right_to_couple_perm[1+count_ns]],[right[0], right[1+count_ns]])
                    L0_line[0, count] = L0_line[0, count] + 1j * Hnj
                else:
                    left_to_couple_perm = get_permutation_equivalent(right, right_to_couple, left_to_couple)
                    if len(left_to_couple_perm) != 0:
                        # degin = degeneracy(copy(right_to_couple[1:]), copy(left_to_couple[1:]))
                        Hin = get_element(H, [left[0], left[1+count_ns]],[left_to_couple_perm[0],left_to_couple_perm[1+count_ns]])
                        L0_line[0, count] = L0_line[0, count] -1j * Hin
                
            
    L0_line = csr_matrix(L0_line)
    return L0_line, L1_line
        
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

# def degeneracy(arr1, arr2):
#     """ Find number of simultaneous permutations of arr1 and arr2, such that arr2
#     stays invariant."""
#     from itertools import permutations
#     from numpy import concatenate, array
#     perms = [arr2]
#     for p in permutations(range(len(arr1))):
#         arr1_cp = array([arr1[i] for i in p])
#         arr2_cp = array([arr2[i] for i in p])
#         if(any(all(existing_list == arr2_cp) for existing_list in perms)):
#             continue
#         perms.append(arr2_cp)
#     print(arr1,arr2)
#     print(perms)
#     # return(len(perms))
#     return 1
    
# def degeneracy1(arr1, arr2, left,right):
#     """ Find number of simultaneous permutations of arr1 and arr2, such that arr2
#     stays invariant AND left, right stay invariant"""
#     from itertools import permutations
#     from numpy import concatenate, array
#     perms = [arr2]
#     for p in permutations(range(len(arr1))):
#         arr1_cp = array([arr1[i] for i in p])
#         arr2_cp = array([arr2[i] for i in p])
#         if(any(all(existing_list == arr2_cp) for existing_list in perms)):
#             continue
#         perms.append(arr2_cp)
#     print(arr1,arr2)
#     print(perms)
#     return(len(perms))
    
    
        
def get_permutation_equivalent(_basis, _permute, _output):
    """_basis, _permute, _output are of the form [photon number, spin state] in 
    the compressed form. This function swaps the spin indices of _permute, such
    that they align exactly with the spin indices of _basis. The _output's spin
    indices are swapped in the same fashion, such that the total matrix element of
    _permute and _output stays invariant."""
    from numpy import array, where, copy, intersect1d, setdiff1d,concatenate
    # try to find permutation such that left1 = left2
    basis = _basis[1:]
    permute = _permute[1:]
    output = _output[1:]
    if(sum(basis) != sum(permute)):
        return []
    else:
        # check indices, where the arrays have ones
        idx1 = where(basis == 1)[0]
        idx2 = where(permute == 1)[0]
        
        # find common indices and remove them, because they are already in order
        common_elements = intersect1d(idx1,idx2)
        idx_ones1 = setdiff1d(idx1, common_elements)
        idx_ones2 = setdiff1d(idx2, common_elements)
        # now we know that we need to put the elements at index idx_ones2 from left2
        # at the index idx_ones1. Then, both are the same.
        cp_permute = copy(permute)
        cp_output = copy(output)
        for i in range(len(idx_ones1)):
            cp_permute[idx_ones2[i]] = permute[idx_ones1[i]]
            cp_permute[idx_ones1[i]] = permute[idx_ones2[i]]
            
            cp_output[idx_ones2[i]] = output[idx_ones1[i]]
            cp_output[idx_ones1[i]] = output[idx_ones2[i]]
            
    return concatenate(([_output[0]],cp_output))

def get_permutation_list(_invariant, _permute):
    """_invariant, _permute are of the form [photon number, spin state] in 
    the compressed form. This function lists all the possible permutations of 
    the _permute spin indices, that necessarily leave the _invariant spin indices
    invariant"""
    invariant = _invariant[1:]
    permute = _permute[1:]
    
    from sympy.utilities.iterables import multiset_permutations
    
    
    
    
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

def get_element_block(H,left,right):
    global ldim_s
    return H[ldim_s*left[0] + 1-left[1], ldim_s*right[0] + 1 - right[1]]
    
    


