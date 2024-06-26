indices_elements = []
indices_elements_inv = {}
mapping_block = []
elements_block = []

# Sets up two dictionaries:
""" 1) indices_elements maps from a reduced density matrix index to a vector 
with the full set indices for that element
 e.g for 3 spins this could be [s00 s10 s20 s01 s11 s21]
 2) indices_elements_inv maps from a TUPLE containing a list of 
 dm indices to an element in the compressed space

THESE DICTIONARIES NOW DO NOT INCLUDE THE PHOTON STATE
The photon state can be calculated using the fact that the full dm is just the tensor 
product of the photon with the compressed list of spin indices"""

def list_equivalent_elements():
    """Generate basis list, needs to be run at the beginning of
    each calculation"""
    
    global indices_elements, indices_elements_inv
    
    from basis import nspins, ldim_p, ldim_s
    from numpy import concatenate, copy, array

    indices_elements = []
    indices_elements_inv = {}
    
    count = 0
    
    #get minimal list of left and right spin indices (in combined form)
    spins = setup_spin_indices(nspins)
    
    left =[]
    right = []
    
    #split combined indices into left/right form
    for count in range(len(spins)):
        leftadd, rightadd = _to_hilbert(spins[count])
        left.append(leftadd)
        right.append(rightadd)

    
    left = array(left)
    right = array(right)

    #loop over each photon state and each spin configuration
    for count in range(len(spins)):
                
                #calculate element and index 
                element = concatenate((left[count], right[count]))
                
                #add appropriate entries to dictionaries
                indices_elements.append(copy(element))
                indices_elements_inv[_comp_tuple(element)] = count
   # print(spins)
    #print(indices_elements)
    


def setup_spin_indices(ns):
    """get minimal list of left and right spin indices"""
    
    from basis import ldim_s
    from numpy import concatenate, array, copy
    
    spin_indices = []
    spin_indices_temp = []
    
    #construct all combinations for one spin
    for count in range(ldim_s**2):
        spin_indices_temp.append([count])
    spin_indices_temp = array(spin_indices_temp)
    spin_indices = [array(x) for x in spin_indices_temp] # Used if ns == 1
    
    #loop over all other spins
    for count in range(ns-1):
        #make sure spin indices is empty 
        spin_indices = []   
        #loop over all states with count-1 spins
        for index_count in range(len(spin_indices_temp)):
         
            #add all numbers equal to or less than the last value in the current list
            for to_add in range(spin_indices_temp[index_count, -1]+1):
                spin_indices.append(concatenate((spin_indices_temp[index_count, :], [to_add])))
        spin_indices_temp = copy(spin_indices)
    
    return spin_indices

def mapping_task(args_tuple):
    from basis import nspins, ldim_p, ldim_s
    from numpy import concatenate
    nu_max = nspins
    count_p1, count_p2, count = args_tuple
    num_elements = len(indices_elements)
    element = indices_elements[count]
    element_index = ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
    left = element[0:nspins]
    right = element[nspins:2*nspins]
    m_left = nspins-sum(left)
    m_right = nspins-sum(right)
    nu_left = m_left + count_p1
    nu_right = m_right + count_p2
    if nu_left == nu_right and nu_left <= nu_max:                   
        el = concatenate(([count_p1], left, [count_p2],right))
        return (nu_left, element_index, el)

def setup_mapping_block(parallel=False):
    """
    Generate mapping between reduced representation of density matrix and
    the block structure, which is grouped in different numbers of total excitations
    of photons + spins. Note: 0 in spin array means spin up!
    For now, nu_max (maximum excitation) is set to nspins, because the initial 
    condition is always all spins up and zero photons in the cavity.
    
    Structure of mapping_block = [ [indices of nu=0] , [indices of nu=1], ... [indices of nu_max] ]

    """
    from basis import nspins, ldim_p, ldim_s
    global mapping_block, indices_elements, elements_block
    from numpy import concatenate
    import numpy as np
    from time import time
    
    num_elements = len(indices_elements)
    
    nu_max = nspins # maximum excitation number IF initial state is all spins up and zero photons
    mapping_block = [ [] for _ in range(nu_max+1)] # list of nu_max+1 empty lists
    mapping_block2 = [ [] for _ in range(nu_max+1)] # list of nu_max+1 empty lists
    elements_block = [ [] for _ in range(nu_max+1)]
    
    if parallel:
        from multiprocessing import Pool
        from itertools import product
    
        arglist = []
        t0 = time()
        for count_p1, count_p2, count in product(range(ldim_p), range(ldim_p), range(num_elements)):
            arglist.append((count_p1, count_p2, count))
    
        with Pool() as p:
            results = p.map(mapping_task, arglist)
    
        #for nu, element_index in results: # do we know how long block will be at each nu? 
        for result in results:
            if result is None:
                continue
            # try to avoid this?
            mapping_block[result[0]].append(result[1])
            elements_block[result[0]].append(result[2])
        
   # print('Parallel mapping block in {:.1f}s'.format(time()-t0)) 
   # t0 = time()
    else:
        for count_p1 in range(ldim_p):
            for count_p2 in range(ldim_p):
                for count in range(num_elements):
                    element = indices_elements[count]
                    element_index = ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                    left = element[0:nspins]
                    right = element[nspins:2*nspins]
                    
                    # calculate excitations. Important: ZEOR MEANS SPIN UP, ONE MEANS SPIN DOWN.
                    m_left = nspins-sum(left)
                    m_right = nspins-sum(right)
                    # calculate nu
                    nu_left = m_left + count_p1
                    nu_right = m_right + count_p2
                    if nu_left == nu_right and nu_left <= nu_max:                   
                        el = concatenate(([count_p1], left, [count_p2],right))
                        mapping_block[nu_left].append(element_index)
                        elements_block[nu_left].append(el)
        #print('Serial mapping block in {:.1f}s'.format(time()-t0)) 
        # for bi in range(len(mapping_block)):
        #     assert np.allclose(mapping_block[bi], mapping_block2[bi])
    # print(mapping_block)
    # print(elements_block)
    # for i in mapping_block:
        # print(len(i))
    

def _index_to_element(index, ns= None):
    """convert a combined spin index to an element with ns spins
    NOT for converting from a full dm index to an element"""
    
    from basis import nspins, ldim_s
    
    if ns == None:
        ns = nspins
    element = []
    
    #do appropriate modulo arithmatic 
    for count in range(ns):
        element.append(index%ldim_s)
        index = (index - element[-1])//ldim_s
    return element
            

def get_equivalent_dm_tuple(dm_element):
    """calculate tuple representation of dm element which is equivalent to dm_element"""
    
    from basis import nspins,ldim_s, ldim_p
    from numpy import sort, concatenate,array
    
    if len(dm_element) != 2*(nspins):
        raise TypeError('dm_index has the wrong number of elements')
    
    left = array(dm_element[0:nspins])
    right = array(dm_element[nspins:2*nspins])
    
    #use combined Hilbert space indices for both left and right vectors
    combined = _full_to_combined(left, right)
    #The equivalent element is one in which this list is sorted
    combined = -sort(-1*combined)


    newleft, newright = _combined_to_full(combined)
    dm_element_new = concatenate((newleft, newright))
  
    #return the index of the equvalent element
    return _comp_tuple(dm_element_new)


def _to_hilbert(combined):
    """convert to Hilbert space index"""
    
    left = []
    right = []
    for count in range(len(combined)):
        leftadd, rightadd = _combined_to_full(combined[count])
        left.append(leftadd)
        right.append(rightadd)
    return left, right
    

def _combined_to_full(combined):
    """create left and right Hilbert space indices from combined index"""
    
    from basis import ldim_s    
    right = combined%ldim_s
    left = (combined - right)//ldim_s
    
    return left, right


def _full_to_combined(left, right):
    """create left and right Hilbert space indices from combined index"""
    from basis import ldim_s
    return ldim_s*left + right


def _comp_tuple(element):
    """compress the tuple used in the dictionary"""
    from basis import nspins, ldim_s, ldim_p
    
    element_comp = []
    
    for count in range(nspins):
        element_comp.append(element[count]*ldim_s + element[count+nspins])
    return tuple(element_comp)

