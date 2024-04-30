# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:00:09 2024

@author: lukas
"""
from numpy import array, where
from math import factorial

def degeneracy_outer_invariant(outer1, outer2, inner):
    xi = outer1+ 2*outer2
    # print(xi)
    # print('inner',inner)
    deg = 1
    for i in range(4):
        l = where(xi==i)[0]
        # print('loop',i)
        # print(l)
        if len(l) == 0:
            continue
        
        sub_inner = inner[l]
        # print('subinner',sub_inner)
        s = sum(sub_inner)
        
        factor =  factorial(len(l)) / (factorial(s)*factorial(len(l)-s))
        # print('factor',factor)
        
        deg = deg * factor
        
    print(int(deg))
    
    
def degeneracy_outer_invariant1(outer1, outer2, inner):
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
        
        # print(outer1_cp, inner_cp, outer2_cp)
        
        if (all(outer1_cp == outer1) and all(outer2_cp == outer2)):
            perms.append(inner_cp)

    return len(perms)

def degeneracy_outer_invariant_inner2(outer1,outer2,inner1,inner2):
    """ For sigma^- lindblad operator. Calculate the number of permutations,
    that leave outer1 and outer2 invariant, but produce a new combined state
    (inner1, inner2)"""
    
    from itertools import permutations
    from numpy import array, concatenate
    perms = [concatenate((inner1, inner2))]
    for p in permutations(range(len(inner1))):
        inner1_cp = array([inner1[i] for i in p])
        inner2_cp = array([inner2[i] for i in p])
        inner_total = concatenate((inner1_cp, inner2_cp))
        if(any(all(existing_list == inner_total) for existing_list in perms)):
            continue
        
        outer1_cp = array([outer1[i] for i in p])
        outer2_cp = array([outer2[i] for i in p])
        
        if (all(outer1_cp == outer1) and all(outer2_cp == outer2)):
            perms.append(concatenate((inner1_cp, inner2_cp)))

    return len(perms)
    
    
    
    
def degeneracy_permutation(left, right):
    """ Calculate the degeneracy due to simultaneous permutation of left and 
    right spin indices"""
    from math import factorial
    from numpy import where
    xi = left + 2*right
    deg = factorial(len(xi))
    for i in range(4):
        l = len(where(xi==i)[0])
        deg = deg / factorial(l)
        
    return deg




def align_ones(left, right):
    from numpy import where, setdiff1d, intersect1d, copy
    
    # check indices, where the arrays have ones
    idx1 = where(left == 1)[0]
    idx2 = where(right == 1)[0]

    # find common indices and remove them, because they are already in order
    common_elements = intersect1d(idx1,idx2)
    idx_ones1 = setdiff1d(idx1, common_elements)
    idx_ones2 = setdiff1d(idx2, common_elements)
    
    # go through idx_ones1 and permute elements, such that ones align
    left_cp = copy(left)
    for i in range(len(idx_ones1)):
        left_cp[idx_ones2[i]] = left[idx_ones1[i]]
        left_cp[idx_ones1[i]] = left[idx_ones2[i]]
        

    return 0

# Example usage
left = array([0, 1, 0, 0, 1])
right = array([1, 1, 0, 1, 0])

x = align_ones(left,right)





# outer1 = array([1,1,1])
# outer2 = array([1,1,1])
# inner1 = array([1,1,0])
# inner2 = array([1,1,0])

# print(degeneracy_outer_invariant_inner2(outer1, outer2, inner1, inner2))













