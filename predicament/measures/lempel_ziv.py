import string  
from strenum import StrEnum
import numpy as np
import math

BoxLabel = StrEnum(
    'BoxLabel',
    list(string.ascii_uppercase)[:18])
BoxLabels = set([l for l in BoxLabel])

def contains(small, big):
    try:
        big.tobytes().index(small.tobytes())//big.itemsize
        return True
    except ValueError:
        return False

def lempel_ziv_casali_flow(data):
    """
    Checks the lempel Ziv complexity of a multi-channel time-series, by
    checking the channels in turn. Novel subsequences are only counted
    if they cannot be found in a channel so far, or in any previous channel. 
    
    
    parameters
    ----------
    data : a L1 x L2 2d array of data, with dtype=uint8, and where L1 is the
        time dimension (I think) while L2 is the number of channels
    """
    raise NotImplementedError(
        "There is an undiagonosed error in this implementation")
    L1, L2 = data.shape
    stop = False
    box = BoxLabel.A
    while not stop:
        if box == BoxLabel.A:
            # Lempel Ziv complexity
            complexity = 1
            # check for this
            # small array column - 
            r = 1
            # small array row index
            i = 1
            # small array length
            k = 1
            # check in this
            # big array column
            q = 1
            box = BoxLabel.B
        elif box == BoxLabel.B:
            if q == r:
                box = BoxLabel.D
            else:
                box = BoxLabel.C
        elif box == BoxLabel.C:
            # checking in a previously explored column
            # big array length
            a = L1
            box = BoxLabel.E
        elif box == BoxLabel.D:
            # checking in the same column        
            # big array length
            a = i+k+1
            box = BoxLabel.E
        elif box == BoxLabel.E:
            # The original source indexes both rows and columns from 1
            # and elements s(i,j) for i=1...L1, j=1...L2 actually refer
            # to numpy element data[i-1,j-1] as python/numpy index from 0
            # However slice Data(i:i+k,j) contains elements:
            # s(i+1,j),s(i+2,j),...,s(i+k,j) which translates as
            # data[i,j-1],data[i+1,j-1],...,data[i+k-1,j-1] = data[i:i+k,j-1]
            # 
            # here we want small = Data(i:i+k,r)
            small = data[i:i+k,r-1]
            # and big = Data(0:a,q)
            big = data[0:a,q-1]
            found = contains(small, big)
            box = BoxLabel.F
        elif box == BoxLabel.F:
            if found:
                box = BoxLabel.I
            else:
                box = BoxLabel.G                
        elif box == BoxLabel.G:
            q = q-1
            box = BoxLabel.H
        elif box == BoxLabel.H:
            if q < 1:
                box = BoxLabel.K
            else:
                box = BoxLabel.B
        elif box == BoxLabel.I:
            k = k+1
            box = BoxLabel.J
        elif box == BoxLabel.J:
            if (i+k) > L1:
                box = BoxLabel.O
            else:
                box = BoxLabel.B
        elif box == BoxLabel.K:
            complexity = complexity+1
            i = i+k
            box = BoxLabel.L
        elif box == BoxLabel.L:
            if (i+1) > L1:
                box = BoxLabel.O
            else:
                box = BoxLabel.M
        elif box == BoxLabel.M:
            q = r
            k = 1
            box = BoxLabel.B
        elif box == BoxLabel.N:
            i = 0
            q = r-1
            k = 1
            box = BoxLabel.B
        elif box == BoxLabel.O:
            r = r+1
            box = BoxLabel.P
        elif box == BoxLabel.P:
            if r > L2:
                box = BoxLabel.Q
            else:
                box = BoxLabel.N
        elif box == BoxLabel.Q:
            stop = True
    return complexity            
            
## Explaining the initial conditions at the start of each iteration of r
# when r changes, we are starting a new column, so
# we need to start indexing at i=0. If it is the first
# such column we know it will be a novel bit and so we index from i=1
# q is 1 less than the r index, unless it is the first column
# k always begins at 1 (a length 1 sequence to check for)
# a is the end of the pre-existing sequence in which to check
# so if we are checking in the same column it is from just before
# the end of the current sequence were checking,
# otherwise it is the full length of a previous column


def lempel_ziv_casali_loop(data):
    L1, L2 = data.shape
    complexity = 1
    for r in range(1, L2 + 1):
            i = 1 if r == 1 else 0
            q = max(r-1, 1)
            k = 1
            stop = False
            while not stop:
                a = i+k-1 if q == r else L1
                small, big = data[i:i+k, r-1], data[0:a, q-1]
                found = contains(small, big)
                if found:
                    # make subseq longer
                    k = k+1
                else:
                    # keep looking in previous column for current subseq
                    q -= 1
                if q < 1: 
                    # no more columns to check, so novel subseq
                    complexity += 1
                    i += k
                    k = 1
                    q = r
                if i + k > L1:
                    # completed checking novelty of this column, move to next
                    stop = True
    return complexity


## Lempel-Ziv entropy estimator
# adapted from:
# https://stackoverflow.com/questions/46296891/entropy-estimator-based-on-the-lempel-ziv-algorithm-using-python
# This implementation is problematic as it isn't clear whether after finding a
# match it should be continuing from the end of the matched substring or from
# one after the start of the matched substring.
#
def lempel_ziv_entropy(l):
    """
    input
    -----
    l - a 1d numpy timeseries array with dtype=bool
    """
    n = len(l)
    sum_gamma = 0

    for i in range(1, n):
        sequence = l[:i]

        for j in range(i+1, n+1):
            s = l[i:j]
            if not contains(s, sequence):
                sum_gamma += len(s)
                break

    ae = 1 / (sum_gamma / n) * math.log(n)
    return ae
    
## End of Lempel-Ziv entropy estimator


