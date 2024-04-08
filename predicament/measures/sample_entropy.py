import numba
import numpy as np

@numba.njit()
def construct_templates_(x, m=2):
    res = np.empty((x.size - m + 1, m), dtype=x.dtype.type)
    for i in range(res.shape[0]):
        res[i] = x[i:i+m]
    return res

@numba.njit()
def is_match_numba(a, b, r):
    for i in range(a.size):
        if np.abs(a[i] - b[i]) >= r:
            return False
    return True

@numba.njit()
def get_matches_numba(t, r):
    res = 0
    for i in range(t.shape[0] - 1):
        for j in range(i+1, t.shape[0]):
            if is_match_numba(t[i], t[j], r):
                res += 1
    return res

@numba.njit()
def sample_entropy_numba(x, w, r):
    B = get_matches_numba(construct_templates_(x, w), r)
    A = get_matches_numba(construct_templates_(x, w+1), r)
    return -np.log(A/B)
