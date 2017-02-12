from numpy import empty, sqrt

from numba import jit


@jit(nopython=True, nogil=True)
def norm3(a):
    x, y, z = a
    return sqrt(x*x + y*y + z*z)


@jit(nopython=True, nogil=True)
def norm1d(a):
    N = a.shape[0]
    mag = 0
    for i in range(N):
        mag += a[i]**2

    return sqrt(mag)


@jit(nopython=True, nogil=True)
def cross3(a, b):
    out = empty(3, a.dtype)
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]
    return out


@jit(nopython=True)
def cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]
