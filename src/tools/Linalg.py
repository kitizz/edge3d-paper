from numpy import empty, sqrt

from numba import jit


@jit(nopython=True)
def eigh2(A):
    a, b, c = A[0,0], A[0,1], A[1,1]
    V = empty((2,2))
    w = empty(2)

    if b != 0:
        mag = sqrt(a**2 + 4*b**2 - 2*a*c + c**2)
        V[0, 0] = -(-a + c + mag)/(2*b)
        V[0, 1] = -(-a + c - mag)/(2*b)
        V[1] = 1

        w[0] = 0.5 * (-mag + a + c)
        w[1] = 0.5 * (mag + a + c)

    else:
        # Already diagonal
        if a < c:
            V[0] = 1, 0
            V[1] = 0, 1
            w[0] = a
            w[1] = c
        else:
            V[0] = 0, 1
            V[1] = 1, 0
            w[0] = c
            w[1] = a

    return w, V
