import numpy as np
from numpy import empty, dot, cos, sin

from skimage import draw

from numba import jit, int64


def draw_poly(im, pts, val=1):
    N = len(pts)
    vec_val = hasattr(val, '__len__')
    It = []
    Jt = []
    for i in range(1, N):
        x0, y0 = pts[i-1]
        x1, y1 = pts[i]

        I, J = draw.line(y0, x0, y1, x1)
        It.append(I[:-1])
        Jt.append(J[:-1])
        if vec_val:
            im[I, J] = np.linspace(val[i-1], val[i], len(I))
        else:
            im[I, J] = val

    It = np.hstack(It)
    Jt = np.hstack(Jt)
    coord = empty((len(It),2))
    coord[:,0] = Jt
    coord[:,1] = It
    return coord


def draw_frame(im, T, K, scale=1):
    h, w = im.shape[:2]

    P = scale*np.eye(4)
    P[3] = 1

    # fx, fy, cx, cy = cam

    Plocal = dot(K, dot(T, P))
    pu, pv = (Plocal[:2] / Plocal[2]).astype(int)

    # pu = (p[0]*fx + cx).astype(int)
    # pv = (p[1]*fy + cy).astype(int)

    colors = [(255,0,0), (0,255,0), (0,0,255)]

    for i in range(3):
        I, J = draw.line(pv[i], pu[i], pv[3], pu[3])
        good = np.where((I >= 0) & (I < h-1) & (J >= 0) & (J < w-1))[0]
        I = I[good]
        J = J[good]
        im[I, J] = colors[i]
        im[I+1, J] = colors[i]
        im[I, J+1] = colors[i]


@jit(nopython=True, cache=True)
def line(x1f, y1f, x2f, y2f):
    x1 = int(np.round(x1f))
    y1 = int(np.round(y1f))
    x2 = int(np.round(x2f))
    y2 = int(np.round(y2f))
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        IJ = np.empty((2,1), np.int64)
        IJ[:,0] = y1, x1
        return IJ[0], IJ[1]

    switch = False
    flipped = False
    if abs(dy) > abs(dx):
        switch = True
        dy, dx = dx, dy
        x1, y1, x2, y2 = y1, x1, y2, x2

    if x2 < x1:
        flipped = True
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        dx = -dx
        dy = -dy

    # yup = np.sign(dy)

    N = int(dx) + 1

    IJ = np.empty((2,N), np.int64)
    I = IJ[0]
    J = IJ[1]

    # error = -1.0
    # derror = abs(dy/dx)
    # y = y1

    slope = dy/dx
    i = 0
    for x in range(x1, x2+1):
        I[i] = np.round(y1 + i*slope)
        J[i] = x
        i += 1

        # error += derror

        # if error >= 0:
        #     y += yup
        #     error -= 1

    if not switch:
        if flipped:
            return I[::-1], J[::-1]
        else:
            return I, J
    else:
        if flipped:
            return J[::-1], I[::-1]
        else:
            return J, I


@jit(nopython=True)
def line2(x1, y1, x2, y2):
    N = int(x2 - x1 + 1)
    g = (y2 - y1) / (x2 - x1)

    x = empty(N, np.int64)
    y = empty(N, np.int64)
    for i in range(N):
        x[i] = x1 + i
        y[i] = y1 + g*i

    return y, x


@jit(nopython=True)
def _fpart(x):
    return x - int(x)


@jit(nopython=True)
def _rfpart(x):
    return 1 - _fpart(x)


@jit(nopython=True)
def line_aa(x1, y1, x2, y2, linewidth=2):
    """Draws an anti-aliased line in img from p1 to p2 with the given color."""
    dx, dy = x2 - x1, y2 - y1
    steep = abs(dx) < abs(dy)

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    flip = x2 < x1
    if flip:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    if dx < 1:
        x1 -= 0.5
        x2 += 0.5
        if flip:
            dx -= 1
        else:
            dx += 1

    grad = dy/dx
    intery = y1 + _rfpart(x1) * grad

    # Handle start point
    x1s = round(x1)
    y1s = y1 + grad * (x1s - x1)
    x1gap = _rfpart(x1 + 0.5)
    xs, ys = int(x1s), int(y1s)

    # Handle start point
    x2e = round(x2)
    y2e = y2 + grad * (x2e - x2)
    x2gap = _fpart(x2 + 0.5)
    xe, ye = int(x2e), int(y2e)

    N = (1 + linewidth) * int(xe - xs + 1)
    IJ = np.empty((2,N), np.int64)
    I = IJ[0]
    J = IJ[1]
    alpha = np.empty(N)

    # Flipping the ends around cos of some weird bug..
    J[0], I[0], alpha[0] = xs, ys, _rfpart(y1s) * x1gap
    J[1], I[1], alpha[1] = xs, ys + linewidth, _fpart(y1s) * x1gap
    J[2], I[2], alpha[2] = xe, ye, _rfpart(y2e) * x2gap
    J[3], I[3], alpha[3] = xe, ye + linewidth, _fpart(y2e) * x2gap
    i = 4
    for j in range(1, linewidth):
        J[i], I[i], alpha[i] = xs, ys + j, x1gap
        i += 1
        J[i], I[i], alpha[i] = xe, ye + j, x2gap
        i += 1

    # Main loop..
    for x in range(xs + 1, xe):
        y = int(intery)
        J[i], I[i], alpha[i] = x, y, _rfpart(intery)
        i += 1
        for j in range(1, linewidth):
            J[i], I[i], alpha[i] = x, y + j, 1
            i += 1
        J[i], I[i], alpha[i] = x, y + linewidth, _fpart(intery)
        i += 1
        intery += grad

    I, J, alpha = I[:i], J[:i], alpha[:i]

    if steep:
        if flip:
            return J[::-1], I[::-1], alpha[::-1]
        else:
            return J, I, alpha
    else:
        if flip:
            return I[::-1], J[::-1], alpha[::-1]
        else:
            return I, J, alpha


def circle_poly(cx, cy, r, N=50):
    '''
    Return a poly line describing a circle of center, (cx, cy) and radius, r
    with N line segments.
    '''
    M = N + 1
    angles = np.linspace(0, 2*np.pi, M)
    return cx + r * cos(angles), cy + r * sin(angles)


@jit(nopython=True, nogil=True)
def draw_points(im, I, J, color=1.0):
    N = len(I)
    H = im.shape[0]
    W = im.shape[1]
    im = im.reshape(H,W,-1)
    D = im.shape[2]
    c = np.empty(D)
    c[:] = color

    for n in range(N):
        i = I[n]
        j = J[n]
        if i < 0 or j < 0 or i >= H or j >= W:
            continue

        for k in range(D):
            im[i,j,k] = c[k]


@jit(nopython=True, nogil=True)
def draw_connections(im, x1, y1, x2, y2, color=1.0):
    N = len(x1)

    for n in range(N):
        i1, j1 = y1[n], x1[n]
        i2, j2 = y2[n], x2[n]
        I, J = line(j1, i1, j2, i2)

        draw_points(im, I, J, color)
