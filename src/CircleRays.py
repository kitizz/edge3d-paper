from numpy import empty, zeros, r_, dot, sin, cos, sqrt
from numpy.linalg import norm
import numpy as np

from tools import Geom, IO

from RayTracks import RayTracks
# import PlayTrack2

import pynutmeg

from numba import jit, bool_
import time

np.set_printoptions(suppress=True, linewidth=150)


class Button(object):
    def __init__(self, param):
        self.value = param.read()
        self.param = param

    def wait_pressed(self, timeout=np.inf):
        t0 = time.time()
        while self.param.read() == self.value and time.time() - t0 < timeout:
            # print(self.param.read())
            time.sleep(0.005)

        self.value = self.param.read()

    def pressed(self):
        old_val = self.value
        self.value = self.param.read()
        return self.value != old_val


def vis(Cs, qs, ns, path='tmp/rays.ply'):
    M = len(Cs)
    verts = empty((4*M,3), Cs.dtype)

    verts[::4] = Cs
    verts[1::4] = Cs + 20*qs

    verts[2::4] = Cs
    verts[3::4] = Cs + 0.5*ns

    edges = empty((2*M,2),int)
    edges[:M,0] = r_[0:2*M:2]
    edges[:M,1] = r_[1:2*M:2]

    edges[M:,0] = 2*M + r_[0:2*M:2]
    edges[M:,1] = 2*M + r_[1:2*M:2]

    IO.save_point_cloud(path, verts, edges)


@jit(nopython=True, nogil=True)
def lstsq(A, b):
    q, r = np.linalg.qr(A)
    p = np.dot(q.T, b)
    return np.dot(np.linalg.inv(r), p)


@jit(nopython=True, nogil=True)
def lstsq_L1(A, b, its=20):
    u = lstsq(A,b)
    res = empty(A.shape[0], A.dtype)
    w = np.empty_like(res)

    for i in range(its):
        res[:] = dot(A,u) - b
        w[:] = 1/sqrt(np.abs(res) + 1e-8)
        u[:] = lstsq(A * w.reshape(-1,1), b*w)

    return u


@jit(nopython=True, nogil=True)
def lstsq_percentile(A, b, percentile=0.5, chunk=5):
    u = lstsq(A,b)
    res = empty(A.shape[0], A.dtype)
    w = np.empty_like(res)
    w[:] = 1

    M = int(percentile * A.shape[0])
    m = M % chunk

    for n in range(m, M + 1, chunk):
        res[:] = dot(A,u) - b
        ind = nth_largest(np.abs(res), n=n)
        w[:] = 1
        w[ind] = 0
        u[:] = lstsq(A * w.reshape(-1,1), b*w)

    return u, np.where(w)[0]


@jit(nopython=True, nogil=True)
def nth_largest(v, n=1):
    high = empty(n, v.dtype)
    high[:] = v[0]
    ind = empty(n, np.int64)
    ind[:] = 0

    N = v.shape[0]
    for i in range(N):
        val = v[i]
        if val > high[0]:
            high[0] = val
            ind[0] = i
            for j in range(1, n):
                if val <= high[j]:
                    break
                high[j-1] = high[j]
                ind[j-1] = ind[j]
                high[j] = val
                ind[j] = i

    return ind


@jit(nopython=True, nogil=True)
def sum_ax(A, axis):
    if axis == 0:
        N = A.shape[0]
        M = A.shape[1]

        tot = zeros(M, A.dtype)
        for i in range(N):
            tot += A[i]

        return tot

    else:
        N = A.shape[1]
        M = A.shape[0]

        tot = zeros(M, A.dtype)
        for i in range(M):
            for j in range(N):
                tot[i] += A[i, j]

        return tot


@jit(nopython=True, nogil=True)
def mean(A, axis):
    tot = sum_ax(A, axis)

    if axis == 0:
        return tot / A.shape[0]
    else:
        return tot / A.shape[1]


@jit(nopython=True, nogil=True)
def norm_ax(A, axis):
    if axis == 0:
        N = A.shape[0]
        M = A.shape[1]

        tot = zeros(M, A.dtype)
        for i in range(N):
            tot += A[i]**2

        return sqrt(tot)

    else:
        N = A.shape[1]
        M = A.shape[0]

        tot = zeros(M, A.dtype)
        for i in range(M):
            for j in range(N):
                tot[i] += A[i, j]**2

        return sqrt(tot)


@jit(nopython=True, nogil=True)
def normalize(A):
    return A / Geom.norm1d(A)


# _fig = pynutmeg.figure('rays', 'figs/ray.qml')
# _fig.set_gui('figs/sil_gui.qml')
# _next = Button( _fig.parameter('next') )
# def _plot_circle(cx, cy, r, n=100):
#     thetas = np.linspace(0, 2*np.pi, n)
#     x = r*cos(thetas) + cx
#     y = r*sin(thetas) + cy
#     _fig.set('ax.circle', x=x, y=y)


# def _plot_rays(cs, rays, l=2):
#     end = cs + l*rays
#     _fig.set('ax.rays',
#              x=cs[0], y=cs[1],
#              endX=end[0], endY=end[1])


# def _plot_points(P):
#     _fig.set('ax.points', x=P[0], y=P[1])

# def _plot_line(P):
#     _fig.set('ax.line', x=P[0], y=P[1])


@jit(nopython=True)
def intersect_pairs(c, q):
    '''
    Find intersections of each sequential pair of rays
    c: ray origin
    q: ray direction
    '''
    N = c.shape[1]
    lines = empty((N,3))
    lines[:,0] = -q[1]
    lines[:,1] = q[0]
    lines[:,2] = -sum_ax(c.T * lines[:,:2], axis=1)

    points = empty((N-1,2))
    for i in range(N-1):
        p = Geom.cross3(lines[i], lines[i+1])
        points[i] = p[:2] / p[2]

    return points.T


@jit(nopython=True)
def find_cluster(p, radius):
    N = p.shape[0]
    r2 = radius*radius
    best_st = 0
    best_sz = 0
    st = -1
    sz = 0
    nbad = 0
    limbad = 2
    j = 0
    for i in range(N-1):
        x, y = p[j] - p[i+1]
        d2 = x*x + y*y
        if d2 < r2:
            sz += 1
            nbad = 0
            if st < 0:
                st = j
            j += 1

        elif nbad < limbad:
            nbad += 1
            if st < 0:
                j = i+1

        else:
            st = -1
            sz = 0
            j = i+1

        if sz > best_sz:
            best_st = st
            best_sz = sz

    return np.arange(best_st, best_st + best_sz + 1)


@jit(nopython=True)
def solve_circle(A, b, max_dev, perc=0.9, chunk=10):
    u, sel = lstsq_percentile(A, b, perc, chunk)
    r = u[2]
    # center = u[:2]

    A = A[sel]
    b = b[sel]
    if r < 0:
        A[:,2] *= -1

    rs = np.linspace(0, r, 20).astype(A.dtype)
    for i, nr in enumerate(rs):
        u[:2] = lstsq(A[:,:2], b - nr)
        u[2] = nr

        res = dot(A, u) - b
        max_res = np.abs(res).max()
        # Use circle center, x (u[0]) as proxy for depth
        if max_res < u[0]*max_dev:
            # r = nr
            # center = u[:2]
            break

    return u


@jit(nopython=True)
def det3(A):
    return A[0,0] * (A[1,1]*A[2,2] - A[1,2]*A[2,1]) -\
        A[0,1] * (A[1,0]*A[2,2] - A[1,2]*A[2,0]) +\
        A[0,2] * (A[1,0]*A[2,1] - A[1,1]*A[2,0])


@jit(nopython=True)
def solve_circle_ransac(A, b, eps, percentile, its=30):
    '''
    Using a version of LO-RANSAC.
    Ref: http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
    '''
    N = A.shape[0]
    M = int(percentile * N)

    best_circle = zeros(3, A.dtype)
    best_inliers = empty(0, np.int64)
    best = 0

    if N < 10:
        return best_circle, best_inliers, False

    for it in range(its):
        sample = np.random.choice(N, 3, replace=False)
        A3 = A[sample]
        b3 = b[sample]
        A3[2,:2] = 0, 1
        b3[2] = 0

        if abs(det3(A3)) < 1e-8:
            continue

        circle = dot(np.linalg.inv(A3), b3)
        # Find and check inliers (eps scaled by depth from camera)
        res = np.abs( dot(A, circle) - b )
        inliers = np.where( res < eps*circle[0] )[0]

        n = len(inliers)
        if n > best:
            # Iterative local optimization (see ref)
            K = np.linspace(1.0, 3.0, 5)
            for k in K[::-1]:
                if len(inliers) < 10:
                    break
                circle = lstsq(A[inliers], b[inliers])
                res = np.abs( dot(A, circle) - b )
                inliers = np.where( res < k*eps*circle[0] )[0]

            n = len(inliers)

            best = n
            best_inliers = inliers
            best_circle = circle

        if best >= M and it > 10:
            break

    good_fit = it < its - 1

    if good_fit:
        if abs(best_circle[2]) < 2*eps:
            best_circle[2] = 0

    return best_circle, best_inliers, good_fit


@jit(nopython=True)
def solve_point_ransac_valid(A, b, eps, percentile, valid, its=100):
    '''
    Using a version of LO-RANSAC.
    Ref: http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
    '''
    # np.random.seed(11231)
    pool = np.where( valid )[0]
    N = len(pool)
    # N = A.shape[0]
    M = int(percentile * N)

    best_point = zeros(2, A.dtype)
    best_inliers = empty(0, np.int64)
    best = 0

    if N < 10:
        return best_point, best_inliers, False

    for it in range(its):
        sample = pool[ np.random.choice(N, 2, replace=False) ]
        A2 = A[sample]
        b2 = b[sample]

        if abs(A2[0,0]*A2[1,1] - A2[1,0]*A2[0,1]) < 1e-8:
            continue

        point = dot(np.linalg.inv(A2), b2)
        # Find and check inliers (eps scaled by depth from camera)
        res = np.abs( dot(A, point) - b )
        inliers = np.where( (res < eps)  )[0]

        n = len(inliers)
        if n > best:
            # Iterative local optimization (see ref)
            K = np.linspace(1.0, 3.0, 4)
            for k in K[::-1]:
                if len(inliers) < 10:
                    break
                point = np.linalg.lstsq(A[inliers], b[inliers])[0]
                res = np.abs( dot(A, point) - b )
                inliers = np.where( (res < k*eps) )[0]

            n = len(inliers)

            best = n
            best_inliers = inliers
            best_point = point

        if best >= M and it > 15:
            break

    good_fit = it < its - 1

    return best_point, best_inliers, good_fit


@jit(nopython=True)
def solve_point_ransac(A, b, eps, percentile, its=100):
    '''
    Using a version of LO-RANSAC.
    Ref: http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
    '''
    # np.random.seed(11231)
    N = A.shape[0]
    M = int(percentile * N)

    best_point = zeros(2, A.dtype)
    best_inliers = empty(0, np.int64)
    best = 0

    if N < 10:
        return best_point, best_inliers, False

    for it in range(its):
        sample = np.random.choice(N, 2, replace=False)
        A2 = A[sample]
        b2 = b[sample]

        if abs(A2[0,0]*A2[1,1] - A2[1,0]*A2[0,1]) < 1e-8:
            continue

        point = dot(np.linalg.inv(A2), b2)
        # Find and check inliers (eps scaled by depth from camera)
        res = np.abs( dot(A, point) - b )
        inliers = np.where( (res < eps) )[0]

        n = len(inliers)
        if n > best:
            # Iterative local optimization (see ref)
            K = np.linspace(1.0, 3.0, 4)
            for k in K[::-1]:
                if len(inliers) < 10:
                    break
                point = np.linalg.lstsq(A[inliers], b[inliers])[0]
                res = np.abs( dot(A, point) - b )
                inliers = np.where( (res < k*eps) )[0]

            n = len(inliers)

            best = n
            best_inliers = inliers
            best_point = point

        if best >= M and it > 15:
            break

    good_fit = it < its - 1

    return best_point, best_inliers, good_fit


@jit(nopython=True)
def prepare(cs, qs):
    # First find a point that minimises distances to rays
    # Use this to determine normal directions
    N = cs.shape[0]
    A = empty((N,3), cs.dtype)

    imag = 1/norm_ax(qs, axis=1)
    A[:, 0] = -qs[:,1]*imag
    A[:,1] = qs[:,0]*imag

    # q_mean = mean(qs, axis=0)
    # to_reverse = dot(qs, q_mean) < 0
    to_reverse = qs[:,0] < 0
    A[to_reverse] *= -1
    ns = A[:,:2]

    A[:,2] = 1
    b = sum_ax(cs*ns, axis=1)

    return A, b


@jit(nopython=True)
def fit_circle_2d(cs, qs, eps=1e-3):
    '''
    return circle, goodfit: [cx cy r]
    '''
    A, b = prepare(cs, qs)

    # circle = lstsq_percentile(A, b, 0.9, 10)
    # circle = solve_circle(A, b, eps, 0.95, 5)
    goodfit = False
    circle = zeros(3, cs.dtype)
    for i in range(3):
        # try:
        circle, inliers, goodfit = solve_circle_ransac(A, b, 2*eps, 0.3, 1000)
        if goodfit:
            break
        # except ValueError:
            # pass

    circle[2] = abs(circle[2])
    return circle, goodfit


@jit(nopython=True, nogil=True)
def dot_vec(Rs, v):
    N = v.shape[0]
    out = np.empty_like(v)

    for i in range(N):
        out[i] = dot(Rs[i], v[i])

    return out


@jit(nopython=True)
def circle_space_ransac(depths, angles, init_depth, angle_support, eps, valid, its=100, fig=None):
    '''
    This space is used because anything approximating a circle generates
    linear lines. The slope of which is a function of the radius of the circle.

    depths: Depths of intersections of rays with current ray
    angles: I say angles, but really these values should be calculated from the
        normalized rays (ux, uy) as: (1 - ux)/uy
    init_depth: A guess as to the rough "main" depth of the cluster of interest

    Ref: http://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
    '''
    N = len(depths)

    ab_angles = np.abs(angles)
    good_angles = (ab_angles > angle_support) & (ab_angles < 2)
    pool = np.where( good_angles & valid )[0]
    M = len(pool)

    d = depths - init_depth
    best_line = zeros(2, angles.dtype)
    best_line[1] = np.inf
    best_inliers = empty(0, np.int64)
    best = 0

    # base = np.arange(0, N, 1, np.int64)
    base = np.where( (ab_angles <= angle_support) & (np.abs(d) < 10*eps)  )[0]
    B = len(base)

    if M < 5 or B < 5:
        return init_depth, 100.0, best_inliers, False

    # TODO: Put the actual maths in the comments here...
    # Solving the following will provide the x-intersection of the line and its inverse gradient
    A = empty((N,2), angles.dtype)
    b = empty(N, angles.dtype)
    A[:,0] = 1
    A[:,1] = angles
    b[:] = d

    min_inliers = int(0.03 * N)
    good_fit = False

    for it in range(its):
        s0 = np.random.randint(B)
        s1 = np.random.randint(M)
        ind0 = base[s0]
        ind1 = pool[s1]

        # Points are centered around init depth. Angle of sampled point is the guessed model
        dx, dy = d[ind1] - d[ind0], angles[ind1] - angles[ind0]
        mag = sqrt(dx*dx + dy*dy)
        if mag == 0:
            continue
        nx, ny = -dy/mag, dx/mag

        # n . p is distance to model
        dists = (d - d[ind0]) * nx + (angles - angles[ind0]) * ny
        inliers = np.where( (np.abs(dists) < 2*eps) & valid )[0]

        n = len(inliers)
        # if n > best:
        if n > best or n > min_inliers:
            # Iterative local optimization (see ref)
            K = np.linspace(1.0, 5.0, 5)
            for k in K[::-1]:
                if len(inliers) < 10:
                    break
                line = lstsq(A[inliers], b[inliers])
                mag = sqrt(1 + line[1]**2)
                if mag == 0:
                    n = 0
                    break

                nx, ny = -1/mag, line[1]/mag
                dists = (d - line[0]) * nx + angles * ny
                inliers = np.where( (np.abs(dists) < k*eps) & valid )[0]
                n = len(inliers)

            better = False
            if best == 0 and n >= 10:
                better = True
            elif best < min_inliers and n >= 10:
                better = True
            # elif n >= min_inliers and abs(line[1]) < abs(best_line[1]):
            #     better = True
            elif n > best and n >= 10:
                better = True

            # if n >= 10:
            # if ((best == 0 and n >= 10) or (best > 0 and n >= best)) and abs(line[1]) < 1.1*abs(best_line[1]):
            if better:
                best = n
                best_inliers = inliers
                best_line = line

        if best >= min_inliers and it > 10:
            good_fit = True
            break

    # good_fit = best >= min_inliers

    return init_depth + best_line[0], best_line[1], best_inliers, good_fit


@jit(nopython=True)
def sequential_circle_space(depths, angles, init_depth, angle_support, eps, its=100):
    n = 2

    best = 0
    best_inliers = empty(0, np.int64)
    best_r = np.inf
    best_d = 0

    valid = np.ones(len(depths), bool_)

    for i in range(n):
        d, r, inliers, good = circle_space_ransac(depths, angles, init_depth, angle_support, eps, valid, its)
        valid[inliers] = False

        if abs(r) < abs(best_r) and len(inliers) > 0.4*best:
            best = len(inliers)
            best_inliers = inliers
            best_r = r
            best_d = d

    return best_d, best_r, best_inliers, True


@jit(nopython=True)
def fit_point(cs, qs, eps=1e-3, percentile=0.5):
    N = len(cs)

    A = empty((N,2))
    A[:,0] = -qs[:,1]
    A[:,1] = qs[:,0]
    b = sum_ax(cs*A, axis=1)

    valid = np.ones(N, bool_)

    best = 0
    best_inliers = empty(0, np.int64)
    best_r = np.inf
    best_d = 0

    tries = 2
    for i in range(tries):
        point, inliers, good = solve_point_ransac_valid(A, b, eps, percentile, valid)

        d, r = point

        if abs(r) < abs(best_r) and len(inliers) > 0.8*best:
            best = len(inliers)
            best_inliers = inliers
            best_r = r
            best_d = d

        if i < tries - 1:
            valid[inliers] = False

    return best_d, best_r, best_inliers


@jit(nopython=True)
def fit_tangent_circle(cs, qs):
    N = len(cs)

    A = empty((N,2), cs.dtype)
    b = empty(N, A.dtype)

    # flip = (qs[:,0] < 0)*2 - 1
    nx = -qs[:, 1]
    ny = qs[:, 0]

    A = empty((N,2), qs.dtype)
    A[:,0] = nx
    A[:,1] = ny - 1

    # A = qs  #+ flip.reshape(-1, 1)
    # A[:, 1] *= -1
    b = cs[:, 0] * nx + cs[:, 1] * ny  # dot(ci, ni)
    # A[:, 0] -= 1  # for radius

    d, r = np.linalg.lstsq(A, b)[0]

    return d, r


def _tinker():
    import Drawing
    path = 'tmp/circle3.npz'
    npz = np.load(path)
    P, Q, eps = npz['P'], npz['Q'], np.float(npz['eps'])

    fig = pynutmeg.figure('circle', 'figs/rayplay.qml')
    x1, y1 = P
    x2, y2 = P + 1.5*Q
    fig.set('ax.rays', x=x1, y=y1, endX=x2, endY=y2)

    bad = 0
    circle, good = fit_circle_2d(P.T, Q.T, eps)
    its = 100

    # t0 = time.clock()
    for i in range(its):
        print(i, "...")
        circle, good = fit_circle_2d(P.T, Q.T, 2*eps)
        if circle[2] == 0:
            circle[2] = 1e-6
        # if circle[1] < 0:
        #     bad += 1
        #     print(" ^ Bad one!")
        cx, cy = Drawing.circle_poly(*circle)
        fig.set('ax.circle', x=cx, y=cy)
        time.sleep(0.1)
    # dt = int( ((time.clock() - t0)/its) * 1e6 )

    print("Bad:", bad)
    # print("Time: {} us".format(dt))

    pynutmeg.wait_for_nutmeg()
    pynutmeg.check_errors()


def _tinker2():
    npz = np.load('tmp/cluster_line.npz')
    d = npz['d']
    a = npz['a']
    d0 = npz['d0']
    eps = npz['eps']

    print("Eps:", eps)

    fig = pynutmeg.figure('circle', 'figs/rayplay.qml')

    doff = d - d0
    fig.set('ax.P0', x=doff, y=a)

    c, r, cluster, good = circle_space_ransac(d, a, d0, eps=2e-3, fig=fig)
    print("c, r:", c, r)

    fig.set('ax.P1', x=doff[cluster], y=a[cluster])

    pynutmeg.wait_for_nutmeg()


if __name__ == '__main__':
    _tinker2()
