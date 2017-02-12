from __future__ import print_function, division
from numpy import empty, zeros, ones, r_, c_, dot, sqrt
from numpy.linalg import norm
import numpy as np

# from scipy import spatial
# from scipy import ndimage
# from scipy import linalg as splinalg
# from sklearn import neighbors

from numba import jit, bool_

from . import Linalg
from .FastMath import cross3, norm3


@jit(nopython=True, nogil=True)
def global_to_local(Rs, ts):
    lRs = np.empty_like(Rs)
    lts = np.empty_like(ts)

    for i in range(ts.shape[0]):
        Rw = Rs[i]
        lRs[i] = Rw.T
        lts[i] = -dot(Rw.T, ts[i])

    return lRs, lts


def cam_params(f, w, h, sw, cx=None, cy=None):
    '''
    Assumes non distorted pixels (fx == fy)
    f: Focal length in real units
    sw: Sensor width and height in real units
    w, h: Pixel width and height
    cx, cy: Offset of center
    '''
    if cx is None:
        cx = w*0.5
    if cy is None:
        cy = h*0.5

    alpha = w/sw

    fx = f*alpha
    fy = f*alpha

    return np.r_[fx, fy, cx, cy]


def blender_to_R(euler):
    '''
    https://en.wikipedia.org/wiki/Euler_angles (Z1Y2X3)
    Also, blender's camera is rotated 180deg around the x axis
    So Y and Z columns need to be adjusted accordingly.
    Never lose this code!!
    '''
    one = False
    if euler.ndim == 1:
        one = True
        euler = euler.reshape(1,3)

    c1, s1 = np.cos(euler[:,0]), np.sin(euler[:,0])
    c2, s2 = np.cos(euler[:,1]), np.sin(euler[:,1])
    c3, s3 = np.cos(euler[:,2]), np.sin(euler[:,2])

    N = euler.shape[0]
    R = np.empty((N,3,3))

    R[:,0,0] = c2*c3
    R[:,0,1] = -(c3*s1*s2 - c1*s3)
    R[:,0,2] = -(c1*c3*s2 + s1*s3)

    R[:,1,0] = c2*s3
    R[:,1,1] = -(c1*c3 + s1*s2*s3)
    R[:,1,2] = -(c1*s2*s3 - c3*s1)

    R[:,2,0] = -s2
    R[:,2,1] = -(c2*s1)
    R[:,2,2] = -(c1*c2)

    if one:
        return R[0]
    else:
        return R


@jit(nopython=True, nogil=True)
def fit_normal_2d(P):
    N = P.shape[0]

    sm = zeros(2)
    for i in range(N):
        sm += P[i]
    P = N*P - sm

    cov = zeros((2,2))
    for i in range(N):
        p = P[i:i+1]
        cov += dot(p.T, p)

    w, V = Linalg.eigh2(cov)
    # we, Ve = np.linalg.eigh(cov)
    # print(w, V)
    # print(we, Ve)
    # print(cov)
    n = V[:,0]
    mag = sqrt((n**2).sum())
    n /= mag
    curv = w[0] / w.sum()
    return n, curv


@jit(nopython=True, cache=True)
def unproject_normal(q, n):
    m = q + n
    m[2] = 1  # Homogenize

    m -= q * (dot(m, q)/(q**2).sum())
    mag = np.sqrt((m**2).sum())
    m *= 1/mag   # Normalize

    return m


@jit(nopython=True, cache=True)
def unproject_normals(Qs, Ns):
    N = Qs.shape[0]
    out = empty((N,3), Ns.dtype)

    for i in range(N):
        m = out[i]
        q = Qs[i]
        n = Ns[i]

        m[:2] = q[:2] + n
        m[2] = 1  # Homogenize

        m -= q * (dot(m, q)/(q**2).sum())
        mag = np.sqrt((m**2).sum())
        m *= 1/mag   # Normalize

        # out[i] = m

    return out


@jit(nopython=True, nogil=True)
def project_normal(q, n):
    '''
    Assumes q[2] == 1
    '''
    dn = q + n
    dn[:2] = dn[:2]/dn[2] - q[:2]
    mag = np.sqrt((dn[:2]**2).sum())
    dn[:2] *= 1/mag
    dn[2] = 0
    return dn


@jit(nopython=True, nogil=True)
def line_distance_3d(p1, q1, p2, q2):
    '''
    Return the shortest distance between two lines in 3D
    return dist: dist between lines
    return depth: distance along q1
    '''
    n = cross3(q1, q2)
    mag = norm3(n)
    if mag == 0:
        # Lines are parallel
        dp = p2 - p1
        return norm3( dp - dot(q1, dp) ), 1.0

    n /= mag
    dist = abs( dot(n, p2 - p1) )
    n2 = cross3(q2, n)
    depth = dot(p2 - p1, n2) / dot(q1, n2)

    return dist, depth


@jit(nopython=True, nogil=True)
def intersect_2_planes(P1, P2):
    ''' Determine the line of intersection between 2 planes '''
    u = cross3(P1[:3], P2[:3])
    if np.all(u == 0):
        return zeros(3, P1.dtype), zeros(3, P1.dtype), False

    k = np.argmax( np.abs(u) )
    i = (k + 1) % 3
    j = (k + 2) % 3

    p = empty(3, P1.dtype)

    den = 1/u[k]
    # print(u[k], P1[i] * P2[j] - P2[i] * P1[j])
    p[i] = (P1[j] * P2[3] - P2[j] * P1[3]) * den
    p[j] = (P2[i] * P1[3] - P1[i] * P2[3]) * den
    p[k] = 0

    return p, u, True


# def intersect_planes_with_plane(planes, o, normal):
#     N = len(planes)

#     p = empty((N, 3), planes.dtype)
#     v = empty((N, 3), planes.dtype)
#     good = empty(N, np.uint8)

#     for n in range(N):
#         u = cross3(planes[n,:3], normal)
#         if np.all(u == 0):
#             p[n] = 0
#             v[n] = 0
#             good[n] = 0
#             continue

#         k = np.argmax(np.abs(u))
#         i = (k + 1) % 3
#         j = (k + 2) % 3


@jit(nopython=True, nogil=True)
def intersect_line_plane(p, u, P):
    '''
    Determine the point at which the line intersects the plane
    ref: https://goo.gl/BzaC9j

    p, u: Ray such that (x,y) = p + t*u
    return: t, valid_intersection
    '''
    det = u[0]*P[0] + u[1]*P[1] + u[2]*P[2]
    # den = np.signbit(det) * max(abs(det), 1e-10)
    if det != 0:
        t = -(p[0]*P[0] + p[1]*P[1] + p[2]*P[2] + P[3]) / det
        return t, True
    else:
        return 0, False


@jit(nopython=True, nogil=True)
def points_to_line_distance(p, v, P):
    N = P.shape[0]
    d = empty(N, P.dtype)

    for i in range(N):
        w = P[i] - p
        d[i] = norm3(cross3(v, w))

    return d


@jit(nopython=True, nogil=True)
def point_to_segments_distance(p, P1, P2):
    N = len(P1)
    V = P2 - P1

    Q = p - P1
    # p_out = empty((N,3), P1.dtype)
    off_segs = empty(N, bool_)

    dists = empty(N, P1.dtype)
    for i in range(N):
        v = V[i]
        q = Q[i]
        mag = norm3(v)
        if mag != 0:
            v /= mag
            t0 = dot(v, q)
            # t = t0/mag
            if t0 <= 0:
                dists[i] = norm3(q)
                off_segs[i] = True
            elif t0 >= mag:
                dists[i] = norm3(p - P2[i])
                off_segs[i] = True
            else:
                dists[i] = norm3(q - v*t0)
                off_segs[i] = False

        else:
            # A simple point distance
            dists[i] = norm3(q)
            off_segs[i] = True

    return dists, off_segs



@jit(nopython=True, nogil=True)
def check_intersect_cone_ray(cone_center, cone_ray, cone_radius, ray_center, ray, cone_height=np.inf):
    '''
    cone_center: Point of cone
    cone_ray: Direction of cone body from point
    cone_radius: Growth of cone radius in units/distance along cone_ray from center
    ray_center, ray: The ray...

    Note: cone_ray and ray MUST be unit length

    ref: http://www.geometrictools.com/Documentation/IntersectionLineCone.pdf
    '''
    delta = ray_center - cone_center
    # First check if the start is in the cone:
    depth = dot(delta, cone_ray)
    dist = norm3( delta - depth*cone_ray )  # Distance to cone_ray
    in_cone = depth > 0 and dist <= depth*cone_radius
    if in_cone and depth < cone_height:
        return True

    # Do a full check
    cray = cone_ray.reshape(3,1)

    I = np.identity(3, ray.dtype)
    I *= 1.0/(1.0 + cone_radius*cone_radius)

    M = dot(cray, cray.T) - I
    c2 = dot(ray, dot(M, ray))
    c1 = dot(ray, dot(M, delta))
    c0 = dot(delta, dot(M, delta))

    ts = empty(2, ray.dtype)
    if c2 == 0:
        if c1 == 0:
            # Ray sits on the cone entirely if c0 == 0
            return c0 == 0 and depth <= cone_height
        else:
            ts[0] = -c0/c1
            ts[1] = -1

    else:
        d = c1*c1 - c0*c2
        if d < 0:
            return False
        drt = sqrt(d)
        ts[0] = (-c1 + drt)/c2
        ts[1] = (-c1 - drt)/c2

    # Check if either t0 or t1 are positive and collide at a good depth
    for t in ts:
        if t >= 0:
            p = ray_center + t*ray
            depth_p = dot( p - cone_center, cone_ray )
            if 0 <= depth_p <= cone_height:
                return True

    return False


def _fbb_helper(center, size, res, dtype):
    b_min = center - 0.5*size
    b_max = center + 0.5*size

    N = res*res*res
    P = empty((N,3), dtype)

    xs, ys, zs = [ np.linspace(b_min[i], b_max[i], res) for i in range(3) ]
    i = 0
    for x in xs:
        for y in ys:
            for z in zs:
                P[i] = x, y, z
                i += 1

    return P


def frustrum_bounding_box(Rs, ts, cam, bbox_inds, bboxes, res=None, center=None, size=None, mean_depth=1.5):
    if res is None:
        res = 50

    if center is None:
        Rw, tw = global_to_local(Rs, ts)
        # 3rd column of rotation is the view's center ray (camera z-axis)
        qs = Rw[:,:,2]
        b_min = np.minimum( tw.min(axis=0), (tw + mean_depth*qs).min(axis=0) )
        b_max = np.maximum( tw.max(axis=0), (tw + mean_depth*qs).max(axis=0) )
        size = (b_max - b_min)*1.2
        center = r_[0.0, 0.0, 1.0]

    # Generate a grid of points
    P = _fbb_helper(center, size, res, Rs.dtype)
    count = zeros(P.shape[0], np.uint8)

    fx, fy, cx, cy = cam

    B = len(bbox_inds)
    for i in range(B):
        f = bbox_inds[i]
        minu, minv, maxu, maxv = bboxes[i]
        minu = (minu - cx)/fx
        minv = (minv - cy)/fy
        maxu = (maxu - cx)/fx
        maxv = (maxv - cy)/fy

        R, t = Rs[f], ts[f]
        x, y, z = dot(R, P.T) + t[:,None]
        z[z <= 0] = -1
        u = x/z
        v = y/z
        inside = (u >= minu) & (u <= maxu) & (v >= minv) & (v <= maxv) & (z > 0)
        count[ inside ] += 1

    # From all the points that were contained by the bounding box
    sel = np.where(count == len(bbox_inds))[0]
    p = P[sel]

    if len(sel) < 5:
        import IO
        IO.export_cam('tmp/bounding_cam.py', Rs, ts, 29.97)
        IO.save_point_cloud('tmp/bound_grid_full.ply', P)

        sel = np.where(count > 0)[0]
        p = P[sel]
        IO.save_point_cloud('tmp/bound_grid.ply', p)
        raise ValueError("Bad bounds in the grid")

    b_min = p.min(axis=0)
    b_max = p.max(axis=0)

    return b_min, b_max


@jit(nopython=True, nogil=True)
def intersect_ray_aabb(c, q, b_min, b_max):
    '''
    c, q: Ray center and direction
    center, size: Center and size of aabb

    Inspired by https://tavianator.com/fast-branchless-raybounding-box-intersections/
    '''
    # Check if the ray touches the map at all
    iq = 1/q
    if np.all( (c >= b_min) & (c <= b_max) ):
        return True

    # Want the latest point at which the ray crosses a zero
    t0 = (b_min - c)*iq
    t1 = (b_max - c)*iq
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    tminmax = tmin.max()
    tmaxmin = tmax.min()

    # min's max should be smaller than max's min.
    # Also the smallest max should be greater than one (not behind ray)
    return (tminmax < tmaxmin) & (tmaxmin >= 0)


@jit(nopython=True, nogil=True)
def to_plucker(Cs, Qs):
    N = Cs.shape[0]

    Pl = empty((N,6), Cs.dtype)

    for i in range(N):
        Pl[i, :3] = Qs[i]
        Pl[i, 3:] = cross3(Qs[i], Cs[i])

    return Pl


@jit(nopython=True, nogil=True)
def check_plucker(p1, p2):
    return p1[0]*p2[3] + p1[1]*p2[4] + p1[2]*p2[5] +\
        p1[3]*p2[0] + p1[4]*p2[1] + p1[5]*p2[2]


@jit(nopython=True, nogil=True)
def cross_plucker(p, P):
    '''
    Check the crossing of one plucker line (p) with many (P)
    '''
    N = P.shape[0]

    w = empty(N, P.dtype)
    W = p * P
    # a, b = p[:3], p[3:]

    for i in range(N):
        s = 0
        for j in range(6):
            s += W[i, j]
        w[i] = s

    return w


@jit(nopython=True, nogil=True)
def point_to_shards_distance(p, Cs, Q1, Q2, planes, inds):
    # TODO: Need an algorithm for determining the actual distance from a point to a shard
    # Do this by projecting point to plane, barycentric coords gives whether inside shard
    # If inside, do planar distance. If outside, do point -> ray distance.
    N = len(inds)
    dists = empty(N, Cs.dtype)
    off_shard = empty(N, bool_)
    # ps = empty((N,3), Cs.dtype)

    for i, ind in enumerate(inds):
        c = Cs[ind]
        q1, q2 = Q1[ind], Q2[ind]
        # n = planes[ind,:3]
        n = cross3(q1, q2)
        n /= norm3(n)
        # Recenter at shard origin
        w = p - c
        # Find barycentric coords of p0 (derived from http://math.stackexchange.com/a/544947)
        # This isn't a finite triangle, so we're not interested in all 3 values
        b = dot( cross3(q1, w), n )
        a = dot( cross3(w, q2), n )

        # ps[i] = c + (w - n * dot(w,n))

        off_shard[i] = True
        if a >= 0 and b >= 0:
            off_shard[i] = False
            # Projected point is "inside" shard
            dists[i] = abs(dot(w, n))  # Distance to plane

        elif a < 0 and b < 0:
            # Point is behind shard, and closest to center point
            dists[i] = norm3(w)

        elif a > 0:
            # Point is closest to q1
            # dists[i] = norm3( cross3(q1, w) )
            dists[i] = norm3( w - q1 * dot(w, q1))
        else:
            # Closest to q2
            # dists[i] = norm3( cross3(q2, w) )
            dists[i] = norm3( w - q2 * dot(w, q2))

    return dists, off_shard


@jit(nopython=True, nogil=True)
def point_to_tri_distance(p, tri):

    # TODO: Need to do the final check for the "back" of the triangle. This is taken from the shard code...

    c = tri[0]
    q1 = tri[1] - c
    q2 = tri[2] - c
    # q1, q2 = Qs[ind], Qs[ind + 1]

    n = cross3(q1, q2)
    n /= norm3(n)
    # Recenter at shard origin
    w = p - c
    # Find barycentric coords of p0 (derived from http://math.stackexchange.com/a/544947)
    # This isn't a finite triangle, so we're not interested in all 3 values
    b = dot( cross3(q1, w), n )
    a = dot( cross3(w, q2), n )

    # ps[i] = c + (w - n * dot(w,n))

    if a >= 0 and b >= 0:
        # Projected point is "inside" shard
        dists[i] = abs(dot(w, n))  # Distance to plane

    elif a < 0 and b < 0:
        # Point is behind shard, and closest to center point
        dists[i] = norm3(w)

    elif a > 0:
        # Point is closest to q1
        # dists[i] = norm3( cross3(q1, w) )
        dists[i] = norm3( w - q1 * dot(w, q1))
    else:
        # Closest to q2
        # dists[i] = norm3( cross3(q2, w) )
        dists[i] = norm3( w - q2 * dot(w, q2))

    return dists#, ps


@jit(nopython=True, nogil=True)
def points_to_tris_distance(p, tris):
    # TODO: Quickly filter for nearest triangle before doing final distance search
    # KD Tree on verts? And have a verts -> tri mapping.
    N = len(inds)
    dists = empty(N, tris.dtype)
    # ps = empty((N,3), Cs.dtype)

    for i in range(N):
        pass


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


# @jit(nopython=True, nogil=True)
def _solve_origin(U):
    N = U.shape[0]
    A = empty((N+3, 3), U.dtype)
    b = empty(N+3, U.dtype)

    A[N:] = 0
    s = 1e-9
    for i in range(3):
        A[N+i,i] = s

    A[:N] = U[:,:-1]
    b[:N] = -U[:,-1]
    b[N:] = 0

    u = lstsq(A, b)
    return u


# @jit(nopython=True, nogil=True)
def fit_hyperline(U):
    o = _solve_origin(U)

    # Assuming just 1 nullspace
    P = U[:,:-1]
    e_vals, e_vecs = np.linalg.eig(dot(P.T, P))
    V = e_vecs[:, np.argmin(e_vals)]

    return o, V


@jit(nopython=True, nogil=True)
def make_skew(w):
    K = empty((3,3), w.dtype)
    K[0,0], K[0,1], K[0,2] = 0, -w[2], w[1]
    K[1,0], K[1,1], K[1,2] = w[2], 0, -w[0]
    K[2,0], K[2,1], K[2,2] = -w[1], w[0], 0
    return K


@jit(nopython=True, nogil=True)
def w_to_R(w):
    # AKA: Rodrigues
    # Ref: https://en.wikipedia.org/wiki/Axis-angle_representation
    theta = norm3(w)
    if theta == 0:
        return np.eye(3)

    ax = w/theta
    K = make_skew(ax)  # Cross product matrix
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*dot(K,K)
    return R

@jit(nopython=True, nogil=True)
def R_to_w(R):
    trace = 0.0
    for i in range(3):
        trace += R[i,i]

    theta = np.arccos( (trace - 1)*0.5 )

    w = empty(3)
    w[0] = R[2,1] - R[1,2]
    w[1] = R[0,2] - R[2,0]
    w[2] = R[1,0] - R[0,1]
    w *= 1/(2*np.sin(theta))

    return w
