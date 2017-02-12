import numpy as np
from numpy import zeros, ones, empty, r_, c_, sqrt, dot
from numpy.linalg import norm

import skimage.io as imio
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.color import rgb2grey
from skimage.feature import canny

import os
import glob

from tools import IO, Image, Canny, Geom, Util

import time

import numba as nb
from numba import jit


@jit(nopython=True)
def min_touching(blob, ref, nblobs, max_val):
    ''' At coords in blob, find minimum touching value in im '''
    N = nblobs
    H = blob.shape[0]
    W = blob.shape[1]
    deltas = ((0,1),(-1,0),(0,-1),(1,0))

    min_val = empty(N, ref.dtype)
    min_val[:] = max_val

    for i in range(H):
        for j in range(W):
            blob_ind = blob[i, j]
            if blob_ind < 0:
                continue

            for di, dj in deltas:
                ni, nj = i+di, j+dj
                if ni >= 0 and nj >= 0 and ni < H and nj < W and ref[ni, nj] > 0:
                    min_val[blob_ind] = min(min_val[blob_ind], ref[ni, nj])

    return min_val


@jit(nopython=True, cache=False)
def _outlines(im, blob, blob_vals, thresh, values, flags):
    deltas = ((0,1),(-1,0),(0,-1),(1,0))

    H = im.shape[0]
    W = im.shape[1]
    for i in range(1, H-1):
        for j in range(1, W-1):
            if blob[i, j] >= 0:
                continue
            val = im[i, j]
            flag = 0
            for k in range(4):
                di, dj = deltas[k]
                ni, nj = i + di, j + dj
                blob_ind = blob[ni, nj]
                f = (im[ni, nj] >= val + thresh) or (blob_ind >= 0 and blob_vals[blob_ind] + 3*thresh >= val)
                flag += (1 << k)*f

            if flag > 0:
                flags[i, j] = flag
                values[i, j] = val

    return values, flags


def find_outlines(im, blob, blob_vals, thresh):
    values = np.zeros_like(im)
    flags = zeros(im.shape, np.uint8)

    return _outlines(im, blob, blob_vals, thresh, values, flags)


def find_contours(rgb, d, max_depth=2**16-1, snakes=False):
    holes = Image.find_blobs(d == 0)
    N = len(holes)

    hole_lookup = np.full(d.shape, -1, int)
    for i, hole in enumerate(holes):
        I, J = hole.T
        hole_lookup[I, J] = i

    max_val = np.iinfo(d.dtype).max
    blob_mins = min_touching(hole_lookup, d, N, max_val)

    thresh = 30
    outlines, flags = find_outlines(d, hole_lookup, blob_mins, thresh=thresh)

    contours, values, _ = Image.find_contours(d, thresh=100*thresh, min_len=10, values=outlines, flags=flags, dtype=float)

    if snakes:
        grey = rgb2grey(rgb)
        edges = 255*canny(grey, sigma=2).astype(np.uint8)
        edges = gaussian(edges, sigma=2)

    new_contours = []
    normals = []
    for i, cont in enumerate(contours):
        vals = values[i]
        if (vals > max_depth).any():
            continue
        pts = Geom.smooth_points(cont, sigma=3, support=5)
        if snakes:
            # pts += 2*Geom.smooth_contour_normals(pts, radius=4)
            pts = active_contour(edges, pts, alpha=0.1, beta=20, gamma=0.005, w_line=5, bc='fixed')

        # outlines has been conveniently emptied by find_contours
        coords = Image.draw_poly(outlines, np.round(pts).astype(int), val=vals)
        new_contours.append(coords)
        normals.append( Geom.smooth_contour_normals(coords, radius=4) )

    # show_d(outlines)

    return outlines, new_contours, normals


def find_contours_rgb(rgb, D, d_sigma=2, d_scale=1):
    from skimage.morphology import skeletonize
    # from skimage.filters import gaussian

    tim = Util.Timer()

    # grey = rgb2grey(rgb)
    grey = Image.rgb2grey(rgb)
    tim.add("Gray")
    edges_b = Image.find_edges(grey, sigma=2)
    tim.add("Find edges")

    edge_map = skeletonize(edges_b)
    tim.add("Skel")

    I, J = np.where(edge_map)
    N = len(I)

    P = empty((2,N), int)
    P[0] = J  # x
    P[1] = I  # y

    # Find normals based on depth
    # D_blur = gaussian(D, sigma=d_sigma)
    D_blur = D
    dgy, dgx = Image.gradient(D_blur)

    grey_blur = Image.gaussian(grey, sigma=2)
    gy, gx = Image.gradient(grey_blur)
    tim.add("Grad")

    normals = empty((2,N))
    normals[0] = gx[I, J]
    normals[1] = gy[I, J]
    normals /= norm(normals, axis=0)

    # Use depth map to make sure pointing correct way
    dnormals = zeros((2,N))
    dnormals[0] = dgx[I//d_scale, J//d_scale]
    dnormals[1] = dgy[I//d_scale, J//d_scale]
    mag = norm(dnormals, axis=0)
    sel = mag > 0
    dnormals[:, sel] /= mag[sel]

    flip = (normals * dnormals).sum(axis=0) > 0
    normals[:,flip] *= -1
    tim.add("The rest")
    # print(tim)

    return P.T, normals.T, edge_map


def find_contours_edge(E, D=None, low=25, high=50, depth_threshold=10, min_length=10):
    P, labels = Canny.double_threshold(E, low_threshold=low, high_threshold=high, min_length=min_length)

    return P, labels


@jit(nopython=True)
def estimate_normals(pts, labels, support=3):
    N = pts.shape[0]

    ns = empty((N,2), np.float32)
    curvs = empty(N, np.float32)

    for i in range(N):
        label = labels[i]
        s = max(0, i - support)
        e = min(N, i + support + 1)
        while labels[s] != label:
            s += 1
        while labels[e-1] != label:
            e -= 1

        P = pts[s:e]
        normal, curv = Geom.fit_normal_2d(P)
        ns[i] = normal
        curvs[i] = curv

    return ns, curvs


@jit(nopython=True)
def break_segments(pts, curvs, labels, max_breaks=10, max_curv=15e-3):
    '''
    Break segments (given by ordered labels) at points of high curvature.

    Note: A bit of magic in the choice of max_curv. It will depend on the
    support used when estimating the curvature. The discretization of pixesl
    means a smaller support will have more noise.

    In future, could use rate of change of normals along the line... Later.
    '''
    acurvs = np.abs(curvs)
    curv_smooth = Image.gaussian1d(acurvs, 2.0)
    peaks = Util.find_peaks(curv_smooth)
    corners = peaks[ np.where( acurvs[peaks] > max_curv)[0] ]

    newlabels = labels * max_breaks

    # Relabel
    N = len(newlabels)
    for j in corners:
        label = newlabels[j]
        newlabel = label + 1
        i = j
        while i < N and newlabels[i] == label:
            newlabels[i] = newlabel
            i += 1

    # Compress labels
    newlabel = 1
    i = 0
    while i < N:
        currentlabel = newlabels[i]
        j = i
        while j < N and newlabels[j] == currentlabel:
            newlabels[j] = newlabel
            j += 1

        i = j
        newlabel += 1

    return newlabels


@jit(nb.void(nb.int64[:,:], nb.bool_[:], nb.float32, nb.int64, nb.int64), nopython=True)
def _simp_helper(P, mask, eps, s, e):
    dmax = 0
    index = 0
    o = P[s]
    vx, vy = P[e] - P[s]
    mag = sqrt(vx*vx + vy*vy)
    nx, ny = -vy/mag, vx/mag

    for i in range(s + 1, e):
        dx, dy = P[i] - o
        d = abs(dx*nx + dy*ny)
        # d = abs( ((P[i] - o) * n).sum() )
        # d = abs( dot(P[i] - o, n) )
        if d > dmax:
            dmax = d
            index = i

    # print("Max:", index, dmax)

    if dmax > eps:
        _simp_helper(P, mask, eps, s, index)
        _simp_helper(P, mask, eps, index, e)
    else:
        mask[s+1:e] = False


@jit(nopython=True, nogil=True)
def simplify(P, eps=1.0):
    '''
    Simplify a polyline according to the Ramer–Douglas–Peucker algorithm
    ref: Wikipedia (https://goo.gl/YSFKgX)
    '''
    N = P.shape[0]
    mask = ones(N, nb.bool_)
    _simp_helper(P, mask, eps, 0, N-1)
    inds = np.where(mask)[0]
    return inds


@jit(nopython=True)
def simplify_labeled(P, labels, eps=1.0):
    N = P.shape[0]

    j = 0
    P_out = np.empty_like(P)
    labels_out = np.empty_like(labels)

    s = 0
    while s < N:
        label = labels[s]
        e = s
        while e < N and labels[e] == label:
            e += 1

        inds = simplify(P[s:e], eps)

        for i in inds + s:
            P_out[j] = P[i]
            labels_out[j] = label
            j += 1

        s = e

    return P_out[:j], labels_out[:j]


@jit(nopython=True)
def segment_tangents(P, labels, dtype=np.float32, thresh=25.0):
    '''
    Return the normal of each segment defined by sequential points in P.
    The last normal is undefined.
    eps: Maximum angle deflection in degrees allowed before a break
    '''
    N = P.shape[0]
    max_angle = np.deg2rad(thresh)
    min_dot = np.cos(max_angle)  # Save repeating arccos ops

    tangents = empty((N,2), dtype=dtype)

    for i in range(N - 1):
        tx, ty = P[i + 1] - P[i]
        mag = sqrt(tx*tx + ty*ty)
        tangents[i] = tx/mag, ty/mag  # Left rotate
    tangents[-1] = 1, 0

    newlabels = empty(N, dtype=np.int64)
    label = 1
    applied = 0
    for i in range(N - 1):
        if labels[i+1] != labels[i]:
            # Increment if the existing labels show a break, this segment is invalid
            newlabels[i] = 0
            label += applied  # Only increments if label has already been applied
            applied = 0
            continue

        # This segment is valid, label it up
        newlabels[i] = label
        applied = 1

        # Get the delta angle to the next segment
        if dot(tangents[i], tangents[i + 1]) < min_dot:
            # Deflection is large enough to classify a break
            # This segment is valid, but the next one must be newly labelled
            label += 1
            applied = 0

    newlabels[-1] = 0

    return tangents, newlabels


def test_contours():
    seq = 'data/rgbd/plant'
    out_d = os.path.join(seq, 'depth_out')

    try:
        os.makedirs(out_d)
    except FileExistsError:
        pass

    i = 0
    contour_set = []
    normal_set = []
    try:
        for im, D in load_datas(seq):
            print("Frame {}".format(i))
            # if i < 240:
            #     i += 1
            #     continue
            outlines, contours, normals = find_contours(im, D, max_depth=3000, snakes=False)
            imio.imsave(os.path.join(out_d, "{:04d}.png".format(i)), outlines.astype(np.uint16)*2**3)
            contour_set.append(contours)
            normal_set.append(normals)

            i += 1
    except Exception as e:
        print(e)
        raise

    np.savez(os.path.join(seq, 'contours.npz'), contours=contour_set, normals=normal_set)


def test_contours_rgb():
    import Edge_Tracker

    seq = 'data/synth/plant_out'
    out_d = os.path.join(seq, 'outlines_out')
    IO.imshow(zeros((3,3,3), np.uint8))

    try:
        os.makedirs(out_d)
    except FileExistsError:
        pass

    h, w = 360, 640
    cam = Geom.cam_params(f=29.9, sw=35.0, w=w, h=h)
    fx, fy, cx, cy = cam

    def to_homogenious(pts):
        N = pts.shape[0]
        out = empty((N,3))
        out[:,2] = 1
        out[:,0] = (pts[:,0] - cx)*(1/fx)
        out[:,1] = (pts[:,1] - cy)*(1/fy)
        return out

    def from_homogenious(pts):
        x = (fx*pts[:,0] + cx).astype(int)
        y = (fy*pts[:,1] + cy).astype(int)
        return x, y

    i = 0
    contour_set = []
    normal_set = []
    P_last = None
    pose = zeros(6)
    try:
        for im, D in load_datas(seq, imext='png'):
            print("Frame {}".format(i))
            P, normals, edge_map = find_contours_rgb(im, D)
            P_hom = to_homogenious(P)

            H, W = im.shape[:2]

            if P_last is not None:
                P_tr, pose, tree = Edge_Tracker.align(P_last, P_hom, pose=pose)

                P_tr = from_homogenious(P_tr)
                np.clip(P_tr[0], 0, W-1, out=P_tr[0])
                np.clip(P_tr[1], 0, H-1, out=P_tr[1])

                align = np.zeros_like(im)

                last_u, last_v = from_homogenious(P_last)

                align[P[:,1], P[:,0], 2] = 255
                align[last_v, last_u, 1] = 255
                align[P_tr[1], P_tr[0], 0] = 255

                IO.imshow(align, "align")
            # outlines, contours, normals = find_contours_rgb(im, D, snakes=False)
            # imio.imsave(os.path.join(out_d, "{:04d}.png".format(i)), outlines.astype(np.uint16)*2**3)
            # contour_set.append(contours)
            # normal_set.append(normals)
            # time.sleep(1)

            P_last = P_hom

            i += 1
    except Exception as e:
        print(e)
        raise

    np.savez(os.path.join(seq, 'contours.npz'), contours=contour_set, normals=normal_set)


if __name__ == '__main__':
    test_contours_rgb()
