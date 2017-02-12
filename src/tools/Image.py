import numpy as np
from numpy import clip, floor, zeros, empty, r_, sqrt

import numba
from numba import jit

from skimage import feature
from skimage import color

from . import Util, Canny, IO

import pynutmeg
import time


def linear_interpolate(im, x):
    x = np.asarray(x)

    x0 = floor(x).astype(int)
    x1 = x0 + 1

    x0 = clip(x0, 0, im.shape[0]-1)
    x1 = clip(x1, 0, im.shape[0]-1)

    Ia = im[ x0 ]
    Ib = im[ x1 ]

    wa = (x1-x)
    wb = (x-x0)

    return (wa*Ia.T + wb*Ib.T).T


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = floor(x).astype(int)
    x1 = x0 + 1
    y0 = floor(y).astype(int)
    y1 = y0 + 1

    x0 = clip(x0, 0, im.shape[1]-1)
    x1 = clip(x1, 0, im.shape[1]-1)
    y0 = clip(y0, 0, im.shape[0]-1)
    y1 = clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ].T
    Ib = im[ y1, x0 ].T
    Ic = im[ y0, x1 ].T
    Id = im[ y1, x1 ].T

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


@jit(nopython=True, nogil=True)
def bilinear_interpolate_numba(im, x, y):
    N = len(x)

    x0_ = empty(N, np.int64)
    y0_ = empty(N, np.int64)
    for i in range(N):
        x0_[i] = floor(x[i])
        y0_[i] = floor(y[i])
    x1_ = x0_ + 1
    y1_ = y0_ + 1

    x0 = Util.clip_int(x0_, 0, im.shape[1]-1)
    x1 = Util.clip_int(x1_, 0, im.shape[1]-1)
    y0 = Util.clip_int(y0_, 0, im.shape[0]-1)
    y1 = Util.clip_int(y1_, 0, im.shape[0]-1)

    Ia = empty(N)
    Ib = empty(N)
    Ic = empty(N)
    Id = empty(N)
    for i in range(N):
        Ia[i] = im[ y0[i], x0[i] ]
        Ib[i] = im[ y1[i], x0[i] ]
        Ic[i] = im[ y0[i], x1[i] ]
        Id[i] = im[ y1[i], x1[i] ]

    # Ia = im[ y0, x0 ].T
    # Ib = im[ y1, x0 ].T
    # Ic = im[ y0, x1 ].T
    # Id = im[ y1, x1 ].T

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def bilinear_interpolate_weights(sz, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = floor(x).astype(int)
    x1 = x0 + 1
    y0 = floor(y).astype(int)
    y1 = y0 + 1

    x0 = clip(x0, 0, sz[1]-1)
    x1 = clip(x1, 0, sz[1]-1)
    y0 = clip(y0, 0, sz[0]-1)
    y1 = clip(y1, 0, sz[0]-1)

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return x0, x1, y0, y1, wa, wb, wc, wd


def bilinear_interpolate_par(args):
    im, x0, x1, y0, y1, wa, wb, wc, wd = args

    Ia = im[ y0, x0 ].T
    Ib = im[ y1, x0 ].T
    Ic = im[ y0, x1 ].T
    Id = im[ y1, x1 ].T
    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def bilinear_interpolate_mult(ims, x, y):
    '''
    Assumes all same shape
    '''
    h, w = ims[0].shape
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = floor(x).astype(int)
    x1 = x0 + 1
    y0 = floor(y).astype(int)
    y1 = y0 + 1

    x0 = clip(x0, 0, w-1)
    x1 = clip(x1, 0, w-1)
    y0 = clip(y0, 0, h-1)
    y1 = clip(y1, 0, h-1)

    # Ia = im[ y0, x0 ]
    # Ib = im[ y1, x0 ]
    # Ic = im[ y0, x1 ]
    # Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return [ wa*I[y0, x0] + wb*I[y1, x0] + wc*I[y0, x1] + wd*I[y1, x1]
             for I in ims ]


@jit(nopython=True, nogil=True)
def rgb2grey(rgb):
    # 0.21 R + 0.72 G + 0.07 B.
    H = rgb.shape[0]
    W = rgb.shape[1]
    grey = empty((H,W))

    r = 0.21/255
    g = 0.72/255
    b = 0.07/255

    for i in range(H):
        for j in range(W):
            col = rgb[i,j]
            grey[i,j] = r*col[0] + g*col[1] + b*col[2]

    return grey


@jit(nopython=True, nogil=True)
def gradient(im):
    H = im.shape[0]
    W = im.shape[1]

    gx = empty((H,W))
    gy = empty((H,W))

    for i in range(H):
        gx[i,0] = im[i,1] - im[i,0]
        for j in range(1,W-1):
            gx[i,j] = 0.5*(im[i,j+1] - im[i,j-1])
        gx[i,W-1] = im[i,W-1] - im[i,W-2]

    # Border...
    for j in range(W):
        gy[0,j] = im[1,j] - im[0,j]
        gy[H-1,j] = im[H-1,j] - im[H-2,j]

    for i in range(1,H-1):
        for j in range(W):
            gy[i,j] = 0.5*(im[i+1,j] - im[i-1,j])

    return gy, gx


@jit(nopython=True, nogil=True)
def gauss_boxes(sigma, n):
    wIdeal = sqrt((12*sigma*sigma/n) + 1)  # Ideal averaging filter width
    wl = np.floor(wIdeal)
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2

    mIdeal = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4)
    m = np.round(mIdeal)
    # sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 )
    sizes = empty(n, np.int64)
    for i in range(n):
        if i < m:
            sizes[i] = wl
        else:
            sizes[i] = wu
    return sizes


@jit(nopython=True, nogil=True)
def gaussian1d(v, sigma):
    u = v.reshape(1,-1)

    dst = np.empty_like(u)
    src = u.copy()

    bxs = gauss_boxes(sigma, 3)
    box_blur_h(src, dst, (bxs[0]-1)//2)
    box_blur_h(dst, src, (bxs[1]-1)//2)
    box_blur_h(src, dst, (bxs[2]-1)//2)
    return dst[0]


@jit(nopython=True, nogil=True)
def gaussian(grey, sigma):
    H = grey.shape[0]
    W = grey.shape[1]

    tmp = empty((H,W))
    dst = grey.copy()

    bxs = gauss_boxes(sigma, 3)
    box_blur(dst, tmp, (bxs[0]-1)//2)
    box_blur(dst, tmp, (bxs[1]-1)//2)
    box_blur(dst, tmp, (bxs[2]-1)//2)

    return dst


@jit(nopython=True, nogil=True)
def box_blur(src, tmp, r):
    box_blur_h(src, tmp, r)
    box_blur_v(tmp, src, r)


@jit(nopython=True, nogil=True)
def box_blur_h(src, dst, r):
    h = src.shape[0]
    w = src.shape[1]

    iarr = 1 / (r+r+1)
    # val = 0

    for i in range(h):
        ssrc = src[i]
        sdst = dst[i]

        val = r * ssrc[0]
        for k in range(r+1):
            val += ssrc[k]

        for j in range(r):
            sdst[j] = val * iarr
            val = val - ssrc[0] + ssrc[j + r + 1]

        for j in range(r, w-r-1):
            sdst[j] = val * iarr
            val = val - ssrc[j - r] + ssrc[j + r + 1]

        for j in range(w-r-1, w):
            sdst[j] = val * iarr
            val = val - ssrc[j - r] + ssrc[w-1]


@jit(nopython=True, nogil=True)
def box_blur_v(src, dst, r):
    h = src.shape[0]

    iarr = 1 / (r+r+1)

    val = r * src[0]
    for k in range(r+1):
        val += src[k]

    for j in range(r):
        dst[j] = val * iarr
        val = val - src[0] + src[j + r + 1]

    for j in range(r, h-r-1):
        dst[j] = val * iarr
        val = val - src[j - r] + src[j + r + 1]

    for j in range(h-r-1, h):
        dst[j] = val * iarr
        val = val - src[j - r] + src[h-1]


@jit(nopython=True, cache=True)
def _fc(im, thresh, flags, values):
    H = im.shape[0]
    W = im.shape[1]
    for i in range(1, H-1):
        for j in range(1, W-1):
            val = im[i, j] + thresh
            flag = (
                1*(im[i, j+1] >= val) +
                2*(im[i-1, j] >= val) +
                4*(im[i, j-1] >= val) +
                8*(im[i+1, j] >= val)
                )

            if flag > 0:
                flags[i, j] = flag
                values[i, j] = val - thresh


def find_outlines(im, thresh, values=None, flags=None):
    if values is None:
        values = np.zeros_like(im)
    if flags is None:
        flags = zeros(im.shape, np.uint8)

    _fc(im, thresh, flags, values)

    return values, flags


@jit(nopython=True, cache=True)
def find_next(im, i, j):
    H = im.shape[0]
    W = im.shape[1]

    while i < H:
        while j < W:
            if im[i, j] != 0:
                return i, j
            j += 1
        i += 1
        j = 0

    return -1, -1


@jit(nopython=True, cache=True)
def find_next_node(im, i, j):
    H = im.shape[0]
    W = im.shape[1]

    while i < H:
        while j < W:
            if im[i, j] != 0:
                connected = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni = i + di
                        nj = j + dj
                        if ni >= 0 and nj >= 0 and ni < H and nj < W and im[ni, nj]:
                            connected += 1
                if connected > 1 and connected != 3:
                    # Including center, 3 means middle of segment
                    return i, j
                if connected == 1:
                    im[i, j] = 0

            j += 1
        i += 1
        j = 0

    return -1, -1


@jit(nopython=True, cache=True)
def follow_contour(values, flags, thresh, i, j, connect=8):
    # Follow the outlines from find_outlines using flags as an indication of its direction
    # Join all the outside corner points (flags = [3, 6, 12, 9])
    H = values.shape[0]
    W = values.shape[1]
    I = []
    J = []
    vals = []
    si, sj = i, j
    sval = values[i, j]
    atstart = True

    # Note these are row, col deltas
    # if connect == 8:
    deltas = ((0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1))

    while True:
        # First go ccw
        # Check if this is a corner
        val = values[i, j]
        values[i, j] = 0
        flag = flags[i, j]

        I.append(i)
        J.append(j)
        vals.append(val)

        if flag & 4:
            d = 5
        elif flag & 8:
            d = 7
        elif flag & 1:
            d = 1
        elif flag & 2:
            d = 3

        # Consume this value and move on
        atend = True
        for k in range(8):
            m = (d + k) % 8
            # if connect == 4 and (m % 2) != 0:
                # continue
            di, dj = deltas[m]
            ni, nj = i+di, j+dj
            if ni < 0 or nj < 0 or ni >= H or nj >= W:
                continue
            nval = values[ni, nj]
            if nval != 0 and abs(nval - val) <= thresh:
                if atstart:
                    next_flag = flags[ni,nj]
                    if (next_flag & 1) or \
                            ((m > 5 or m == 0) and next_flag & 2) or \
                            (m == 5 and (flag & 8 or next_flag & 8)) or \
                            (m == 6 and flag & 1 and next_flag & 8):
                        # We're at an endpoint, and should be going CW
                        break
                i = ni
                j = nj
                atend = False
                break

        atstart = False
        if atend:
            break  # End of contour

    atstart = True
    i, j = si, sj
    while True:
        # Now go cw
        if atstart:
            # We've gone back to the start, ignore this one first
            val = sval
            atstart = False

        else:
            val = values[i, j]
            values[i, j] = 0
            I.insert(0, i)
            J.insert(0, j)
            vals.insert(0, val)

        flag = flags[i, j]

        if flag & 1:
            d = 7
        elif flag & 2:
            d = 1
        elif flag & 4:
            d = 3
        elif flag & 8:
            d = 5

        # Consume this value and move on
        for k in range(8):
            m = (d - k) % 8
            # if connect == 4 and (m % 2) != 0:
                # continue
            di, dj = deltas[m]
            ni, nj = i+di, j+dj
            if ni < 0 or nj < 0 or ni >= H or nj >= W:
                continue
            nval = values[ni, nj]
            if nval != 0 and abs(nval - val) <= thresh:
                i = ni
                j = nj
                break
        else:
            break  # End of contour

    return I, J, vals


def find_contours(im, thresh, min_len=5, values=None, flags=None, dtype=int):
    if values is None:
        values, flags = find_outlines(im, thresh)

    i, j = find_next(values, 0, 0)

    pt_set = []
    val_set = []
    flag_set = []
    means = []

    while i >= 0 and j >= 0:
        I, J, vals = follow_contour(values, flags, thresh, i, j)
        N = len(I)

        i, j = find_next(values, i, j)
        if N < min_len:
            continue

        pts = empty((N,2),dtype)
        pts[:,0] = J
        pts[:,1] = I
        vals = np.array(vals)

        pt_set.append(pts)
        val_set.append(vals)
        flag_set.append(flags[I,J].copy())
        means.append(vals.mean())

    order = np.argsort(means)
    pt_set = np.array(pt_set)[order]
    val_set = np.array(val_set)[order]
    flag_set = np.array(flag_set)[order]

    return pt_set, val_set, flag_set


@jit(nopython=True, cache=True)
def align(p, v):
    '''
    Align smaller signal v in bigger signal p
    '''
    N = len(p)
    M = len(v)

    score = empty(N-M)
    for i in range(N - M):
        score[i] = np.abs(p[i:i+M] - v).sum()

    return score


@jit(nopython=True, cache=True)
def resample(arr, interval):
    '''
    arr: Boolean array
    interval: Interval to sample at
    '''
    res = []
    N = len(arr)
    if N < interval:
        return res

    run = interval - 1
    minrun = 2*interval - 1

    for i in range(N + interval - 1):
        if i >= N or not arr[i]:
            run += 1
        else:
            run = 0
        if run >= minrun:
            res.append(i - interval + 1)
            run = interval - 1

    return res


@jit(nopython=True, cache=True)
def consume_blob(b, i, j, deltas):
    I = [i]
    J = [j]
    H = b.shape[0]
    W = b.shape[1]
    n = 1
    to_explore = [0]
    b[i, j] = False

    while len(to_explore) > 0:
        ind = to_explore.pop(-1)
        i, j = I[ind], J[ind]

        # for di, dj in deltas:
        for k in range(len(deltas)):
            di = deltas[k,0]
            dj = deltas[k,1]
            ni, nj = i+di, j+dj
            if ni >= 0 and nj >= 0 and ni < H and nj < W and b[ni, nj]:
                to_explore.append(n)
                b[ni, nj] = False
                I.append(ni)
                J.append(nj)
                n += 1

    return I, J

# fig = pynutmeg.figure('blob', 'figs/segments.qml')
# fig.set_gui('figs/segments_gui.qml')
# fig.set('ax.im', xOffset=-0.5, yOffset=-0.5)
# # nextframe = fig.parameter('nextframe')
# # nextframe.wait_changed()
# nnxt = fig.parameter('nnext')
# nxt = fig.parameter('next')


@jit(nopython=True, cache=True)
def consume_segment(b, i, j, deltas, latest_label):
    I = [i]
    J = [j]
    labels = [latest_label]
    H = b.shape[0]
    W = b.shape[1]
    n = 1
    to_explore = [0]
    b[i, j] = False

    # fig.set('ax.im', binary=b.astype(np.uint8) - 1)
    # skip = 0

    while len(to_explore) > 0:
        ind = to_explore.pop(-1)
        i, j = I[ind], J[ind]
        current_label = labels[ind]

        branches = 0
        # for di, dj in deltas:
        for k in range(len(deltas)):
            di = deltas[k,0]
            dj = deltas[k,1]
            ni, nj = i+di, j+dj
            if ni >= 0 and nj >= 0 and ni < H and nj < W and b[ni, nj]:
                to_explore.append(n)
                b[ni, nj] = False
                I.append(ni)
                J.append(nj)
                branches += 1
                n += 1

        # skip -= 1
        # if skip < 0:
        #     fig.set('ax.P0', x=[int(j)], y=[int(i)])
        #     print("Branches:", branches)
        #     if branches > 0:
        #         print("I:", I[-branches:])
        #         print("J:", J[-branches:])
        #         x = np.array(J[-branches:], float)
        #         y = np.array(I[-branches:], float)
        #         fig.set('ax.P1', x=x, y=y)

        #     while True:
        #         if nxt.read_changed():
        #             skip = 0
        #             break
        #         if nnxt.read_changed():
        #             skip = 20
        #             break
        #         time.sleep(0.002)

        if branches > 1:
            # Create new labels if at a node
            for _ in range(branches):
                latest_label += 1
                labels.append(latest_label)

        elif branches == 1:
            # Keep going with current label otherwise
            labels.append(current_label)

    return I, J, labels, latest_label + 1


# @jit(nopython=True, cache=False)
def consume_blob_depth(b, D, i, j, deltas, depth_thresh, scale):
    I = [i]
    J = [j]
    H = b.shape[0]
    W = b.shape[1]
    n = 1
    to_explore = [0]
    b[i, j] = False

    while len(to_explore) > 0:
        ind = to_explore.pop(-1)
        i, j = I[ind], J[ind]
        d = D[i//scale, j//scale]

        # for di, dj in deltas:
        for k in range(len(deltas)):
            di = deltas[k,0]
            dj = deltas[k,1]
            ni, nj = i+di, j+dj
            if ni >= 0 and nj >= 0 and ni < H and nj < W and b[ni, nj] and abs(D[ni//scale, nj//scale] - d) < depth_thresh:
                to_explore.append(n)
                b[ni, nj] = False
                I.append(ni)
                J.append(nj)
                n += 1

    return I, J


def find_blobs(b, connectivity=4):
    ''' Find blobs in binary image '''
    i, j = find_next(b, 0, 0)
    blobs = []

    deltas = np.array( ((0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)) )
    if connectivity == 4:
        deltas = deltas[::2]
    # elif connectivity == 8:

    while i >= 0 and j >= 0:
        I, J = consume_blob(b, i, j, deltas)
        sz = len(I)

        coords = empty((sz,2), int)
        coords[:,0] = I
        coords[:,1] = J

        blobs.append(coords)

        i, j = find_next(b, i, j)

    return blobs


def find_segments(b, connectivity=4, depth_thresh=0.1, scale=4):
    ''' Find blobs in binary image '''
    i, j = find_next_node(b, 0, 0)
    N = 1024
    blobs = np.empty((N,2), int)
    labels = np.empty(N, int)
    starts = []
    sizes = []
    n = 0

    def new_blobs(n):
        M = N
        while M < n:
            M *= 2
        tmp_blobs = np.empty((M,2), int)
        tmp_blobs[:N] = blobs
        tmp_lab = np.empty(M, int)
        tmp_lab[:N] = labels
        return tmp_blobs, tmp_lab, M

    deltas = np.array( ((0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)) )
    if connectivity == 4:
        deltas = deltas[::2]
    # elif connectivity == 8:

    current_label = 1

    biggest = 1

    # skip = False

    node_run = True

    while i >= 0 and j >= 0:
        I, J, labs, current_label = consume_segment(b, i, j, deltas, current_label)
        # if not skip:
        #     skip = IO.imshow(b.astype(np.uint8)*255, wait=-1)

        sz = len(I)
        biggest = max(sz, biggest)

        new_n = n + sz
        if new_n > N:
            blobs, labels, N = new_blobs(new_n)

        blobs[n:new_n,0] = I
        blobs[n:new_n,1] = J
        labels[n:new_n] = labs
        n = new_n

        if node_run:
            i, j = find_next_node(b, i, j)
            if i < 0 or j < 0:
                node_run = False
                i, j = find_next(b, 0, 0)
        else:
            i, j = find_next(b, i, j)

    # To get a lexical ordering based on label and original order,
    # we multiply labels by the biggest possible segment size
    values = labels[:n]*biggest + r_[0:n]
    ind_sorted = np.argsort(values)
    labels = labels[ind_sorted]

    # Chunk the data
    boundaries = np.where(np.diff(labels) != 0)[0]
    starts = r_[0, boundaries + 1]
    ends = r_[boundaries + 1, n]
    sizes = ends - starts

    return blobs[ind_sorted], starts, sizes


def find_edges(grey, sigma=3):
    # grey = color.rgb2grey(im)
    # return feature.canny(grey, sigma)
    return Canny.canny(grey, sigma)
