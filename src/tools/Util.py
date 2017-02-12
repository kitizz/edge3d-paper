from __future__ import division, print_function
import numpy as np
from numpy import empty, zeros

from collections import OrderedDict

import pynutmeg
import skimage.io
import time
import struct
import itertools

import sys
import os
import glob

import threading
from numba import jit, autojit, guvectorize, bool_, int64, float64, vectorize

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


def in_bounds(x, y, w, h, border=0):
    return (x >= border) & (y >= border) & (x <= w - border) & (y <= h - border)


def next_power_of_2(n):
    '''
    Return next power of 2 greater than or equal to n
    http://stackoverflow.com/a/19164783
    '''
    return 2**int(n-1).bit_length()


def clip_vec(v, mag):
    l = np.linalg.norm(v)
    if l > mag:
        return v*(mag/l)
    else:
        return v


def lengths(lst):
    ''' Return an array whose values are the lengths of each element in lst '''
    return np.array( list(map(len, lst)), int )


def repeat_group(arr, sizes, repeat):
    ''' Repeat groups in arr by the amount defined by repeat. Grouped by lengths '''
    # Check the sizes are good
    sizes = np.array(sizes).reshape(-1)
    repeat = np.array(repeat).reshape(-1)

    L = len(sizes)
    if len(repeat) == 1:
        repeat = repeat.repeat(L)

    if L == 1:
        ''' Special case, all groups are same size '''
        result = arr.reshape(-1,sizes[0]).repeat(repeat, axis=0).ravel()
        return result

    L_cum = np.r_[0, sizes.cumsum()]

    if L_cum[-1] != len(arr):
        raise ValueError("Sizes does not properly span input array")
    if len(repeat) != L:
        raise ValueError("Repeat array must be scalar or same length as sizes")

    new_sizes = sizes * repeat
    N_cum = np.r_[0, new_sizes.cumsum()]
    result = np.empty(N_cum[-1], arr.dtype)

    for i in range(L):
        result[N_cum[i]: N_cum[i+1]] = np.tile(arr[L_cum[i]: L_cum[i+1]], repeat[i])

    return result


def round_int(v):
    '''
    Round and integerize the vector
    '''
    return np.round(v).astype(int)


# def find(v):
#     if v.ndim == 1:
#         return np.where(v)[0]
#     else:
#         return np.where(v)


# def find_peaks(v, as_indices=True, strict=True):
#     '''
#     :param binary: If False (default) return indices of peaks.
#     '''
#     b = np.empty(len(v), bool)
#     b[0], b[-1] = False, False
#     if strict:
#         b[1:-1] = (v[:-2] < v[1:-1]) & (v[1:-1] > v[2:])
#     else:
#         b[1:-1] = (v[:-2] <= v[1:-1]) & (v[1:-1] >= v[2:])
#     if as_indices:
#         return np.where(b)[0]
#     else:
#         return b


@jit(nopython=True, nogil=True)
def find_peaks(v, strict=True):
    '''
    :param binary: If False (default) return indices of peaks.
    '''
    N = len(v)
    res = np.empty(N, np.int64)
    j = 0
    if strict:
        for i in range(1, N-1):
            if (v[i-1] < v[i]) & (v[i] > v[i+1]):
                res[j] = i
                j += 1
    else:
        for i in range(1, N-1):
            if (v[i-1] <= v[i]) & (v[i] >= v[i+1]):
                res[j] = i
                j += 1

    # Save memory...
    cp = np.empty(j, np.int64)
    cp[:] = res[:j]

    return cp


@jit(nopython=True, cache=True)
def unwhere(inds, N=-1):
    if N < 0:
        N = inds.max()

    out = np.zeros(N, bool_)
    out[inds] = True

    return out


def random_pairs(N, radius=20, per_N=4):
    # Generate a set of sequential pairs. They consist of at least all neighbouring frames
    # Plus extras selected randomly
    pairs = [ [n+1, n] for n in range(N - 1) ]
    Nextra = min(N - 3, per_N)

    # Each frame gets Nextra extra pairings:
    np.random.seed(18316)
    for n in range(N):
        for c in range(Nextra):
            minv, maxv = np.clip([n - radius, n + radius], 0, N)
            while True:
                other_n = np.random.randint(minv, maxv)
                if abs(other_n - n) > 1:
                    # Got a good pair
                    break
            new_pair = [n, other_n]
            new_pair.sort()
            pairs.append(new_pair[::-1])

    pairs.sort()
    return np.array(pairs)


def resize(arr, shape, fill=None):
    brr = np.empty(shape, arr.dtype)
    if fill is not None:
        brr.fill(fill)

    # Copy arr over
    if arr.ndim == 1:
        brr[:arr.shape[0]] = arr

    else:
        rng = [np.s_[0:size] for size in arr.shape]
        brr[rng] = arr

    return brr


def flatten(lst):
    '''
    Flatten a list of lists
    '''
    return list(itertools.chain.from_iterable(lst))


class Timer(object):
    def __init__(self, clock=True):
        self.names = []
        self.times = []
        if clock:
            self.clock = time.clock
        else:
            self.clock = time.time
        self.times.append(self.clock())

    def add(self, name):
        self.times.append(self.clock())
        self.names.append(name)

    def to_str(self, tab=0):
        out = ""
        dt = np.diff(self.times)
        N = len(self.names)
        for i in range(N):
            out += '\t'*tab + "%s: %.1fms\n" % (self.names[i], dt[i]*1000)

        return out

    def __str__(self):
        return self.to_str()


class LoopTimer(object):
    def __init__(self, clock=True):
        self.times = OrderedDict()
        if clock:
            self.clock = time.clock
        else:
            self.clock = time.time

        self.t0 = zeros(256)

    def loop_start(self, ind):
        self.current = ind
        N = len(self.t0)
        while N <= ind:
            N *= 2
        if N != len(self.t0):
            t0 = zeros(N)
            t0[:len(self.t0)] = self.t0
            self.t0 = t0

        self.t0[ind] = time.clock()
        self.last_t = self.t0[ind]

    def add(self, name):
        if name not in self.times:
            N, dt_sum = 0, 0.0
        else:
            N, dt_sum = self.times[name]

        time = self.clock()
        N += 1
        dt_sum += time - self.last_t
        self.last_t = time

        self.times[name] = (N, dt_sum)

    def to_str(self, tab=0):
        out = ""
        for name in self.times:
            N, dt = self.times[name]
            out += '\t'*tab + "{}: {:.1f}ms\n".format(name, dt*1000/N)

        return out

    def __str__(self):
        return self.to_str()


class KeyboardPoller(threading.Thread):
    def start(self, key=None):
        self.key = key
        self.event = threading.Event()
        # self.stop = threading.Event()
        self.daemon = True

        threading.Thread.start(self)

    # def stop(self):
    #     self.stop.set()

    def read(self):
        pressed = self.event.is_set()
        self.event.clear()
        return pressed

    def run(self):
        while True:
            ch = sys.stdin.read(1)
            print("Received:", ch)
            if self.key is None:
                self.event.set()
            elif ch == self.key:
                self.event.set()


def grouped_count(lengths):
    # Over a total length of sum(lengths), this creates an array that counts up
    # from zero until each length is reached, then resets.
    return np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)


def vrange(starts, lengths):
    """ Create concatenated ranges of integers for multiple start/length
    Ref: [ http://codereview.stackexchange.com/q/83018 ]

    Args:
        starts (numpy.array): starts for each range
        lengths (numpy.array): lengths for each range (same length as starts)

    Returns:
        numpy.array: concatenated ranges

    See the following illustrative example:

        starts = np.array([1, 3, 4, 6])
        lengths = np.array([0, 2, 3, 0])

        print vrange(starts, lengths)
        >>> [3 4 4 5 6]

    """
    # Repeat start position index length times and concatenate
    cat_start = np.repeat(starts, lengths)

    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)

    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter

    return cat_range


def lex_order(arrs, maxs=None):
    if maxs is None:
        maxs = np.array([arr.max() + 1 for arr in arrs], float)

    v = np.zeros(len(arrs[0]))  # Assume all same length
    prod = 1
    for i in range(len(arrs) - 1, -1, -1):
        v += arrs[i] * prod
        prod *= maxs[i]

    # print(maxs)
    # print(v)

    return v.argsort()


@vectorize([float64(float64, float64, float64)], nopython=True)
def clip(v, a, b):
    return max(min(v, b), a)


@vectorize([int64(int64, int64, int64)], nopython=True)
def clip_int(v, a, b):
    return max(min(v, b), a)


@jit(nopython=True)
def place_and_sort(buf, i, val, order, j, N):
    buf[i] = val
    order[i] = j

    while i > 0 and val < buf[i - 1]:
        buf[i - 1], buf[i] = buf[i], buf[i - 1]
        order[i - 1], order[i] = order[i], order[i - 1]
        i -= 1

    while i < N - 1 and val > buf[i + 1]:
        buf[i + 1], buf[i] = buf[i], buf[i + 1]
        order[i + 1], order[i] = order[i], order[i + 1]
        i += 1


@jit(nopython=True)
def median_filter(v, size):
    if (size % 2) == 0 or size < 3:
        raise ValueError("Size must be odd and >= 3")

    N = len(v)
    buf = np.zeros(size, v.dtype)
    order = np.empty(size, dtype=np.int64)
    order[:] = -size
    out = np.zeros_like(v)

    radius = (size-1)//2
    for i in range(size - 1):
        j = i - radius
        place_and_sort(buf, i, v[abs(j)], order, j, i + 1)

    for i in range(0, N - radius):
        j = np.argmin(order)
        place_and_sort(buf, j, v[i + radius], order, i + radius, size)

        out[i] = buf[radius]

    for i in range(N - radius, N):
        j = np.argmin(order)
        place_and_sort(buf, j, v[2*N - (i + radius) - 2], order, i + radius, size)

        out[i] = buf[radius]

    return out


@jit(nopython=True, cache=True)
def cross3(a, b):
    out = np.empty(3, np.float64)
    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]
    return out


@jit(nopython=True, cache=True)
def norm3(a):
    x = a[0]
    y = a[1]
    z = a[2]
    return np.sqrt(x*x + y*y + z*z)


@jit(nopython=True, nogil=True)
def make_gaussian(sigma=3):
    N = int(sigma*6)
    N += 1 - (N % 2)

    x = np.linspace(-sigma*3, sigma*3, N)
    return np.exp(-x**2 / (2 * sigma**2))


@jit(nopython=True, nogil=True)
def convolve(a, b):
    # if len(b) > len(a):
    #     a, b = v, u
    # else:
    #     a, b = u, v

    N = len(a)
    M = len(b)
    res = np.zeros(N)

    s = (M-1)//2
    for i in range(N - M + 1):
        j = i + s
        sm = 0
        for m in range(M):
            sm += a[i + m] * b[m]
        res[j] = sm

    for j in range(s):
        i = j - s
        for m in range(M):
            res[j] += a[abs(i + m)] * b[m]

    for j in range(N - M + 1 + s, N):
        i = j - N + 1 - s
        for m in range(M):
            res[j] += a[N - 1 - abs(i + m)] * b[m]

    return res


@jit(nopython=True)
def histogram(a, bins=10, rng=(0,0)):
    if rng == (0,0):
        rng = a.min(), a.max() + 1e-12

    hist = zeros(bins, np.int64)

    idelta = bins / (rng[1] - rng[0])
    b = (a - rng[0])*idelta

    N = a.shape[0]
    for i in range(N):
        j = int(b[i])
        if j >= 0 and j < bins:
            hist[ j ] += 1

    histx = np.arange(bins + 1)/idelta + rng[0]

    return hist, histx


@jit(nopython=True)
def histogram2d(x, y, bins=(0,0), res=(0,0), rangex=(0,0), rangey=(0,0), sel=empty(0,np.int64)):
    if rangex == (0,0):
        rangex = x.min(), x.max() + 1e-12
    if rangey == (0,0):
        rangey = y.min(), y.max() + 1e-12

    if res == (0,0):
        # No resolution specified, use number of bins
        if bins == (0,0):
            bins = 10, 10
        idx = bins[0] / (rangex[1] - rangex[0])
        idy = bins[1] / (rangey[1] - rangey[0])

    else:
        # Figure out the right number of bins for this resolution
        idx = 1/res[0]
        idy = 1/res[1]
        bx = np.ceil(idx * (rangex[1] - rangex[0]))
        by = np.ceil(idy * (rangey[1] - rangey[0]))
        bins = int(bx), int(by)

    N = x.shape[0]
    if len(sel) == 0:
        sel = np.arange(0, N, 1, np.int64)

    result = zeros(bins, np.int64)

    # xscaled = (x - rangex[0])*idx
    # yscaled = (y - rangey[0])*idy

    for ind in sel:
        i = int( (x[ind] - rangex[0])*idx )
        j = int( (y[ind] - rangey[0])*idy )
        # i, j = int(xscaled[ind]), int(yscaled[ind])
        if i < 0 or j < 0 or i >= bins[0] or j >= bins[1]:
            continue
        result[i, j] += 1

    histx = np.arange(bins[0] + 1)/idx + rangex[0]
    histy = np.arange(bins[1] + 1)/idy + rangey[0]

    return result, histx, histy


def min_category(v, labels, max_val):
    '''
    Return the indices of the smallest values of v within their category.
    Assumes values are sorted by labels.
    '''
    N = len(v)
    inds = empty(N, np.int64)

    label = labels[0]
    inds[0] = 0
    best = v[0]
    j = 0

    for i in range(1, N):
        if v[i] > max_val:
            continue

        if labels[i] != label:
            label = labels[i]
            j += 1
            inds[j] = i
            best = v[i]

        elif v[i] < best:
            inds[j] = i
            best = v[i]

    return inds[:j+1]


def cluster_category(hist, histx, v, labels, thresh=0.1, max_delta=0.1):
    H = len(hist)
    best = hist.argmax()

    # Search back and forward to find start and end of cluster
    vcenter = 0.5 * (histx[best + 1] + histx[best])
    hthresh = thresh*hist[best]
    s = best
    while s > 0 and histx[s] > vcenter - max_delta and hist[s] > hthresh:
        s -= 1
    e = best
    while e < H - 1 and histx[e] < vcenter + max_delta and hist[e] > hthresh:
        e += 1

    vmin = histx[s]
    vmax = histx[e]

    # Find unique elements according to labels that are closest to vcenter
    # and are within the [min, max]
    N = len(v)
    inds = empty(N, np.int64)
    inds[0] = 0
    label = labels[0]
    best_v = abs(v[0] - vcenter)
    j = 0

    for i in range(1, N):
        if v[i] < vmin or v[i] > vmax:
            continue

        voff = abs(v[i] - vcenter)

        if labels[i] != label:
            label = labels[i]
            j += 1
            inds[j] = i
            best_v = voff

        elif voff < best_v:
            inds[j] = i
            best_v = voff

    return inds[:j+1]


# @jit(nopython=True)
def histogram_category(a, labels, bins=10, rng=(0,0), weights=1):
    '''
    Count the number of unique labels in each bin
    Assume labels are ordered/clustered
    '''
    if rng == (0,0):
        rng = a.min(), a.max() + 1e-12

    delta = (rng[1] - rng[0])/bins
    b = (a - rng[0])/delta

    N = a.shape[0]
    w = empty(N)
    w[:] = weights

    # Create a map to make it quick to look up the label bin
    # This helps save memory by not having to alocate a (bins x maxlabel)
    # matrix, but instead a (bin x nlabels).
    nlabels = 0
    labelmap = np.full(labels.max() + 1, -1, np.int64)

    for i in range(N):
        label = labels[i]
        if labelmap[label] < 0:
            labelmap[label] = nlabels
            nlabels += 1

    H = zeros((bins, nlabels))
    label = labels[0] - 1

    for i in range(N):
        col = labelmap[ labels[i] ]
        row = int(b[i])
        if row >= 0 and row < bins:
            H[row, col] = max(H[row, col], w[i])

    result = zeros(bins, H.dtype)
    for i in range(bins):
        for j in range(nlabels):
            result[i] += H[i,j]

    return result, np.linspace(rng[0], rng[1], bins + 1)

@jit(nopython=True)
def mean_shift_1d(x, x0, dx, its=3):
    # Recenter the mean until convergence
    for k in range(its):
        inside = np.where( (np.abs(x - x0) < dx) )[0]
        if len(inside) == 0:
            break
        x0 = np.mean( x[inside] )

    return x0, inside


@jit(nopython=True)
def mean_shift_2d(x, y, x0, y0, dx, dy, its=3):
    # Recenter the mean until convergence
    for k in range(its):
        inside = np.where( (np.abs(x - x0) < dx) & (np.abs(y - y0) < dy) )[0]
        if len(inside) == 0:
            break
        x0 = np.mean( x[inside] )
        y0 = np.mean( y[inside] )

    return x0, y0, inside


def try_mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


@jit(nopython=True)
def find(b):
    N = len(b)
    res = empty(N, np.int64)

    j = 0
    for i in range(N):
        res[j] = i
        j += b[i]

    return res[:j]


@jit(nopython=True)
def unique(arr, inplace=True):
    if not inplace:
        arr = arr.copy()

    arr.sort()
    N = len(arr)

    out = np.empty_like(arr)

    last = arr[0]
    out[0] = last
    j = 1
    for i in range(1, N):
        v = arr[i]
        if v != last:
            out[j] = v
            last = v
            j += 1

    return out[:j]


@jit(nopython=True)
def unique2(arr, inplace=True):
    if not inplace:
        arr = arr.copy()

    # arr.sort()
    N = len(arr)

    # out = np.empty_like(arr)

    last = arr[0]
    # out[0] = last
    j = 1
    for i in range(1, N):
        v = arr[i]
        if v != last:
            # if i != j:
            arr[j] = v
            last = v
            j += 1

    return arr[:j]


@jit(nopython=True)
def _partition(A, inds, lo, hi):
    k = np.random.randint(lo, hi+1)
    pivot = A[k]
    i = lo - 1
    j = hi + 1
    while True:
        i += 1
        while A[i] < pivot:
            i += 1

        j -= 1
        while A[j] > pivot:
            j -= 1
        
        if i >= j:
            return j
        
        A[i], A[j] = A[j], A[i]
        inds[i], inds[j] = inds[j], inds[i]


@jit(nopython=True)
def _quicksort(A, inds, lo, hi):
    if lo < hi:
        p = _partition(A, inds, lo, hi)
        _quicksort(A, inds, lo, p)
        _quicksort(A, inds, p + 1, hi)


@jit(nopython=True)
def argsort(v):
    v = v.copy()
    inds = np.arange(0, len(v), 1, np.int64)
    _quicksort(v, inds, 0, len(v) - 1)
    return inds


@jit(nopython=True)
def partition(values, idxs, left, right):
    """
    Partition method
    """

    piv = values[idxs[left]]
    i = left + 1
    j = right

    while True:
        while i <= j and values[idxs[i]] <= piv:
            i += 1
        while j >= i and values[idxs[j]] >= piv:
            j -= 1
        if j <= i:
            break

        idxs[i], idxs[j] = idxs[j], idxs[i]

    idxs[left], idxs[j] = idxs[j], idxs[left]

    return j


@jit(nopython=True)
def argsort1D(values):

    idxs = np.arange(values.shape[0])

    left = 0
    right = values.shape[0] - 1

    max_depth = np.int(right / 2)

    ndx = 0

    tmp = np.zeros((max_depth, 2), dtype=np.int64)

    tmp[ndx, 0] = left
    tmp[ndx, 1] = right

    ndx = 1
    while ndx > 0:

        ndx -= 1
        right = tmp[ndx, 1]
        left = tmp[ndx, 0]

        piv = partition(values, idxs, left, right)

        if piv - 1 > left:
            tmp[ndx, 0] = left
            tmp[ndx, 1] = piv - 1
            ndx += 1

        if piv + 1 < right:
            tmp[ndx, 0] = piv + 1
            tmp[ndx, 1] = right
            ndx += 1

    return idxs


@jit(nopython=True)
def argmax2d(mat):
    peak = mat.argmax()
    return peak//mat.shape[1], peak % mat.shape[1]


def test_poller():
    poller = KeyboardPoller(daemon=True)
    poller.start()

    print("Sleeping")
    time.sleep(5)

    print("Reading:")
    print(poller.read())


if __name__ == '__main__':
    # test_interp()
    # test_imshow_nutmeg()

    test_poller()
