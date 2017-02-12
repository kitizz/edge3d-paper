import numpy as np
from numpy import sqrt, empty, zeros

import numba as numba
from numba import jit

from . import Image, IO

from skimage.morphology import skeletonize


@jit(nopython=True, nogil=True)
def sobel_filt(im):
    H = im.shape[0]
    W = im.shape[1]

    buf = empty((H,W))
    out = empty((H,W))

    # Pre buffer grad
    for i in range(H):
        scan = im[i]
        for j in range(1,W-1):
            buf[i,j] = (scan[j+1] - scan[j-1])
        buf[i,0] = buf[i,1]
        buf[i,-1] = buf[i,-2]

    out[0] = 2*buf[0] + 2*buf[1]

    for i in range(1, H-1):
        for j in range(W):
            out[i,j] = buf[i-1,j] + 2*buf[i,j] + buf[i+1,j]

    out[-1] = 2*buf[-2] + 2*buf[-1]

    return out


@jit(nopython=True, nogil=True)
def grad(im, sigma):
    H = im.shape[0]
    W = im.shape[1]

    smoothed = Image.gaussian(im, sigma)
    # smoothed = ndi.gaussian_filter(im, sigma, mode='constant')
    # imshow(smoothed, 'smth')

    gx = sobel_filt(smoothed)
    gy = sobel_filt(smoothed.T).T
    # imshow(gy, 'isob')

    mag = empty((H,W))
    for i in range(H):
        for j in range(W):
            y, x = gy[i,j], gx[i,j]
            mag[i,j] = sqrt(x*x + y*y)

    return gx, gy, mag


@jit(nopython=True, nogil=True)
def non_local_suppression(mag, gx, gy):
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    H = mag.shape[0]
    W = mag.shape[1]

    bool = numba.bool_
    local_maxima = np.empty((H,W), bool)
    for i in range(1,H-1):
        for j in range(1,W-1):
            dy = gy[i,j]
            dx = gx[i,j]
            grad = mag[i,j]

            ady = abs(dy)
            adx = abs(dx)

            if ady == 0 and adx == 0:
                local_maxima[i,j] = False
                # continue

            elif ady < adx:
                if dx * dy > 0:
                    # 0 - 45
                    w = ady / adx
                    m1 = (1 - w)*mag[i,j+1] + (w)*mag[i+1,j+1]
                    m2 = (1 - w)*mag[i,j-1] + (w)*mag[i-1,j-1]
                    local_maxima[i,j] = grad >= max(m1, m2)

                else:
                    # 135 - 180
                    w = ady / adx
                    m1 = (1 - w)*mag[i,j-1] + (w)*mag[i+1,j-1]
                    m2 = (1 - w)*mag[i,j+1] + (w)*mag[i-1,j+1]
                    local_maxima[i,j] = grad >= max(m1, m2)

            else:
                if dx * dy > 0:
                    # 45 - 90
                    w = adx / ady
                    m1 = (1 - w)*mag[i+1,j] + (w)*mag[i+1,j+1]
                    m2 = (1 - w)*mag[i-1,j] + (w)*mag[i-1,j-1]
                    local_maxima[i,j] = grad >= max(m1, m2)

                else:
                    # 90 - 135
                    w = adx / ady
                    m1 = (1 - w)*mag[i+1,j] + (w)*mag[i+1,j-1]
                    m2 = (1 - w)*mag[i-1,j] + (w)*mag[i-1,j+1]
                    local_maxima[i,j] = grad >= max(m1, m2)

    return local_maxima


@jit(nopython=True, nogil=True)
def threshold(local_maxima, mag, low, high):
    #---- Create two masks at the two thresholds.
    H = mag.shape[0]
    W = mag.shape[1]
    bool_ = numba.bool_
    high_mask = empty((H,W), bool_)
    low_mask = empty((H,W), bool_)

    for i in range(H):
        for j in range(W):
            mx = local_maxima[i,j]
            m = mag[i,j]
            high_mask[i,j] = mx and (m > high)
            low_mask[i,j] = mx and (m > low)

    return low_mask, high_mask


@jit(nopython=True, nogil=True)
def mag_threshold(mag, low, high):
    #---- Create two masks at the two thresholds.
    H = mag.shape[0]
    W = mag.shape[1]
    bool_ = numba.bool_
    high_mask = empty((H,W), bool_)
    low_mask = empty((H,W), bool_)

    for i in range(H):
        for j in range(W):
            m = mag[i,j]
            high_mask[i,j] = (m > high)
            low_mask[i,j] = (m > low)

    return low_mask, high_mask


@jit(nopython=True, nogil=True)
def connect_blobs(blobs, starts, sizes, high_mask, min_length=-1):
    H = high_mask.shape[0]
    W = high_mask.shape[1]
    blobs_out = np.empty_like(blobs)
    labels = np.empty(blobs.shape[0])

    # kept = []

    label = 1
    m = 0
    for k in range(len(starts)):
        s = starts[k]
        N = sizes[k]
        if N < min_length:
            continue
        # Check associated values from high_mask, break on match
        match = False
        for n in range(s, s + N):
            i, j = blobs[n,0], blobs[n,1]
            if high_mask[i,j]:
                match = True
                break
        if match:
            # kept.append(k)
            for n in range(s, s + N):
                blobs_out[m,1], blobs_out[m,0] = blobs[n,0], blobs[n,1]
                labels[m] = label
                m += 1
                # i, j = blobs[n,0], blobs[n,1]
                # output_mask[i,j] = label
            label += 1

    return blobs_out[:m], labels[:m]


def canny(grey, sigma=1., low_threshold=0.1, high_threshold=0.2, use_quantiles=False):
    # if low_threshold is None:
    #     low_threshold = 0.1 * dtype_limits(image)[1]

    # if high_threshold is None:
    #     high_threshold = 0.2 * dtype_limits(image)[1]

    gx, gy, mag = grad(grey, sigma)

    local_maxima = non_local_suppression(mag, gx, gy)

    #
    #---- If use_quantiles is set then calculate the thresholds to use
    #
    # if use_quantiles:
    #     if high_threshold > 1.0 or low_threshold > 1.0:
    #         raise ValueError("Quantile thresholds must not be > 1.0")
    #     if high_threshold < 0.0 or low_threshold < 0.0:
    #         raise ValueError("Quantile thresholds must not be < 0.0")

    #     high_threshold = np.percentile(mag, 100.0 * high_threshold)
    #     low_threshold = np.percentile(mag, 100.0 * low_threshold)

    low_mask, high_mask = threshold(local_maxima, mag, low_threshold, high_threshold)

    blobs, starts, sizes = Image.find_segments(low_mask, connectivity=8)

    blobs_out, labels = connect_blobs(blobs, starts, sizes, high_mask)

    return blobs_out, labels


def double_threshold(im, low_threshold=0.1, high_threshold=0.2, min_length=-1):
    low_mask = skeletonize( im >= low_threshold )
    high_mask = im >= high_threshold

    blobs, starts, sizes = Image.find_segments(low_mask.copy(), connectivity=8)

    blobs_out, labels = connect_blobs(blobs, starts, sizes, high_mask, min_length)

    return blobs_out, labels


# def labelled_depths(im, D, low_threshold=0.1, high_threshold=0.2, depth_threshold=10, scale=4):
#     # low_mask, high_mask = threshold(im, low_threshold, high_threshold)
#     low_mask = im >= low_threshold
#     high_mask = im >= high_threshold

#     blobs, starts, sizes = Image.find_blobs2(low_mask, D, connectivity=8, depth_thresh=depth_threshold, scale=4)

#     output_mask = connect_blobs(blobs, starts, sizes, high_mask)

#     return output_mask
