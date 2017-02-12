import numpy as np
from numpy import r_, empty, zeros, ones, dot

# import Quat
from . import IO
# import Geometry as Geom
# import Align
# import Calibrate

import os
import sys
# import time

from subprocess import run

# import pynutmeg

from numba import jit
# import numba as nb


def video_to_images(seq, fps=None):
    vid = os.path.join(seq, 'video.mp4')
    image_path = os.path.join(seq, 'seq')

    if not os.path.exists(image_path):
        print("Converting video into image sequence in {}".format(image_path))
        os.makedirs(image_path)
        seq_out = os.path.join(image_path, '%05d.jpg')
        cmd = ['/usr/local/bin/avconv', '-i', vid, '-vf', "format=yuv420p", '-q:v', '1']
        if fps is not None:
            cmd.extend(['-vf', 'fps={}'.format(fps)])

        cmd.append(seq_out)
        run(cmd, check=True, stdout=sys.stdout)

    return image_path


def video_timestamps(seq):
    ''' Pull out timestamps for sequence's video '''
    vid = os.path.join(seq, 'video.mp4')
    time_path = os.path.join(seq, 'timestamps.npy')

    if not os.path.exists(time_path) and os.path.exists(vid):
        print("Extracting timestamps from video to {}".format(time_path))
        times = IO.get_video_timestamps(vid)
        np.save(time_path, times)

    elif os.path.exists(time_path):
        times = np.load(time_path)

    else:
        times = np.empty(0)

    return times


# def update_fig(fig, t, P1, P2, t2=None, valid=None):
#     axes = ['xaxis', 'yaxis', 'zaxis']
#     if t2 is None:
#         t2 = t

#     for i, ax in enumerate(axes):
#         fig.set(ax + '.P1', x=t, y=P1[:,i])
#         fig.set(ax + '.P2', x=t2, y=P2[:,i])

#         if valid is not None:
#             fig.set(ax + '.valid', x=t, y=valid.astype(float))


# @jit(nopython=True, nogil=True)
# def interp_pos(t, t0, P0, valid):
#     N = len(t)
#     M = len(t0)

#     cols = P0.shape[1]

#     P = empty((N, cols))
#     val = empty(N, nb.bool_)

#     ind = 0
#     max_ind = M - 2
#     for i in range(N):
#         time = t[i]
#         while ind < max_ind and t0[ind + 1] < time:
#             ind += 1

#         # if not valid[ind] or not valid[ind + 1]:
#         val[i] = valid[ind] and valid[ind + 1]

#         a, b = t0[ind], t0[ind + 1]
#         frac = (time - a) / (b - a)

#         for c in range(cols):
#             P[i, c] = (1 - frac)*P0[ind, c] + (frac)*P0[ind + 1, c]

#     return P, val
