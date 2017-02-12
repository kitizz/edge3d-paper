import numpy as np
from numpy import empty, zeros, r_

from numba import jit

# import Util

import os


@jit(nopython=True)
def _label_tracks(track_start, track_labels, nxt):
    N = len(track_labels)
    T = len(track_start)
    for i in range(T):
        n = track_start[i]
        track_labels[n] = i
        while nxt[n] >= 0 and nxt[n] < N:
            n = nxt[n]
            track_labels[n] = i


@jit(nopython=True)
def _count_tracks(track_start, track_sizes, nxt):
    N = len(nxt)
    T = len(track_start)
    for i in range(T):
        n = track_start[i]
        track_sizes[i] = 1
        while nxt[n] >= 0 and nxt[n] < N:
            n = nxt[n]
            track_sizes[i] += 1
            # track_labels[n] = i


_dtype = np.dtype([
    ('q', 'f4', 3),
    ('n', 'f4', 3),
    ('ep', 'f4', 2),
    ('frame', 'i4'),
    # ('contour', 'i8'),
    ('c_ind', 'i4'),
    # ('t_ind', 'i8'),
    ('prev', 'i4'),
    ('next', 'i4')
])


class RayTracks(object):
    def __init__(self, seq=None, size=None, sub='sil_data'):
        if seq is not None:
            self.load(seq, sub)

        elif size is not None:
            self.N = 0
            self.size = size
            self.data = self._empty(size)

            self.T = 0
            self.Tsize = 10
            self.track_start = empty(self.Tsize, int)
            self.track_size = zeros(self.Tsize, int)

    def _empty(self, size):
        return np.recarray(
                size,
                dtype=_dtype
            )

    def add(self, q, n, ep, frame, c_ind, prev=None):
        N = self.N
        if q.ndim == 1:
            self.N += 1
        else:
            self.N += len(q)

        newsize = self.size
        while self.N > newsize:
            newsize *= 2

        if newsize > self.size:
            # print("Resizing: {} -> {}".format(self.size, newsize))
            data = self._empty(newsize)
            data[:self.size] = self.data[:self.size]
            self.size = newsize
            self.data = data
            # self.data.resize(self.size, refcheck=False)

        data = self.data
        if q.ndim == 1:
            inds = r_[N]

            data[N].q = q
            data[N].n = n
            data[N].ep = ep

            # data[N].occl = occl
            # data[N].appr = appr
            data[N].frame = frame
            # data[N].contour = contour
            data[N].c_ind = c_ind
            data[N].next = -1

            if prev is None or prev < 0:
                data[N].prev = -1
                # data[N].t_ind = self.T
                new_tracks = [N]

            else:
                data[N].prev = prev
                # data[N].t_ind = data[prev].t_ind
                data[prev].next = N
                # self.track_size[data[N].t_ind] += 1
                new_tracks = []

        else:
            s, e = N, self.N
            inds = r_[s:e]

            data[s:e].q = q
            data[s:e].n = n
            data[s:e].ep = ep

            # data[s:e].occl = occl
            # data[s:e].appr = appr
            data[s:e].frame = frame
            # data[s:e].contour = contour
            data[s:e].c_ind = c_ind
            data[s:e].next = -1

            if prev is None:
                data[s:e].prev = -1
                data[s:e].next = -1
                # data[s:e].t_ind = T + r_[0:len(q)]
                new_tracks = r_[s:e]

            else:
                existing = np.where( prev >= 0 )[0]
                data.prev[s + existing] = prev[existing]
                data.next[prev[existing]] = s + existing
                if (s + existing >= self.N).any():
                    raise ValueError("Bad next values..")
                # t_ind = data[prev[existing]].t_ind
                # data[s + existing].t_ind = t_ind
                # self.track_size[t_ind] += 1

                new_tracks = s + np.where( prev < 0 )[0]
                data.prev[new_tracks] = -1
                # data[new_tracks].t_ind = T + r_[0:len(new_tracks)]

        if len(new_tracks) > 0:
            T = self.T
            self.T += len(new_tracks)
            while self.T > self.Tsize:
                self.Tsize *= 2
                self.track_start = Util.resize(self.track_start, self.Tsize)
                # self.track_size = Util.resize(self.track_size, fill=0)

            self.track_start[T: self.T] = new_tracks
            # self.track_size[T: self.T] = 1

        self.update_views()

        return inds

    # def occl(self, i):
        # return self.data[i].occl

    # def appr(self, i):
        # return self.data[i].appr

    def update_views(self):
        # view = self.data[:self.N].view(dtype=float, type=np.ndarray).reshape(self.N, -1)
        # self.rays = view[:,:3]
        # self.normals = view[:,3:6]
        self.rays = self.data[:self.N].q
        self.normals = self.data[:self.N].n

    def get_tracks(self, subsample=1):
        # labels = empty(self.N, int)
        nxt = self.data.next
        print("Get labels", nxt.min(), nxt.max())

        track_size = zeros(self.T, int)
        _count_tracks(self.track_start[:self.T], track_size, nxt)

        print("Track sizes:", track_size.min(), track_size.max())

        # track_size = np.bincount(labels)
        tracks = np.array([ empty(T, int) for T in track_size[::subsample] ])

        for i, k in enumerate( range(0, self.T, subsample) ):
            n = self.track_start[k]
            tracks[i][0] = n
            j = 1
            # while nxt[n] >= 0 and nxt[n] < self.N:
            for j in range(1, len(tracks[i])):
                n = nxt[n]
                tracks[i][j] = n
                j += 1

        return tracks

    def save(self, seq, sub='sil_data'):
        folder = os.path.join(seq, sub)
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass

        for name in _dtype.fields.keys():
            np.save('{}/data_{}.npy'.format(folder, name), self.data[name][:self.N])

        # self.data[:self.N].dump(seq + '_sildata.dat')
        np.save('{}/tracks.npy'.format(folder), self.track_start[:self.T])
        # np.savez(seq + '_tracks.npz', tracks=self.track_start[:self.T])

    def load(self, seq, sub='sil_data'):
        # self.data = np.recarray.fromfile
        # self.data = np.load(seq + '_sildata.dat')
        self.N = 0
        folder = os.path.join(seq, sub)
        for name in _dtype.fields.keys():
            dat = np.load('{}/data_{}.npy'.format(folder, name))
            if len(dat) > self.N:
                self.N = len(dat)
                self.data = self._empty(self.N)

            self.data[name] = dat

        self.size = self.N
        self.update_views()

        # F = np.load(seq + '_tracks.npz')
        self.track_start = np.load('{}/tracks.npy'.format(folder))
        self.T = len(self.track_start)
        self.Tsize = self.T

        self.update_views()

