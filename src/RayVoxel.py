import numpy as np
from numpy import empty, zeros, dot

import numba as nb
from numba import jit, jitclass, float32, int64, uint16, uint32

import os

from tools import Util


_indtype = np.uint32
bool_ = nb.bool_
# bool_ = bool

_spec = [
    ( 'voxel_size', float32),
    ( 'shape', int64[:]),
    ( 'n_rays', int64),
    ( 'N', int64),
    ( 'max_dim', int64),
    ( 'offset', float32[:]),
    ( 'size', float32[:]),
    ( 'scale', float32),
    ( 'linearize', int64[:]),
    ( 'ray2vox_count', uint16[:]),
    ( 'vox2ray_count', uint16[:]),
    ( 'ray2vox_inds', uint32[:, :]),
    ( 'vox2ray_inds', uint32[:, :]),
]


@jitclass(_spec)
class Grid(object):
    def __init__(self, max_dim, center, size, max_count):
        '''
        max_dim: Desired number of cubes along the largest dimension
        offset, size: Bounding box offset and size
        max_count: Largest expected number of rays to cut through a voxel
        n_rays: Total number of rays to be added
        '''
        # Calculate voxel size based
        self.voxel_size = np.max(size) / max_dim

        # Make shape a multiple of 4 to ensure aligned memory
        shparr = empty(4, np.int64)
        shparr[:3] = size / self.voxel_size
        shparr[3] = max_count
        # self.shape = empty(4, np.int64)
        self.shape = (np.ceil(shparr/4) * 4).astype(np.int64)

        self.n_rays = 0
        self.N = self.shape[0] * self.shape[1] * self.shape[2]
        self.max_dim = np.max(self.shape[:3])

        self.size = (self.shape[:3] * self.voxel_size).astype(np.float32)
        self.offset = (center - 0.5*self.size).astype(np.float32)
        self.scale = 1/self.voxel_size
        self.linearize = empty(3, np.int64)
        self.linearize[0] = self.shape[1] * self.shape[2]
        self.linearize[1] = self.shape[2]
        self.linearize[2] = 1

        # From a voxel, look up which rays it contains
        self.vox2ray_count = zeros(self.N, np.uint16)
        self.vox2ray_inds = empty((self.N, self.shape[3]), _indtype)

        # From a ray, look up which voxels it belongs to
        self.ray2vox_count = zeros(self.n_rays, np.uint16)
        self.ray2vox_inds = empty((self.n_rays, self.max_dim**2), _indtype)

    def to_linear_ind(self, ijk):
        return self.linearize[0] * ijk[0] + self.linearize[1] * ijk[1] + ijk[2]

    def to_3d_ind(self, ind):
        ijk = empty(3, np.int64)
        ijk[0] = ind // (self.shape[1]*self.shape[2])
        ijk[1] = (ind // (self.shape[2])) % self.shape[1]
        ijk[2] = ind % self.shape[2]
        return ijk

    def build_grid(self, cs, qs):
        N = cs.shape[0]
        self.n_rays = N

        # Reset it all
        self.vox2ray_count[:] = 0
        self.ray2vox_count = zeros(N, np.uint16)
        self.ray2vox_inds = empty((N, self.max_dim**2), _indtype)

        for ray_ind in range(N):
            vox_inds = [0]
            vox_inds.pop()
            end, valid = self.add_ray(cs[ray_ind], qs[ray_ind], vox_inds, False)
            vox_inds.sort()
            self.register_voxels_for_ray(ray_ind, vox_inds)

    def build_grid_lines(self, P1, P2):
        N = P1.shape[0]
        self.n_rays = N

        # Reset it all
        self.vox2ray_count[:] = 0
        self.ray2vox_count = zeros(N, np.uint16)
        self.ray2vox_inds = empty((N, self.max_dim**2), _indtype)

        for ind in range(N):
            vox_inds = [0]
            vox_inds.pop()
            end, valid = self.add_ray(P1[ind], P2[ind] - P1[ind], vox_inds, True)
            vox_inds.sort()
            self.register_voxels_for_ray(ind, vox_inds)

    def build_grid_segments(self, cs, qs, labels):
        N = cs.shape[0]
        self.n_rays = N

        # Reset it all
        self.vox2ray_count[:] = 0
        self.ray2vox_count = zeros(N, np.uint16)
        self.ray2vox_inds = empty((N, self.max_dim**2), _indtype)

        s = 0
        # for ray_ind in range(N):
        while s < N - 1:
            label = labels[s]
            if label == 0:
                s += 1
                continue

            e = s + 1
            while e < N and labels[e] == label:
                e += 1
            self.add_connected_rays(cs[s], qs[s:e + 1], s)
            s = e

    def build_grid_segments_select(self, cs, qs, sel):
        '''
        A selection process has already calculated which consecutive rays are of interest
        so no need to filter by labels.
        '''
        N = cs.shape[0]
        M = sel.shape[0]
        self.n_rays = N

        # Reset it all
        self.vox2ray_count[:] = 0
        self.ray2vox_count = zeros(N, np.uint16)
        self.ray2vox_inds = empty((N, self.max_dim**2), _indtype)

        j = 0
        while j < M - 1:
            s = sel[j]
            while j < M - 1 and sel[j + 1] == sel[j] + 1:
                j += 1
            # e is the end inclusive. So if we want a range that also includes
            # an extra index, we need to bump up by 2
            e = sel[j] + 2
            j += 1
            self.add_connected_rays(cs[s], qs[s:e], s)

    def add_ray(self, c0, q, vox_inds, limit):
        '''
        Rasterize the ray in voxel space. Register the ray with those voxels,
        and register the voxels with the ray.

        vox_inds: A list to append new voxel indices to

        ref 0: http://www.cse.yorku.ca/~amana/research/grid.pdf
        ref 1: https://tavianator.com/fast-branchless-raybounding-box-intersections/
        '''
        # Transform into voxel space
        c = self.scale * (c0 - self.offset)
        qend = self.scale * (c0 + q - self.offset)

        shp = self.shape[:3]

        # Check if the ray touches the voxel grid at all
        iq = 1/q
        if np.any( (c < 0) | (c > shp) ):
            # Want the latest point at which the ray crosses a zero
            t0 = -c*iq
            t1 = (shp - c)*iq
            tmin = np.minimum(t0, t1)
            tmax = np.maximum(t0, t1)
            tminmax = tmin.max()
            tmaxmin = tmax.min()
            if tminmax > tmaxmin or tmaxmin < 0:
                return zeros(3, np.int64), False
            c += np.float32(tminmax*1.000001)*q
            c[c < 0] = 0

        tlimit = (qend[0] - c[0])*iq[0]
        # Rasterize the ray in 3D (see ref 0)
        deltas = np.sign(q).astype(np.int64)
        # Current tells us where we are, target tells us where we want the next intersections
        current = np.floor(c).astype(np.int64)
        target = (current + 0).astype(np.int64)
        for i in range(3):
            if deltas[i] >= 0:
                target[i] += 1
            if deltas[i] == 0:
                iq[i] = np.inf

        while np.all( (target >= 0) & (target < shp) ):
            voxind = self.to_linear_ind(current)
            vox_inds.append(voxind)

            # Ray target ~ (x,y,z) = c + t*q
            # So we can calculate the possible ts by:
            ts = (target - c)*iq
            ax = ts.argmin()
            if limit and ts[ax] >= tlimit:
                break
            d = deltas[ax]
            target[ax] += d
            current[ax] += d

        return current, True

    def add_connected_rays(self, c, Qs, ind0):
        '''
        c: Same center for each ray
        Qs: (N x 3) direction for N rays
        ind0: Ray index of the first ray
        '''
        N = Qs.shape[0]

        last_voxinds = [0]
        last_voxinds.pop()
        last_end, last_valid = self.add_ray(c, Qs[0], last_voxinds, False)

        for i in range(1, N):
            vox_inds = [0]
            vox_inds.pop()
            end, valid = self.add_ray(c, Qs[i], vox_inds, False)
            end_voxinds = vox_inds.copy()

            if valid:
                # Test how many voxels away this ray is from the last
                delta = np.abs(end - last_end).sum()
                if last_valid and delta > 1:
                    # Fill the in-between bits in order to register the
                    # full triangle formed by the 2 rays
                    dinv = 1/delta
                    for j in range(1,delta):
                        w = j*dinv
                        q = w*Qs[i] + (w - 1)*Qs[i-1]
                        self.add_ray(c, q, vox_inds, False)

                vox_inds.extend(last_voxinds)
                vox_inds.sort()
                self.register_voxels_for_ray(ind0 + i - 1, vox_inds)

            # Set it up for the next ray
            last_voxinds = end_voxinds
            last_end = end
            last_valid = valid

    def register_voxels_for_ray(self, ray_ind, vox_inds):
        '''
        For a ray (ray_ind) register it in each voxel provided (vox_inds).
        And register those voxels with the ray.
        '''
        max_count = self.shape[3]
        ray2vox_inds = self.ray2vox_inds[ray_ind]
        j = 0
        i = 0
        N = len(vox_inds)
        voxind = -1
        # for voxind in vox_inds:
        while i < N:
            # Skip over non-unique voxels
            while i < N - 1 and vox_inds[i] == voxind:
                i += 1

            voxind = vox_inds[i]
            i += 1

            count = self.vox2ray_count[voxind]
            if count < max_count:
                # If we haven't maxed out capacity for voxel, register ray
                self.vox2ray_inds[voxind, count] = ray_ind
                self.vox2ray_count[voxind] += 1
                ray2vox_inds[j] = voxind
                j += 1
            # else:
            #     print(ray_ind)

        # Update the ray2vox count
        self.ray2vox_count[ray_ind] = j

    def rays_near_ind(self, ray_ind):
        vox_inds = self.ray2vox_inds[ray_ind, :self.ray2vox_count[ray_ind]]
        if len(vox_inds) == 0:
            return empty(0, np.int64)

        sel = zeros(self.n_rays, bool_)

        for i in vox_inds:
            # Select nearby rays
            ray_inds = self.vox2ray_inds[i, :self.vox2ray_count[i]]
            for ind in ray_inds:
                sel[ind] = True

        return np.where(sel)[0]

    def rays_near_point(self, p):
        # Convert to voxel coords:
        coords = (self.scale * (p - self.offset)).astype(np.int64)

        inds = empty(3*3*3, np.int64)
        n = 0
        # for di, dj, dk in ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    c = coords.copy()
                    c[0] += di
                    c[1] += dj
                    c[2] += dk
                    ind = self.to_linear_ind(c)
                    if ind >= 0 and ind < self.N:
                        inds[n] = ind
                        n += 1
        # ind = self.to_linear_ind(coords)
        # if ind < 0 or ind >= self.N:
        #     return empty(0, np.int64)

        N = 0
        count = empty(n, np.int64)
        cumsum = empty(n + 1, np.int64)
        for i in range(n):
            ind = inds[i]
            count[i] = self.vox2ray_count[ind]
            cumsum[i] = N
            N += count[i]
        cumsum[-1] = N

        # N = self.vox2ray_count[ind]
        sel = empty(N, np.int64)
        for i in range(n):
            ind = inds[i]
            sel[cumsum[i]:cumsum[i+1]] = self.vox2ray_inds[ind, :count[i]]
        # sel[:] = self.vox2ray_inds[ind, :N]

        return Util.unique(sel)

    def compressed(self, count, inds):
        bounds = np.cumsum(count)
        N = bounds[-1]

        res = empty(N, _indtype)

        s = 0
        for i in range(len(count)):
            e = bounds[i]
            res[s:e] = inds[i, :count[i]]
            s = e

        return res

    def uncompress(self, data, count, out):
        bounds = np.cumsum(count)

        s = 0
        for i in range(len(bounds)):
            e = bounds[i]
            out[i, :count[i]] = data[s:e]
            s = e

    def compressed_vox2ray_inds(self):
        return self.compressed(self.vox2ray_count, self.vox2ray_inds)

    def compressed_ray2vox_inds(self):
        return self.compressed(self.ray2vox_count, self.ray2vox_inds)

    def set_compressed_vox2ray_inds(self, data):
        vox2ray = empty((self.N, self.shape[3]), _indtype)
        self.uncompress(data, self.vox2ray_count, vox2ray)

        self.vox2ray_inds = vox2ray

    def set_compressed_ray2vox_inds(self, data):
        ray2vox = empty((self.n_rays, self.max_dim**2), _indtype)
        self.uncompress(data, self.ray2vox_count, ray2vox)

        self.ray2vox_inds = ray2vox


def save(path, grid):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    # Save the small stuff
    metapath = os.path.join(path, 'meta.npz')
    out = dict()
    for name, tp in _spec[:-4]:
        out[name] = getattr(grid, name)

    np.savez(metapath, **out)

    # Then the big stuff
    for name, tp in _spec[-4:-2]:
        valpath = os.path.join(path, '{}.npy'.format(name))
        val = getattr(grid, name)
        np.save(valpath, val)

    for name, tp in _spec[-2:]:
        valpath = os.path.join(path, '{}.npy'.format(name))
        call = "compressed_{}".format(name)
        val = getattr(grid, call)()
        np.save(valpath, val)


def load(path):
    metapath = os.path.join(path, 'meta.npz')

    grid = Grid(1, np.r_[0,0,0], np.r_[1,1,1], 1)
    npz = np.load(metapath)
    for name, tp in _spec[:-4]:
        val = npz[name]
        if val.ndim == 0:
            val = tp(val)
        setattr(grid, name, val)

    # Then the big stuff
    for name, tp in _spec[-4:-2]:
        valpath = os.path.join(path, '{}.npy'.format(name))
        val = np.load(valpath)
        setattr(grid, name, val)

    for name, tp in _spec[-2:]:
        valpath = os.path.join(path, '{}.npy'.format(name))
        val = np.load(valpath)
        call = "set_compressed_{}".format(name)
        getattr(grid, call)(val)

    return grid


@jit(nopython=True)
def test_jit(val):
    N = len(val)

    for i in range(N):
        tmp = val[i] + 1

    return tmp


def test_from_cloud(seq, sub):
    import os
    import RayCloud

    print("Loading")
    cloud = RayCloud.load(os.path.join(seq, sub))

    bbox_min, bbox_max = cloud.bbox_min, cloud.bbox_max
    size = 1.2*(bbox_max - bbox_min)
    center = 0.5*(bbox_max + bbox_min)

    Cs, Qs, Ns = cloud.global_rays()

    raygrid = Grid(75, center, size, 2000)

    print("Building")
    raygrid.build_grid_segments(Cs, Qs, cloud.labels_frame)


if __name__ == '__main__':
    seq = '../data/hat_01'
    sub = 'rays_5'

    test_from_cloud(seq, sub)
