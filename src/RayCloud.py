import numpy as np
from numpy import empty, zeros, dot
from numpy.linalg import norm

from numba import jit

from tools import Geom

import os


@jit(nopython=True)
def apply_Rs(rays, frames, Rs):
    N = rays.shape[0]
    out = np.empty_like(rays)

    for i in range(N):
        R = Rs[frames[i]]
        out[i] = dot(R.T, rays[i])

    return out


@jit(nopython=True)
def check_ray_bounds(qs, R, t, labels, b_min, b_max):
    C = -dot(R.T, t)
    Qs = dot(qs, R)

    N = Qs.shape[0]
    good = empty(N, np.int64)
    j = 0
    for i in range(N):
        if Geom.intersect_ray_aabb(C, Qs[i], b_min, b_max):
            good[j] = i
            j += 1
        else:
            labels[i-1: i+2] = 0

    return good[:j]


class RayCloud(object):
    def __init__(self):
        self.set_cam(np.r_[1.0, 1.0, 0.0, 0.0])

        # Ray stuff
        self.size = 1024
        self.N = 0
        self.local_rays = empty((self.size, 3), np.float32)
        self.local_normals = empty((self.size, 3), np.float32)
        self.frames = empty(self.size, np.int64)
        self.labels_frame = empty(self.size, np.int64)
        self.labels_3d = empty(self.size, np.int64)

        # Pose stuff
        self.frame_range = zeros((0,2), np.uint32)
        self.set_poses( empty((0,3,3)), empty((0,3)) )
        self.bbox_min = np.r_[-2.,-2,-2]
        self.bbox_max = np.r_[2.,2,2]

    def _growable(self):
        return dict(
            local_rays=3,
            local_normals=3,
            frames=1,
            labels_frame=1,
            labels_3d=1
        )

    def _names_sizes(self):
        return dict(
            local_rays=self.N,
            local_normals=self.N,
            frames=self.N,
            labels_frame=self.N,
            labels_3d=self.N,
            Rs=self.F,
            ts=self.F,
            Cs=self.F,
            frame_range=self.F,
            cam=4,
            bbox_min=3,
            bbox_max=3,
        )

    def set_cam(self, cam):
        self.cam = cam
        self.focal = cam[:2]
        self.ifocal = 1/self.focal
        self.center = cam[2:]

    def set_poses(self, R, t):
        self.F = R.shape[0]
        self.Rs = R.astype(np.float32)
        self.ts = t.astype(np.float32)
        self.Cs = Geom.global_to_local(R, t)[1].astype(np.float32)

        if len(self.frame_range) == 0:
            self.frame_range = zeros((self.F,2), np.uint32)

    def add_pixels(self, pix, tangents, frame, labels):
        '''
        Add a set of points that are in image pixel coords
        '''
        M = pix.shape[0]
        if M < 2:
            return

        qs = empty((M, 3), np.float32)
        qs[:, :2] = (pix - self.center) * self.ifocal
        qs[:, 2] = 1.0
        tans = empty((M, 3), np.float32)
        tans[:, :2] = tangents * self.ifocal
        tans[:, 2] = 0.0

        keep = check_ray_bounds(qs, self.Rs[frame], self.ts[frame], labels, self.bbox_min, self.bbox_max)
        # keep = np.r_[0:M]
        if len(keep) < 2:
            return

        s = self.N
        e = s + keep.shape[0]

        self._grow(e)  # Ensure we have enough space

        self.local_rays[s:e] = qs[keep]

        self.frames[s:e] = frame
        self.labels_3d[s:e] = 0

        normals_label(tans, qs, keep, labels, s, self.local_normals, self.labels_frame)

        self.frame_range[frame] = s, e

        self.N = e

    def global_rays(self, offset=0):
        N = self.N
        frames = np.clip(self.frames[:N] + offset, 0, self.F-1)
        Cs = self.Cs[frames]
        Qs = apply_Rs(self.local_rays[:N], frames, self.Rs)
        Qs /= norm(Qs, axis=1)[:,None]
        Ns = normals_from_ray_pairs(Qs)
        # Ns = apply_Rs(self.local_normals[:N], self.frames[:N], self.Rs)
        return Cs, Qs, Ns

    def apply_bounding_boxes(self, frames, bboxes):
        tw = Geom.global_to_local(self.Rs, self.ts)[1]
        center = tw.mean(axis=0)
        size = 5.0*np.abs(tw - center).max(axis=0)
        # center = np.r_[0.0, 0.0, 1.0]
        # size = np.r_[1.0, 1.0, 1.0]
        bbox_min, bbox_max = Geom.frustrum_bounding_box(self.Rs, self.ts, self.cam, frames, bboxes, center=center, size=size, res=100)
        print("BBOX:", bbox_min, bbox_max)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def save(self, folder):
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass

        for name, size in self._names_sizes().items():
            arr = getattr(self, name)
            out = os.path.join(folder, '{}.npy'.format(name))
            np.save(out, arr[:size])

    def load(self, folder):
        for name, size in self._names_sizes().items():
            out = os.path.join(folder, '{}.npy'.format(name))
            arr = np.load(out)
            setattr(self, name, arr)

        self.N = self.local_rays.shape[0]
        self.F = self.Rs.shape[0]
        self.size = self.N

        self.set_cam(self.cam)

    def _resize(self, arr, newsize, N=None):
        if N is None:
            N = self.N

        shp = list(arr.shape)
        shp[0] = newsize
        newarr = empty( shp, arr.dtype )
        newarr[:N] = arr[:N]
        return newarr

    def _grow(self, target):
        newsize = self.size
        while newsize < target:
            newsize += newsize//2  # Grow by 1.5

        for name in self._growable().keys():
            arr = getattr(self, name)
            newarr = self._resize(arr, newsize)
            setattr(self, name, newarr)

        self.size = newsize


@jit(nopython=True)
def normals_label(tans, qs, keep, labels, s, local_normals, labels_frame):
    label = 1
    for i in range( len(keep) - 1 ):
        j = keep[i]
        k = s + i
        if labels[j] != 0:
            labels_frame[k] = label
        else:
            labels_frame[k] = 0

        normal = Geom.cross3(tans[j], qs[j])
        mag = Geom.norm3(normal)
        if mag == 0:
            local_normals[k] = 1, 0, 0
            labels_frame[k] = 0
        else:
            normal /= mag
            local_normals[k] = normal

        if labels[j + 1] != labels[j]:
            label += 1

    j = keep[-1]
    normal = Geom.cross3(tans[j], qs[j])
    normal /= Geom.norm3(normal)
    local_normals[k] = normal

    local_normals[s + len(keep) - 1] = normal
    labels_frame[s + len(keep) - 1] = 0


@jit(nopython=True)
def normals_from_ray_pairs(Qs):
    '''
    Each sequential ray is treated as bases for normals
    '''
    N = Qs.shape[0]
    normals = np.empty_like(Qs)

    for i in range(0, N-1):
        n = Geom.cross3(Qs[i+1], Qs[i])
        mag = Geom.norm3(n)
        for j in range(3):
            normals[i, j] = n[j]/mag

    normals[-1] = normals[-2]

    return normals


def load(path):
    cloud = RayCloud()
    cloud.load(path)
    return cloud
