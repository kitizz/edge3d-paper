from __future__ import print_function

import numpy as np
from numpy import zeros

from tools import IO
import Contours
import RayCloud
import RayVoxel

import pynutmeg

import os
import glob
import time
import psutil


def extract_rays(seq, sub, skip=2, max_frame=None, imtype='png'):
    edge_dir = os.path.join(seq, 'edges', '*.'+imtype)
    paths = glob.glob(edge_dir)
    F = len(paths)

    cam = IO.read_cam_param( os.path.join(seq, 'cam.yaml') )
    Rt_path = os.path.join(seq, 'tracking.csv')
    R, t, times, valid = IO.read_tracking(Rt_path)
    valid_bool = zeros(len(R), bool)
    valid_bool[valid] = True

    cloud = RayCloud.RayCloud()
    cloud.set_cam(cam)
    cloud.set_poses(R, t)

    fig = pynutmeg.figure('segments', 'figs/segments.qml')

    if max_frame is None:
        max_frame = F

    print("Estimating bounding box volume from labels")
    frames, boxes = IO.read_rects( os.path.join(seq, 'rects.yaml') )
    cloud.apply_bounding_boxes(frames, boxes)

    print("Extracting Edge Rays")
    for f in range(0, max_frame, skip):
        if not valid_bool[f]:
            continue
        print("\r\tFrame: {} of {}".format(f, F), end=' '*16, flush=True)

        E = IO.imread( paths[f] )
        pts, labels = Contours.find_contours_edge(E, low=20, high=35, min_length=30)
        im = np.empty_like(E)
        im[:] = 255
        im[pts[:,1], pts[:,0]] = 0
        # TODO: Maybe put the next 2 into the one function. This is fine for now though
        pts, labels = Contours.simplify_labeled(pts, labels, eps=1.5)
        tangents, labels = Contours.segment_tangents(pts, labels, thresh=35.)

        # fig.set('ax.im', binary=255-E)
        fig.set('ax.im', binary=im)
        cloud.add_pixels(pts, tangents, f, labels)

    print("\nSaving ray cloud...")
    cloud.save( os.path.join(seq, sub) )


def build_voxel_grid(seq, sub):
    cloudpath = os.path.join(seq, sub)

    cloud = RayCloud.load(cloudpath)

    # Reassign transforms
    Rt_path = os.path.join(seq, 'tracking.csv')
    R, t, times, valid = IO.read_tracking(Rt_path)
    # R, t = IO.read_Rt(os.path.join(seq, 'opti_aligned_Rt.npz'))
    cloud.set_poses(R, t)
    frames, boxes = IO.read_rects( os.path.join(seq, 'rects.yaml') )
    cloud.apply_bounding_boxes(frames, boxes)
    cloud.save(cloudpath)

    voxel_dir = os.path.join(seq, sub + '_voxel')

    print("Building ray voxel grid")
    bbox_min, bbox_max = cloud.bbox_min, cloud.bbox_max
    size = 1.2*(bbox_max - bbox_min)
    center = 0.5*(bbox_max + bbox_min)

    Cs, Qs, Ns = cloud.global_rays()
    print("Making...", cloud.N, Cs.shape)
    raygrid = make_raygrid_info(center, size, Cs, Qs, cloud.labels_frame, showinfo=True)

    print("Saving ray voxels...")
    RayVoxel.save(voxel_dir, raygrid)


def make_raygrid_info(center, size, Cs, Qs, labels, showinfo=False):
    if showinfo:
        t0 = time.time()
        proc = psutil.Process(os.getpid())
        print("Allocating raygrid. Mem: {:.2f} Gb".format(proc.memory_info().rss / 1e9))

    # Initialize the grid
    raygrid = RayVoxel.Grid(80, center, size, 30000)

    if showinfo:
        print("Building raygrid.")

    # Load all the rays in
    raygrid.build_grid_segments(Cs, Qs, labels)
    # raygrid.build_grid(Cs, Qs)

    if showinfo:
        print("Mem: {:.2f} Gb".format(proc.memory_info().rss / 1e9))
        print("Time: {:.1f} s".format(time.time() - t0))
        t0 = time.time()

    return raygrid
