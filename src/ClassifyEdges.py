'''
Detect hard edges from a soup of edge segments/rays
'''

import numpy as np
from numpy import dot, empty, zeros, sqrt, sin, arccos
from numpy import pi as PI

import skimage.color as color

import os
import glob
import time

import numba as nb
from numba import jit

import RayCloud
import RayVoxel
import CircleRays
import Config

from tools import Geom, Drawing, IO, Util

from tools.ParallelJit import distribute


def detect(seq, sub, cloud, config, imtype='png'):
    # ------------- Load in data for sequence ---------------
    voxel_dir = os.path.join(seq, sub + '_voxel')

    # Get a rough scale for closeness threshold based on
    # size of camera center bounding box
    print("Loading voxels")
    raygrid = RayVoxel.load(voxel_dir)

    # ------------ Precalc values from data -----------------
    N = cloud.N
    Cs, Qs, Ns = cloud.global_rays()

    planes = empty((N,4), Qs.dtype)
    planes[:,:3] = Ns
    planes[:,3] = -(Cs*Ns).sum(axis=1)

    pluckers = Geom.to_plucker(Cs, Qs)

    # Load the cam so we can offset the image properly
    fx, fy, cx, cy = cloud.cam

    # Initialize all the output data
    score = zeros(N, int)
    total = zeros(N, int)
    curv_mean = zeros(N)
    m2 = zeros(N)
    depths = zeros((N,3), np.float32)
    radii = np.full(N, np.inf, np.float32)
    inlier_frac = zeros(N, np.float32)
    edge_angles = np.full(N, 0, np.float32)
    neighbors = np.full((N, 300), -1, np.int32)
    status = zeros(N, np.int64)

    # --------------------- The meat ------------------------
    ray_inds = np.arange(0, N, dtype=np.int64)
    args = (
        Cs, Qs, pluckers, planes,
        cloud.labels_frame, cloud.frames, raygrid,
        config,
        score, total, curv_mean, m2, depths, radii, inlier_frac, edge_angles, neighbors, status)

    distribute(detect_loop, ray_inds, args, watcher=loop_watcher, w_args=(status,), n_workers=16)

    # Estimate final variance of curvature
    invalid = np.where(total == 0)[0]
    total[invalid] = 1
    curv_var = m2 / total
    curv_var[invalid] = np.inf

    # Count the neighbors
    neighbor_count, reciprocated = count_neighbors(neighbors)

    # --- Save ---
    segout = os.path.join(seq, sub, 'segment_stats.npz')
    np.savez(segout, score=score, total=total, curv_mean=curv_mean, curv_var=curv_var, depths=depths, radii=radii, inlier_frac=inlier_frac, edge_angles=edge_angles, neighbors=neighbors, neighbor_count=neighbor_count, reciprocated=reciprocated)


_t0 = 0
def loop_watcher(finishing, status):
    # TODO: Remove global!
    global _t0
    if finishing:
        dt = max(1, time.time() - _t0)
        m = int(dt // 60)
        s = int(dt) % 60
        print("Job completed in {}m {}s".format(m, s), end=' '*20 + '\n')
        return

    N = len(status)
    done = np.where(status != 0)[0]
    # good = np.where(status == 1)[0]

    D = len(done)

    if _t0 == 0 or D <= 1:
        _t0 = time.time()

    dt = max(1, time.time() - _t0)
    nps = D/dt  # Rays per second
    n_left = N - D
    seconds_left = n_left/max(1e-3, nps)
    m = int(seconds_left // 60)
    s = int(seconds_left) % 60

    print("Status... Done: {}/{}, ETA: {}m {}s".format(D, N, m, s), end='       \r', flush=True)


@jit(nopython=True, nogil=True)
def filter_frames(near0, frames, ray_ind, support):
    near = np.empty_like(near0)
    j = 0
    f = frames[ray_ind]
    for i in near0:
        if abs(frames[i] - f) <= support:
            near[j] = i
            j += 1

    return near[:j]


@jit(nopython=True, nogil=True)
def detect_loop(
        ray_inds, Cs, Qs, pluckers, planes,
        labels, frames,
        raygrid, config,
        score, total, curv_mean, m2,
        depths_out, radii_out, inlier_out, edge_angles_out, neighbors,
        status):
    '''
    score: Output, integer array classifying the type of each segment
    '''
    # Each valid segment is labelled with a non-zero label
    # tim = Util.LoopTimer()
    # loop_i = 0
    for ray_ind in ray_inds:
        # tim.loop_start(loop_i)
        # loop_i += 1

        if labels[ray_ind] == 0:
            status[ray_ind] = -1
            score[ray_ind] = 0
            continue

        # First find other segments that are nearby, and might overlap
        # The RayVoxel.Grid class offers very fast lookups at the cost of memory
        near_grid = raygrid.rays_near_ind(ray_ind)
        # tim.add("RayGrid Search")

        if len(near_grid) < 10:
            status[ray_ind] = -1
            continue

        # Filter frame support
        near = filter_frames(near_grid, frames, ray_ind, config.frame_support)
        # close_frames = np.abs(frames[near_grid] - frames[ray_ind]) <= config.frame_support
        # near = near_grid[close_frames]
        # tim.add("Filter frames")

        # Intersect the shortlisted segments with this segment
        # ps, us describe the intersection locations of the other segments'
        # start- and end-points.
        ps1, us, intersecting = find_intersections(Cs, Qs, pluckers, ray_ind, planes, near)
        # print("Near:", len(near))
        # tim.add("Find intersections")
        if len(intersecting) < 10:
            status[ray_ind] = -1
            continue

        ray = 0.5*(Qs[ray_ind] + Qs[ray_ind+1])
        ray /= Geom.norm3(ray)
        plane = planes[ray_ind]
        normal = plane[:3]
        tangent = Geom.cross3(ray, normal)

        # Create rotation matrix to orthoganally project intersections
        # into plane of this segment. This plane is tangent to the edge
        R_tang = empty((2,3), ps1.dtype)
        R_tang[0] = ray
        R_tang[1] = tangent

        center = Cs[ray_ind]
        ps2d = dot(R_tang, (ps1 - center).T)
        us2d = dot(R_tang, us.T)
        px, py = ps2d[0], ps2d[1]
        ux, uy = us2d[0], us2d[1]

        # Solve for the intersection with the center of the segment
        # [x, 0] = p + t*u
        uy[uy == 0] = 1e-10
        ts = -py / uy
        depths = px + ts*ux
        # Project the intersecting segments into the normal plane.
        # This plane's normal is roughly aligned with the edge direction.
        R_norm = empty((3,3), ps1.dtype)
        R_norm[0] = ray
        R_norm[1] = normal
        R_norm[2] = tangent
        centers = dot(R_norm, (Cs[intersecting] - center).T)

        # plane_angles = np.arctan(centers[1]/centers[2])

        # Check where start and end points straddles center ray of this segment
        crossing = py * (py + uy) < 0
        good_depths = (depths > 0.2) & (depths < 1e6)
        # close_frames = np.abs(frames[intersecting] - frames[ray_ind]) <= config.frame_support
        # in_normal_plane = np.abs(plane_angles) <= config.planar_support

        sel = np.where(good_depths & crossing)[0]
        # sel = np.where(good_depths & crossing & (close_frames | in_normal_plane))[0]
        # tim.add("Intersect filter")
        # print("Sel:", len(sel))

        # We're looking for clusters of line segments in the tangent plane
        # These indicate a high likelihood of a persistent edge
        # frames = cloud.frames[ intersecting[sel] ]
        # p = ps2d[:,sel]
        u = us2d[:,sel]
        d = depths[sel]

        c_sel = centers[:,sel]
        rays = empty((2, len(sel)), centers.dtype)
        rays[0] = d - c_sel[0]
        rays[1] = -c_sel[1]

        rays /= CircleRays.norm_ax(rays, axis=0)
        # tim.add("Prepare clustering")

        dthresh = config.depth_thresh
        # Cluster based on circle space
        cluster_inds, radius, depth, dual_centers, dual_rays, inlier_frac = find_cluster_line4(
            c_sel, rays, depth_thresh=dthresh, percentile=config.inlier_thresh)

        # tim.add("Clustering")

        cluster_full = intersecting[ sel[cluster_inds] ]
        M = min(neighbors.shape[1], len(cluster_full))

        if inlier_frac > 0.02:
            if abs(radius) < config.min_radius:
                # Hard edge
                status[ray_ind] = 1
                score[ray_ind] += len(cluster_full)
            else:
                status[ray_ind] = 2
                score[ray_ind] += len(cluster_full)

            # Want to estimate the depths of the rays on either side of the segment
            u_cluster = u[:, cluster_inds]
            intersect_angles = np.arctan(-u_cluster[0]/u_cluster[1])
            # Zero is verticle
            alpha = np.median(intersect_angles)
            # Find angle between rays
            theta = 0.5 * arccos( dot(Qs[ray_ind], Qs[ray_ind+1]) )
            d1 = depth * sin(PI/2 + alpha) / sin(PI/2 - alpha - theta)
            d2 = depth * sin(PI/2 - alpha) / sin(PI/2 + alpha - theta)
            depths_out[ray_ind, 0] = d1
            depths_out[ray_ind, 1] = d2
            depths_out[ray_ind, 2] = depth

            # depths_out[ray_ind] = depth
            radii_out[ray_ind] = radius
            inlier_out[ray_ind] = inlier_frac
            edge_angles_out[ray_ind] = alpha

            ray_c = rays[:, cluster_inds]
            cluster_angles = np.abs(np.arctan(ray_c[1]/ray_c[0]))

            closest = Util.argsort(cluster_angles)
            neighbors[ray_ind, :M] = cluster_full[closest[:M]]

        else:
            status[ray_ind] = -1
            score[ray_ind] = 0
            labels[ray_ind] = 0
            continue
        # tim.add("Final checks")
        # print(tim)


@jit(nopython=True, nogil=True)
def find_intersections(Cs, Qs, plucker, ray_ind, planes, sel):
    '''
    Plucker coords make the collision check quite fast, requiring few ops.

    Cs, Qs, plucker: Canonical and Plucker representations of set of rays
    ray_ind: Specific inds of the rays/ray triangle that's being intersected
    plane: Plane describing the ray triangle
    sel: Subset of rays to check
    '''
    N = Cs.shape[0]

    pl1a, pl1b = plucker[ray_ind], plucker[ray_ind + 1]
    plane = planes[ray_ind]

    intersects = empty(N, np.int64)
    ps1 = empty((N,3), Cs.dtype)
    us = empty((N,3), Cs.dtype)
    j = 0

    pl2a = plucker.T[:, sel]
    pl2b = plucker.T[:, sel + 1]
    skews = np.sign( Geom.check_plucker(pl1a, pl2a) ) \
            + np.sign( Geom.check_plucker(pl1a, pl2b) ) \
            + np.sign( Geom.check_plucker(pl1b, pl2a) ) \
            + np.sign( Geom.check_plucker(pl1b, pl2b) )

    for s, i in enumerate(sel):
        if abs(skews[s]) == 4:
            # Both rays skew on the same side of the other 2 rays
            continue

        # Find the intersection of the rays with the plane
        c1, c2 = Cs[i], Cs[i+1]
        q1, q2 = Qs[i], Qs[i+1]
        t1, int1 = Geom.intersect_line_plane(c1, q1, plane)
        t2, int2 = Geom.intersect_line_plane(c2, q2, plane)
        if not int1 or not int2:
            continue

        intersects[j] = i
        for k in range(3):
            ps1[j, k] = c1[k] + t1*q1[k]
            us[j,k] = (c2[k] + t2*q2[k]) - ps1[j,k]
            # ps2[j, k] = c2[k] + t2*q2[k]
        j += 1

    return ps1[:j], us[:j], intersects[:j]


@jit(nopython=True)
def find_cluster_line3(deltas, rays, depths, init_support, res, thresh):
    '''
    Given a set of line segments, defined by (x1, y1) = p and (x2, y2) = p + u,
    locate clusters
    The segments are intersection points of ray segments.
    ray_angles are projected ray angles of camera ray angles.
    Depths are distance the segments intersect the center of the ray of interest
    '''
    N = rays.shape[1]
    if N == 0:
        return empty(0, np.int64), 100.0, 0.0, empty(0, depths.dtype), 0.0

    # Note: (ps, us, depths, frames) should be ordered by frames
    ray_angles = empty(N, depths.dtype)
    for i in range(N):
        iqy = 1/rays[1,i]
        pos = (1 - rays[0,i]) * iqy
        neg = -(1 + rays[0,i]) * iqy
        if abs(pos) < abs(neg):
            ray_angles[i] = pos
        else:
            ray_angles[i] = neg

    dc = np.abs(ray_angles)

    for k in range(2):
        sel = np.where(dc <= init_support)[0]

        if len(sel) == 0:
            init_support += 0.02
            continue

        # Search for a "big enough" cluster of depths according to nearby cameras
        hist, histx = Util.histogram(depths[sel], bins=int(2/res[1]), rng=(0,2))
        i = hist.argmax()

        if hist[i] > 10:
            break
        init_support += 0.02

    else:
        return empty(0, np.int64), 100.0, 0.0, ray_angles, 0.0

    peak_depth = 0.5*(histx[i] + histx[i + 1])
    sel = np.where(dc < init_support * 5)[0]
    depth, radius, inside, good = CircleRays.sequential_circle_space(depths[sel], ray_angles[sel], peak_depth, init_support, thresh[1])

    return sel[inside], radius, depth, ray_angles, len(inside)/len(sel)


@jit(nopython=True, nogil=True)
def find_cluster_line4(centers, rays, depth_thresh=2e-3, percentile=0.03):
    '''
    Given a set of line segments, defined by (x1, y1) = p and (x2, y2) = p + u,
    locate clusters
    The segments are intersection points of ray segments.
    ray_angles are projected ray angles of camera ray angles.
    Depths are distance the segments intersect the center of the ray of interest
    '''
    N = rays.shape[1]
    if N < 5:
        return empty(0, np.int64), 100.0, 0.0, empty((2,0), centers.dtype), empty((2,0), centers.dtype), 0.0

    # Note: (ps, us, depths, frames) should be ordered by frames
    dual_centers = empty((N, 2), centers.dtype)
    dual_rays = empty((N, 2), centers.dtype)
    # dual_normals = empty((N, 2), centers.dtype)
    mult = 5
    for i in range(N):
        nx = rays[1, i]
        ny = rays[0, i]
        if ny < 0:
            ny *= -1
        else:
            nx *= -1
        nyy = ny - 1
        mag = sqrt(nx*nx + nyy*nyy)

        c = centers[0, i] * nx + centers[1, i] * ny

        if ny == 1 or mag == 0:
            dual_centers[i, 0] = 0
            dual_centers[i, 1] = 0
            dual_rays[i, 0] = 0
            dual_rays[i, 1] = 1
            continue

        dual_rays[i, 0] = -mult*nyy/mag
        dual_rays[i, 1] = nx/mag
        dual_centers[i,0] = mult
        dual_centers[i,1] = (c - nx)/nyy

    d, r, inliers = CircleRays.fit_point(dual_centers, dual_rays, eps=mult*depth_thresh, percentile=percentile)
    depth = d / mult
    radius = r

    return inliers, radius, depth, dual_centers.T, dual_rays.T, len(inliers)/N


@jit(nopython=True)
def count_neighbors(neighbors):
    N = neighbors.shape[0]
    M = neighbors.shape[1]
    count = empty(N, np.int64)
    reciprocated = np.full((N, M), -1, np.int64)

    for i in range(N):
        c = 0
        for j in range(M):
            ind = neighbors[i, j]
            if ind < 0:
                continue
            for k in range(M):
                if neighbors[ind, k] == i:
                    # i is reciprocated
                    reciprocated[i, c] = ind
                    c += 1
                    break

        count[i] = c

    return count, reciprocated


def show_edges(seq, cloud, thresh, min_score, imtype='png'):
    edge_dir = os.path.join(seq, 'edges', '*.'+imtype)
    paths = glob.glob(edge_dir)

    segout = os.path.join(seq, 'segment_stats.npz')
    # --- Load data ---
    npz = np.load(segout)
    score = npz['score']
    # total = npz['total']
    # curv_mean = npz['curv_mean']
    # curv_var = npz['curv_var']

    # --- Prepare data ---
    F = len(paths)

    fx, fy, cx, cy = cloud.cam
    labels = cloud.labels_frame

    focal = cloud.cam[:2]
    center = cloud.cam[2:]

    outfolder = os.path.join(seq, 'hard_edges')
    Util.try_mkdir(outfolder)

    # --- Do it ---
    print("Writing out frames...")

    j = 0
    for f in range(F):
        if f % 5:
            continue
        print("Frame {}".format(f))
        path = paths[f]
        im = IO.imread(path)
        col = color.gray2rgb(255-im)

        s, e = cloud.frame_range[f]

        for i in range(s, e):
            if labels[i] == 0 or score[i] < min_score:
                continue

            line = empty(4)
            line[:2] = cloud.local_rays[i, :2] * focal + center
            line[2:] = cloud.local_rays[i + 1, :2] * focal + center
            np.round(line, out=line)
            I, J = Drawing.line(*(line.astype(int)))
            Drawing.draw_points(col, I, J, color=(255, 0, 0))
            Drawing.draw_points(col, I+1, J, color=(255, 0, 0))
            Drawing.draw_points(col, I, J+1, color=(255, 0, 0))
            # col[I, J] = 255, 0, 0

        out = os.path.join(outfolder, '{:04d}.png'.format(f))
        j += 1
        IO.imwrite(out, col)


if __name__ == '__main__':
    # seq = '../data/glass_render'
    # seq = '../data/bottle_04'
    # seq = '../data/ch_bunny_01'
    seq = '../data/ch_bottle_01'
    sub = 'rays_1'
    imtype = 'png'

    import sys
    if len(sys.argv) > 1:
        seq = os.path.join('../data', sys.argv[1])

    print("Loading rays")
    cloud = RayCloud.load( os.path.join(seq, sub) )

    dthresh = 70e-4
    # dthresh = 6e-3

    config = Config.create(
        eps=1/cloud.cam[0], frame_support=150, angle_support=0.06,
        planar_support=0.15,
        min_radius=5e-2,
        angle_res=np.deg2rad(5), angle_thresh=np.deg2rad(10),
        depth_res=7e-3, depth_thresh=dthresh,
        min_score=80)

    # Detect hard edges and save out
    detect(seq, sub, cloud, config, imtype=imtype)

    # Rebuild a voxel grid with the newly classified shards
    # rebuild_persistent_voxel(seq, sub, cloud, config)

    # Draw in and save out marked hard edges
    # show_edges(seq, cloud, config.min_radius, config.min_score, imtype=imtype)
