import numpy as np
from numpy import sin, empty, zeros, empty_like, dot, sqrt
from numpy.linalg import norm

import RayCloud

from tools import Geom, IO

from sklearn.neighbors import KDTree

import os

from numba import jit


def reconstruct(seq, sub, cloud, config, scale=1.0):
    stat_path = os.path.join(seq, sub, 'segment_stats.npz')
    poly_path = os.path.join(seq, sub, 'poly_data.npz')
    # soft_dir = os.path.join(seq, 'soft_' + sub)

    # --- Load data ---
    npz = np.load(stat_path)
    depths = npz['depths']
    score = npz['score']
    radii = npz['radii']
    edge_angles = npz['edge_angles']
    inlier_frac = npz['inlier_frac']
    neighbors = npz['neighbors']
    neighbor_count = npz['neighbor_count']
    reciprocated = npz['reciprocated']

    # --- Prepare vars ---
    N = cloud.N

    # Valid inds start at zero. Invalids are -1
    frames = cloud.frames

    Cs, Qs, Ns = cloud.global_rays()
    planes = empty((N,4), Qs.dtype)
    planes[:,:3] = Ns
    planes[:,3] = -(Cs*Ns).sum(axis=1)

    max_error = 5e-2

    # --- Filter inds ---
    abradii = np.abs(radii)

    min_score = config.min_score
    min_radius = config.min_radius*scale
    min_neighbors = config.min_neighbors
    min_inlier = config.min_inlier
    max_inlier = config.max_inlier

    confident = (score >= min_score) & (abradii > min_radius) & (abradii < 100) & (neighbor_count > min_neighbors) & (inlier_frac > min_inlier) & (inlier_frac < max_inlier)
    inds = np.where( confident )[0]

    print(score.shape, cloud.N)

    # --- Pull out the lines and export them ----
    Pa = Cs[inds] + depths[inds,0:1]*Qs[inds]
    Pb = Cs[inds] + depths[inds,1:2]*Qs[inds+1]

    # Read in the fitted polylines, and remove any segments that come
    print("Filtering nearby")
    nearby = np.empty(0, int)
    # nearby = filter_nearby_segments(Pa, Pb, verts, starts, sizes, min_dist=1e-2, min_angle=25)
    # nearby = inds[nearby]
    print("Done", len(nearby))

    # Filter outside bounding box
    bbox_min, bbox_max = cloud.bbox_min, cloud.bbox_max
    print(bbox_min, bbox_max)
    outside = inds[ ~(
        (Pa > bbox_min).all(axis=1) &
        (Pa < bbox_max).all(axis=1) &
        (Pb > bbox_min).all(axis=1) &
        (Pb < bbox_max).all(axis=1))
    ]
    # sel = inds[bad]
    filt = confident
    filt[nearby] = False
    filt[outside] = False
    sel = np.where(filt)[0]

    P1 = Cs[sel] + depths[sel,0:1]*Qs[sel]
    P2 = Cs[sel] + depths[sel,1:2]*Qs[sel+1]

    print("Saving {} edges".format(len(sel)))

    Ps = np.vstack((P1,P2))
    Es = np.r_[0:len(Ps)].reshape(2,-1).T
    normals = np.vstack((Ns[sel], Ns[sel]))

    # mesh = Render.Mesh()
    # mesh.verts = Ps.astype(np.float32)
    # mesh.edges = Es.astype(np.uint32)
    # mesh.normals = Ns[sel].astype(np.float32)
    # mesh_out = os.path.join(seq, 'occlusion_mesh.npz')
    # Render.save_mesh(mesh_out, mesh)

    # datapath = os.path.join(seq, 'occlusion_data.npz')
    # np.savez(datapath, verts=Ps, edges=Es, inds=sel, score=min_score, radius=min_radius, neighbor_count=min_neighbors, inlier=min_inlier)

    cloud_out = os.path.join(seq, sub, 'nonpersistent.ply')
    IO.write_ply(cloud_out, Ps, edges=Es, normals=normals)
    print("Cloud saved")


def filter_nearby_segments(P1, P2, verts, starts, sizes, min_dist, min_angle, support=10):
    '''
    P1, P2: Segments
    verts: Poly vertices
    starts, sizes: Poly sections, where they begin and how long.
    '''
    # First estimate poly direction
    poly_dir = smooth_polys(verts, starts, sizes)

    tree = KDTree(verts)

    rad = np.deg2rad(min_angle)
    max_dot = np.cos(rad)
    print("Max dot:", max_dot)

    dPs = P2 - P1
    dPs /= np.linalg.norm(dPs, axis=1)[:, None]

    N = len(P1)
    nearby = empty(N, bool)

    for i in range(N):
        p1 = P1[i:i+1]
        p2 = P2[i:i+1]
        dp = dPs[i]

        k = 10
        d1s, ind1s = tree.query(p1, k)
        d2s, ind2s = tree.query(p2, k)

        near = False

        for j in range(k):
            d1 = d1s[0, j]
            v1 = poly_dir[ind1s[0,j]]

            d2 = d2s[0, j]
            v2 = poly_dir[ind2s[0,j]]
            if d1 > min_dist and d1 > min_dist:
                break

            if check_dist_angle(d1, v1, dp, min_dist, max_dot):
                near = True
                break
            if check_dist_angle(d2, v2, dp, min_dist, max_dot):
                near = True
                break

        nearby[i] = near

    return nearby


def check_dist_angle(d, u, v, max_d, min_dot):
    return d < max_d and abs(dot(u, v)) > min_dot


def smooth_polys(verts, starts, sizes, support=10):
    poly_dir = np.empty_like(verts)
    S = len(starts)
    for p in range(S):
        start = starts[p]
        size = sizes[p]
        poly = verts[start: start + size]

        N = len(poly)
        for i in range(N):
            s = max(0, i - support)
            e = min(N-1, i + support)

            v = poly[e] - poly[s]
            v /= Geom.norm3(v)
            poly_dir[start + i] = v

    return poly_dir


if __name__ == '__main__':
    # seq = '../data/bottle_04'
    seq = '../data/ch_bunny_02'
    sub = 'rays_1'

    scale = 1

    import sys
    if len(sys.argv) > 1:
        seq = os.path.join('../data', sys.argv[1])
        if len(sys.argv) > 2:
            scale = float(sys.argv[2])

    print("Loading ray cloud")
    cloud = RayCloud.load( os.path.join(seq, sub) )

    reconstruct(seq, sub, cloud, scale)
