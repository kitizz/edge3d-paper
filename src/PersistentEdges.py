import numpy as np
from numpy import sin, empty, zeros, empty_like, dot, sqrt

import RayCloud
import RayVoxel
import Render

from tools import IO, Util

import networkx as nx
# import metis

import os

from numba import jit


def reconstruct(seq, sub, cloud, config, scale=1):
    stat_path = os.path.join(seq, sub, 'segment_stats.npz')
    filter_path = os.path.join(seq, sub, 'reciprocal_filtered.npz')

    # --- Load data ---
    npz = np.load(stat_path)
    depths = npz['depths']
    score = npz['score']
    radii = npz['radii']
    inlier_frac = npz['inlier_frac']
    neighbor_count = npz['neighbor_count']
    reciprocated = npz['reciprocated']

    # --- Prepare vars ---
    N = cloud.N

    # Valid inds start at zero. Invalids are -1
    Cs, Qs, Ns = cloud.global_rays()
    planes = empty((N,4), Qs.dtype)
    planes[:,:3] = Ns
    planes[:,3] = -(Cs*Ns).sum(axis=1)

    # --- Filter inds ---
    abradii = np.abs(radii)
    min_score = config.min_score
    min_radius = config.min_radius*scale
    min_neighbors = config.min_neighbors
    min_inlier = config.min_inlier

    good = (score >= min_score) & (abradii < min_radius) & (neighbor_count > min_neighbors) & (inlier_frac > min_inlier)
    inds = np.where(good)[0]

    if not os.path.exists(filter_path):
        print("Filtering common")
        rec_filtered, rec_count = filter_common(inds, good, reciprocated, neighbor_count, min_common=20)
        print("Saving")
        np.savez(filter_path, reciprocated=rec_filtered, count=rec_count)
        print("Done")
    else:
        npz = np.load(filter_path)
        rec_filtered = npz['reciprocated']
        rec_count = npz['count']

    bbox_min, bbox_max = cloud.bbox_min, cloud.bbox_max
    print(bbox_min, bbox_max)

    keep = good
    inds2 = np.where(keep)[0]

    Pa = Cs[inds2] + depths[inds2,0:1]*Qs[inds2]
    Pb = Cs[inds2] + depths[inds2,1:2]*Qs[inds2+1]

    inside = inds2[(
        (Pa > bbox_min).all(axis=1) &
        (Pa < bbox_max).all(axis=1) &
        (Pb > bbox_min).all(axis=1) &
        (Pb < bbox_max).all(axis=1))
    ]

    sel = inside

    print("Persistent edges:", len(sel))

    Pa = Cs[sel] + depths[sel,0:1]*Qs[sel]
    Pb = Cs[sel] + depths[sel,1:2]*Qs[sel+1]

    rebuild_persistent_voxel(seq, sub, cloud, sel, Pa, Pb)

    verts = np.vstack((Pa, Pb))
    V = len(verts)
    edges = np.r_[0:V].reshape(2,-1).T

    cloud_out = os.path.join(seq, sub, 'persistent_cloud.ply')
    IO.save_point_cloud(cloud_out, verts, edges)

    pers_out = os.path.join(seq, sub, 'persistent_data.npz')
    np.savez(pers_out, verts=verts, edges=edges, inds=sel, P1=Pa, P2=Pb)

    mesh = Render.Mesh()
    mesh.verts = verts.astype(np.float32)
    mesh.edges = edges.astype(np.uint32)
    mesh.normals = Ns[sel].astype(np.float32)
    mesh_out = os.path.join(seq, 'persistent_mesh.npz')
    Render.save_mesh(mesh_out, mesh)


def rebuild_persistent_voxel(seq, sub, cloud, inds, P1, P2):
    '''
    Rebuild a voxel grid with only the labelled persistent/hard edges
    '''
    hard_dir = os.path.join(seq, 'hard_' + sub)

    # Prepare vars..
    bbox_min, bbox_max = cloud.bbox_min, cloud.bbox_max
    size = 1.2*(bbox_max - bbox_min)
    center = 0.5*(bbox_max + bbox_min)
    Cs, Qs, Ns = cloud.global_rays()

    print("Rebuilding persistent voxel grid")
    # Initialize the grid
    raygrid = RayVoxel.Grid(40, center, size, 1000)
    # Build it
    # raygrid.build_grid_segments_select(Cs, Qs, inds)
    raygrid.build_grid_lines(P1, P2)

    print("Saving out")
    RayVoxel.save(hard_dir, raygrid)
    del raygrid


@jit(nopython=True)
def filter_common(inds, good, reciprocated, reciprocated_count, min_common=10):
    N = len(inds)
    M = reciprocated.shape[1]

    new_rec = np.full((reciprocated.shape[0], M), -1, np.int64)
    new_count = np.zeros(reciprocated.shape[0], np.int64)

    for i in range(N):
        # print("{}/{}".format(i,N))
        if i % 1000 == 0:
            print(i, N)
        ind = inds[i]
        for j in range(reciprocated_count[ind]):
            other = reciprocated[ind, j]
            if not good[other]:
                continue

            common = 0
            # Iterate the other's reciprocates and make sure it has the minimum number in common with ind
            for k in range(reciprocated_count[other]):
                mid = reciprocated[other, k]
                if not good[other]:
                    continue
                for l in range(reciprocated_count[mid]):
                    if reciprocated[mid, l] == ind:
                        common += 1
                        break

            if common >= min_common:
                c = new_count[ind]
                new_rec[ind, c] = other
                new_count[ind] += 1

    remove = np.where(new_count < 10)[0]
    new_rec[remove] = -1
    new_count[remove] = 0
    max_count = new_count.max()

    return new_rec[:,:max_count], new_count


def find_clusters(inds, good, reciprocated, reciprocated_count):
    graph = nx.Graph()

    N = inds.shape[0]
    M = reciprocated.shape[1]

    for i in range(N):
        ind = inds[i]
        for j in range(M):
            other = reciprocated[ind, j]
            if other < 0 or not good[other]:
                continue
            graph.add_edge(ind, other)

    print("Partitioning")
    res, labels = metis.part_graph(graph, 100)

    print("Creating lookup")
    lookup = dict()
    nodes = graph.nodes()
    for i in range(len(nodes)):
        ind = nodes[i]
        # ind = inds[node]
        label = labels[i]
        if label not in lookup:
            lookup[label] = []
        lookup[label].append(ind)

    return lookup


def write_clusters(outdir, lookup, P1, P2, indmap):
    Util.try_mkdir(outdir)
    verts_all = []
    edges_all = []
    E = 0
    for key, cluster in lookup.items():
        # path = os.path.join(outdir, 'set_{}.ply'.format(key))

        inds = indmap[cluster]
        verts = np.vstack((P1[inds], P2[inds]))
        V = len(verts)
        edges1 = np.r_[0:V].reshape(2,-1).T
        edges2 = np.c_[np.r_[0:V-1], np.r_[1:V]]
        edges = np.vstack((edges1, edges2))

        verts_all.append(verts)
        edges_all.append(edges + E)

        E += len(edges)

    path = os.path.join(outdir, '_full.ply')
    IO.save_point_cloud(path, np.vstack(verts_all), np.vstack(edges_all))


if __name__ == '__main__':
    seq = '../data/ch_bottle_01'
    sub = 'rays_1'
    scale = 1

    import sys
    if len(sys.argv) > 1:
        seq = os.path.join('../data', sys.argv[1])
        if len(sys.argv) > 2:
            scale = float(sys.argv[2])

    print("Loading ray cloud")
    cloud = RayCloud.load( os.path.join(seq, sub) )

    reconstruct(seq, sub, cloud, scale=scale)
