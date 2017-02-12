'''
SandBox.py
----------
Code that I was using to experiment with and understand how edges behaved.
Very raw, a bit of a mess. Read at your own peril. I've included it in the
name of being open, and would be a good place to start if anyone wanted to
understand the algorithms with the help of Nutmeg for visualization.
'''

from numpy import empty, zeros, dot, sqrt
from numpy.linalg import norm

from tools import IO, Geom, Util
from tools.FastMath import norm3, cross3

from skimage.morphology import skeletonize

import Contours
import RayCloud
import RayVoxel
from RaySoup import extract_rays, build_voxel_grid

import pynutmeg

import os
import glob
import time

np.set_printoptions(suppress=True, linewidth=160)


@jit(nopython=True)
def to_global(Qs, frames, Rs):
    N = len(frames)
    out = empty((N,3), np.float32)

    for i in range(N):
        out[i] = dot( Rs[frames[i]], Qs[i] )

    return out


@jit(nopython=True)
def find_close_rays(Cs, Qs, frames, ind, eps=1e-3, min_angle=np.pi/6, min_depth=0.1, max_depth=np.inf, cone=False):
    c = Cs[ind]
    q = Qs[ind]
    frame = frames[ind]

    N = Cs.shape[0]

    sel = empty(N, np.int64)
    j = 0

    for i in range(N):
        if frames[i] == frame:
            continue

        c2 = Cs[i]
        q2 = Qs[i]

        dist, depth = Geom.line_distance_3d(c, q, c2, q2)
        if (not cone and dist > eps) or (cone and dist*depth > eps) or not (0.1 < depth < max_depth):
            continue

        # # Want to check if it's close enough to plane defined as
        # # tangent to q and n
        # dc = (c2 - c)
        # dc -= dc * dot(dc, q)  # Remove q component
        # mag = norm3(dc)
        # if mag == 0:
        #     # Other view lies on the same ray
        #     continue

        # dc /= mag

        # if abs(dot(dc, n)) < min_cos:
        #     continue

        sel[j] = i
        j += 1

    return sel[:j]


@jit(nopython=True)
def find_in_cone(cone_center, cone_ray, cone_radius, Cs, Qs):
    N = Cs.shape[0]
    sel = empty(N, np.int64)
    j = 0

    for i in range(N):
        in_cone = Geom.check_intersect_cone_ray(cone_center, cone_ray, cone_radius, Cs[i], Qs[i])
        if in_cone:
            sel[j] = i
            j += 1

    return sel[:j]


@jit(nopython=True)
def find_intersections(Cs, Qs, plucker, ray_ind, plane, sel):
    '''
    Cs, Qs, plucker: Canonical and Plucker representations of set of rays
    ray_ind: Specific inds of the rays/ray triangle that's being intersected
    plane: Plane describing the ray triangle
    sel: Subset of rays to check
    '''
    N = Cs.shape[0]

    pl1 = plucker[ray_ind: ray_ind+2]

    intersects = empty(N, np.int64)
    ps = empty((N,3), Cs.dtype)
    us = empty((N,3), Cs.dtype)
    j = 0

    for k, i in enumerate(sel):
        pl2 = plucker[i: i+2]
        skew = empty(4, np.int16)
        for a in range(2):
            for b in range(2):
                skew[2*a + b] = np.sign( Geom.check_plucker(pl1[a], pl2[b]) )
        if abs(skew.sum()) == 4:
            # Both rays skew on the same side of the other 2 rays
            continue

        # Find the intersection of the rays with the plane
        c1, c2 = Cs[i], Cs[i+1]
        q1, q2 = Qs[i], Qs[i+1]
        t1, int1 = Geom.intersect_line_plane(c1, q1, plane)
        t2, int2 = Geom.intersect_line_plane(c2, q2, plane)
        if not int1 or not int2:
            continue

        intersects[j] = k
        ps[j] = c1 + t1*q1
        us[j] = (c2 + t2*q2) - ps[j]
        j += 1

    return ps[:j], us[:j], intersects[:j]


# @jit(nopython=True)
def cache_segment(Cs, Qs, planes, pluckers, frames, labels, raygrid, inds, eps):
    starts = []
    deltas = []
    depths = []
    sideps = []
    sideds = []
    ray_frames = []
    planar_angles = []
    sel_inds = []
    j = 0

    M = len(inds)
    for ind in range(M):
        ray_ind = inds[ind]
        j += 1
        print(j)

        print("Checking ray_ind:", ray_ind)
        # TODO: Check that this is still working correctly for the triangles
        in_cone = raygrid.rays_near_ind(ray_ind)

        print("Near: {}/{}".format(len(in_cone), raygrid.n_rays))

        if len(in_cone) == 0:
            print("Total grid for ray:", raygrid.ray2vox_count[ray_ind])

        center = Cs[ray_ind]
        ray = Qs[ray_ind: ray_ind+2].mean(axis=0)
        ray /= Geom.norm3(ray)
        plane = planes[ray_ind]
        normal = plane[:3]
        tangent = cross3(ray, normal)

        print("Intersect planes")
        # TODO: Use Plucker coords to quickly calculate intersection dual ray segments (infinite triangles?)
        # Will need labels. Precalc Plucker before (do same as the planes)
        # ps -> us, should describe how the other segment intersects this plane
        # in terms of it's width
        ps, us, intersecting = find_intersections(Cs, Qs, pluckers, ray_ind, plane, in_cone)

        print("Project")
        R = empty((3,3), ps.dtype)
        R[0] = ray
        R[1] = tangent
        R[2] = normal

        ps2d = dot(R[:2], (ps - center).T)
        us2d = dot(R[:2], us.T)

        # Solve for the intersection with the center of the segment
        # [x, 0] = p + t*u
        us2d[1, us2d[1] == 0] = 1e-10
        ts = -ps2d[1] / us2d[1]
        ds = ps2d[0] + ts*us2d[0]

        # Keep only lines that are more vertical
        # vert = np.abs(us2d[0] / (np.abs(us2d[1]) + 1e-12)) < 0.4
        crossing = ps2d[1] * (ps2d[1] + us2d[1]) < 0
        vert = np.arctan(us2d[0] / us2d[1]) < 0.85
        # near = np.abs(ps2d[1]) < eps*ps2d[0]
        forward = (ds > 0.2) & (ds < 1e6)

        intersecting = in_cone[intersecting]
        centers = dot(R, (Cs[intersecting] - center).T)
        sidep = empty((2, len(intersecting)), Cs.dtype)
        sidep[0] = centers[0]
        sidep[1] = centers[2]

        rays = empty((3, len(intersecting)), Cs.dtype)
        rays[0] = ds - centers[0]
        rays[1] = -centers[2]
        rays[2] = -centers[1]
        # rays = dot(R, Qs[intersecting].T)
        sided = np.empty_like(sidep)
        sided[0] = rays[0]
        sided[1] = rays[1]

        # hori = np.abs(sided[1] / sided[0]) < 1.5*eps
        sel = np.where(forward & crossing)[0]

        ps2d = ps2d[:,sel]
        us2d = us2d[:,sel]

        starts.append( ps2d )
        deltas.append( us2d )
        depths.append( ds[sel] )
        ray_frames.append( frames[ intersecting[sel] ] )
        sideps.append( sidep[:, sel] )
        sideds.append( 1.5*sided[:, sel] )

        planar_angles.append( np.arctan(centers[1, sel]/centers[2, sel]) )
        # ray_dot = rays[1, sel] / norm(rays[:,sel], axis=0)
        # planar_angles.append( np.arctan(ray_dot) )

        sel_inds.append(intersecting[sel])

    return starts, deltas, depths, ray_frames, sideps, sideds, sel_inds, planar_angles


def visualize_segments(seq):
    edge_dir = os.path.join(seq, 'edges', '*.jpg')
    paths = glob.glob(edge_dir)

    fig = pynutmeg.figure('segments', 'figs/segments.qml')
    fig.set_gui('figs/segments_gui.qml')

    fig.set('ax.im', xOffset=-0.5, yOffset=-0.5)

    nextframe = fig.parameter('nextframe')
    nextframe.wait_changed()
    prv = fig.parameter('prev')
    nxt = fig.parameter('next')

    for p in paths[::50]:
        print("\nReading in {}".format(p))
        E = IO.imread(p)
        pts, labels = Contours.find_contours_edge(E, low=30, high=50, min_length=10)
        J, I = pts.T

        show = np.empty_like(E)
        show[:] = 255
        show[I, J] = 255 - E[I, J]

        fig.set('ax.im', binary=show)

        label = 0
        x = empty(0)
        y = empty(0)

        while True:
            if nextframe.changed:
                nextframe.read()
                break

            label_changed = nxt.changed or prv.changed
            if nxt.changed:
                label += 1
            elif prv.changed:
                label = max(0, label - 1)

            if label_changed:
                print("Calc for label {}".format(label))
                prv.read()
                nxt.read()

                sel = np.where(labels == label)[0]
                x = J[sel].astype(float)
                y = I[sel].astype(float)

                fig.set('ax.P0', x=x, y=y)
                fig.set('ax.P1', x=x[0:1], y=y[0:1])

            time.sleep(0.005)


def visualize_intersections(seq, sub, skip=20, imtype='png'):
    edge_dir = os.path.join(seq, 'edges', '*.'+imtype)
    paths = glob.glob(edge_dir)

    voxel_dir = os.path.join(seq, sub + '_voxel')

    # ---------- Set up the figure -----------
    fig = pynutmeg.figure('segments', 'figs/intersection.qml')
    fig.set_gui('figs/intersection_gui.qml')

    # Parameters
    sld_frame = fig.parameter('frame')
    sld_frameoffset = fig.parameter('frameoffset')
    sld_segment = fig.parameter('segment')
    sld_index = fig.parameter('index')
    sld_anglesupport = fig.parameter('anglesupport')
    sld_planarsupport = fig.parameter('planarsupport')
    sld_support = fig.parameter('framesupport')
    btn_cache = fig.parameter('cachebtn')
    btn_export = fig.parameter('exportbtn')
    btn_cache.wait_changed(5)
    btn_export.wait_changed(1)

    # ------------- Load in data -------------
    print("Loading rays")
    cloud = RayCloud.load( os.path.join(seq, sub) )

    F = int(cloud.frames.max())
    sld_frame.set(maximumValue=F-1, stepSize=skip)
    sld_support.set(maximumValue=1000, stepSize=skip)

    N = cloud.N
    Cs, Qs, Ns = cloud.global_rays()

    planes = empty((N,4), Qs.dtype)
    planes[:,:3] = Ns
    planes[:,3] = -(Cs*Ns).sum(axis=1)

    plucker = Geom.to_plucker(Cs, Qs)

    # Get a rough scale for closeness threshold based on
    # size of camera center bounding box
    print("Loading voxels")
    raygrid = RayVoxel.load(voxel_dir)

    # longest = (bbox_max - bbox_min).max()
    # eps = longest * 1e-2
    eps = 1/cloud.cam[0]
    print("Eps:", eps)

    # Load the cam so we can offset the image properly
    fx, fy, cx, cy = cloud.cam
    # Make image show in homogenious coords
    fig.set('ax.im', xOffset=-(cx+0.5)/fx, yOffset=-(cy+0.5)/fy, xScale=1/fx, yScale=1/fy)
    fig.set('fit', minX=0.4, maxX=1, minY=-1, maxY=1)

    # Make sure the figure's online
    pynutmeg.wait_for_nutmeg()
    pynutmeg.check_errors()

    # Init state vars
    frame = 0
    label = 1
    index = 0
    frame_offset = 0

    labels = empty(0, int)
    max_label = 1
    max_ind = 0
    s, e = 0, 0
    ray_ind = 0

    frame_changed = True

    cache = [ [], [], [] ]
    validcache = False
    cachechanged = False

    cluster_sel = empty(0, int)

    hough = zeros((500,1000), np.uint32)

    while True:
        # Check parameter update
        if sld_frame.changed:
            frame = max(0, sld_frame.read())
            frame_changed = True

        if sld_segment.changed:
            label = sld_segment.read()
            segment_changed = True

        if sld_index.changed:
            index = sld_index.read()
            index_changed = True

        # Apply updated values
        if frame_changed:
            E = IO.imread( paths[frame] )
            fig.set('ax.im', binary=255-E)

            s, e = cloud.frame_range[frame]
            labels = cloud.labels_frame[s:e]
            if len(labels) > 0:
                max_label = labels.max()
                sld_segment.set(maximumValue=int(max_label))
            else:
                sld_segment.set(maximumValue=0)

            label = 0
            segment_changed = True
            frame_changed = False

        if segment_changed:
            segment_inds = s + np.where(labels == label)[0]

            max_ind = max(0, len(segment_inds))
            sld_index.set(maximumValue=max_ind)
            if len(segment_inds) > 0:
                P_seg = cloud.local_rays[np.r_[segment_inds, segment_inds[-1] + 1]].T
                fig.set('ax.P0', x=P_seg[0], y=P_seg[1])
            else:
                fig.set('ax.P0', x=[], y=[])
                fig.set('ax.P1', x=[], y=[])
                fig.set('ax.rays', x=[], y=[])

            index = min(max_ind, index)

            index_changed = True
            segment_changed = False
            validcache = False

        # if sld_frameoffset.changed:
        #     print("Recalculation frame offset...")
        #     frame_offset = sld_frameoffset.read()

        #     # Slow, but don't care, atm...
        #     Cs, Qs, Ns = cloud.global_rays(frame_offset)
        #     planes = empty((N,4), Qs.dtype)
        #     planes[:,:3] = Ns
        #     planes[:,3] = -(Cs*Ns).sum(axis=1)
        #     plucker = Geom.to_plucker(Cs, Qs)
        #     print("Done")

        #     validcache = False

        if index_changed and index >= 0 and index < len(segment_inds):
            ray_ind = segment_inds[index]

            P_seg = cloud.local_rays[ray_ind: ray_ind+2]
            P_ind = P_seg.mean(axis=0).reshape(-1,1)
            fig.set('ax.P1', x=P_ind[0], y=P_ind[1])

            tx, ty, _ = P_seg[1] - P_seg[0]
            mag = sqrt(tx*tx + ty*ty)
            # nx, ny = Geom.project_normal(q, cloud.local_normals[ray_ind])
            nx, ny = -ty/mag*3e-2, tx/mag*3e-2
            L = empty((2,2))
            L[:,0] = P_ind[:2,0]
            L[:,1] = L[:,0] + (nx, ny)
            fig.set('ax.rays', x=L[0], y=L[1])

        if (index_changed or
                sld_support.changed or sld_anglesupport.changed or
                sld_planarsupport.changed or cachechanged) and validcache:
            frame_support = max(sld_support.read(), 2*sld_support.read() - frame)
            angle_support = sld_anglesupport.read()/10000
            planarsupport = sld_planarsupport.read()/1000
            cachechanged = False
            # print("Cache: {}, Index: {}".format(len(cache[0]), index))
            if len(cache[0]) > 0 and 0 <= index < len(cache[0]):
                frames = cache[3][index]
                df = frames - frame
                planar_angles = cache[7][index]
                keep = np.where( (np.abs(df) <= frame_support) | (np.abs(planar_angles) <= planarsupport) )[0]

                P = cache[0][index][:,keep]
                deltas = cache[1][index][:,keep]
                depths = cache[2][index][keep]
                # angles = np.arctan(deltas[0]/deltas[1])
                angleres = np.deg2rad(5)
                centers = cache[4][index][:,keep]

                rays2d = cache[5][index][:,keep]
                rays2d /= norm(rays2d, axis=0)

                print("Sel:", len(cache[6][index]))

                print("Clustering")
                # cluster_inds, radius, depth, ray_angles, inlier_frac = ClassifyEdges.find_cluster_line3(
                #     deltas, rays2d, depths,
                #     angle_support, res=(angleres, 1e-2),
                #     thresh=(np.deg2rad(10), 4e-3))

                result = ClassifyEdges.find_cluster_line4(centers, rays2d, depth_thresh=angle_support, percentile=0.15)
                cluster_inds, radius, depth, dual_centers, dual_rays, inlier_frac = result
                print(".. Done")

                # np.savez('tmp/frame_cluster/ray_{}.npz'.format(ray_ind), frames=frames-frame, depths=depths, angles=angles, cluster_inds=cluster_inds)
                # print("Saved cluster", ray_ind)

                cluster_sel = cache[6][index][keep][cluster_inds]
                # fig.set('fit.P0', x=depths, y=ray_angles)
                # fig.set('fit.P1', x=depths[cluster_inds], y=ray_angles[cluster_inds])
                # fig.set('fit.P2', x=depths[line_cluster], y=ray_angles[line_cluster])

                x1, y1 = dual_centers - 10*dual_rays
                x2, y2 = dual_centers + 10*dual_rays
                fig.set('fit.rays', x=x1, y=y1, endX=x2, endY=y2)
                fig.set('fit.rays2', x=x1[cluster_inds], y=y1[cluster_inds], endX=x2[cluster_inds], endY=y2[cluster_inds])
                # fig.set('fit.rays3', x=x1[line_cluster], y=y1[line_cluster], endX=x2[line_cluster], endY=y2[line_cluster])

                print(cluster_sel.shape)
                # c_out = Cs[cluster_sel]
                # q_out = (Cs[ray_ind] + Qs[ray_ind] * depths[cluster_inds].reshape(-1,1)) - c_out
                # verts = np.vstack((c_out, c_out + 1.2*q_out))
                # edges = empty((len(verts)//2, 2), np.uint32)
                # edges[:] = np.r_[0:len(verts)].reshape(2,-1).T
                # IO.save_point_cloud("tmp/cluster_rays.ply", verts, edges)

                # fig.set('fit.P2', x=depths[init_inds], y=ray_space[init_inds])

                if len(cluster_inds) >= 10:
                    # hist *= (0.04/hist.max())
                    # fig.set('tangent.l1', x=histx, y=hist)

                    # fig.set('fit.P0', x=angles, y=depths)
                    # fig.set('fit.P1', x=angles[cluster_inds], y=depths[cluster_inds])

                    P = P[:, cluster_inds]
                    deltas = deltas[:, cluster_inds]

                    x1, y1 = P
                    x2, y2 = P + deltas

                    fig.set('tangent.rays', x=x1, y=y1, endX=x2, endY=y2)
                    fig.set('tangent.l0', x=[0.0, 1.5], y=[0.0, 0.0])
                    fig.set('tangent.P0', x=depths, y=zeros(len(depths)))

                    # Determine tangent angle
                    intersect_angles = np.arctan(-deltas[0]/deltas[1])
                    # Zero is verticle
                    alpha = np.median(intersect_angles)
                    qa = np.r_[-np.sin(alpha), np.cos(alpha)]
                    a1 = np.r_[depth, 0] + 0.05*qa
                    a2 = np.r_[depth, 0] - 0.05*qa
                    fig.set('tangent.l1', x=[a1[0], a2[0]], y=[a1[1], a2[1]])

                    # Draw the other axis
                    Q2 = 2*rays2d[:, cluster_inds]
                    P2a = centers[:, cluster_inds]
                    P2b = P2a + Q2
                    fig.set('normal.rays', x=P2a[0], y=P2a[1], endX=P2b[0], endY=P2b[1])
                    fig.set('normal.l0', x=[0.0, 1.5], y=[0.0, 0.0])

                    # fig.set('fit.P0', x=angles2, y=depths)
                    # fig.set('fit.P1', x=angles2[cluster_inds], y=depths[cluster_inds])
                    # np.savez('tmp/circle3.npz', P=P2a, Q=Q2, eps=eps)

                    # depth_std = np.std(depths[cluster_inds])

                    # nearby = np.where( np.abs(ray_angles[cluster_inds]) < 1.5*angle_support )[0]

                    # maxangle = np.percentile(np.abs(ray_angles[cluster_inds]), 95)
                    print("Radius:", radius, depth)
                    # frac = len(cluster_inds)/len(depths)
                    print("Frac:", len(cluster_inds), inlier_frac)
                    # print("Nearby:", len(nearby))
                    # if angle_range > np.deg2rad(20):
                    #     print("Hard edge", len(cluster_inds))
                    if inlier_frac > 0.05 and len(cluster_inds) > 100:
                        if abs(radius) < 1e-2:
                            print("Hard edge", len(cluster_inds))
                        else:
                            print("Occlusion", len(cluster_inds))

                else:
                    print("Cluster too small:", len(cluster_inds))

        index_changed = False

        if btn_cache.read_changed() or not validcache:
            if len(segment_inds) > 0 and label != 0:
                print("Caching segment... Total indices: {}".format(len(segment_inds)), flush=True)
                cache = cache_segment(
                    Cs, Qs, planes, plucker,
                    cloud.frames, cloud.labels_frame, raygrid, segment_inds, eps=eps)
                validcache = True
                cachechanged = True
                print("Done")

                # TODO: Output cluster segments to .ply for blender visualization.......

        if btn_export.read_changed() and validcache and len(cluster_sel) > 0:
            export_triangles('tmp/cluster_tris.ply', Cs, Qs*1.5, cluster_sel, ray_ind)

            c_out = Cs[cluster_sel]
            q_out = (Cs[ray_ind] + Qs[ray_ind] * depths[cluster_inds].reshape(-1,1)) - c_out
            verts = np.vstack((c_out, c_out + 1.5*q_out))
            edges = empty((len(verts)//2, 2), np.uint32)
            edges[:] = np.r_[0:len(verts)].reshape(2,-1).T
            IO.save_point_cloud("tmp/cluster_rays.ply", verts, edges)
            print("Saved .ply")

        time.sleep(0.005)


def visualize_soup(seq):
    # R, t = IO.read_Rt(os.path.join(seq, 'lsdslam_Rt.npz'))
    # print("Loading tracks...")
    # raytracks = RayTracks(seq=seq, sub='fast_edges_4')

    # Rw, tw = Geom.global_to_local(R, t)

    # # To single precision makes it a faster load to Nutmeg
    # frames = raytracks.data.frame[::10000]
    # P1 = tw.astype(np.float32)[frames]
    # print("Transforming data", P1.shape)
    # P2 = to_global(raytracks.data.q, frames, Rw.astype(np.float32))

    # P1 -= P1.mean(axis=0)
    # P2 -= P2.mean(axis=0)

    # np.savez("tmp/raysoup.npz", P1=P1, P2=P2)

    npz = np.load("tmp/raysoup.npz")
    P1 = npz['P1']
    P2 = npz['P2']

    # TODO: Read pear LSD point cloud. Show that instead of rays
    # Show plane
    # Point selected points on 2D slice too

    minz = min(P1[:,2].min(), P2[:,2].min())
    maxz = max(P1[:,2].max(), P2[:,2].max())
    dz = maxz - minz

    print(P1.shape, P2.shape)
    print("Z: [{}, {}]".format(minz, maxz))

    fig = pynutmeg.figure('raysoup', 'figs/raysoup.qml')
    fig.set_gui('figs/raysoup_gui.qml')
    planez = fig.parameter('planez')
    eps = fig.parameter('eps')

    fig.set('ax3d.rays', start=P1, end=P2)
    fig.set('ax', minX=-1, maxX=1, minY=-1, maxY=1)

    while True:
        if planez.changed or eps.changed:
            print("PlaneZ changed")
            epz = eps.read() * dz
            z = planez.read() * dz + minz
            d1 = np.abs(P1[:,2] - z)
            d2 = np.abs(P2[:,2] - z)

            sel = np.where((d1 < epz) & (d2 < epz))[0]
            print("Sel:", sel.shape)

            p1 = P1[sel, :2].T
            p2 = P2[sel, :2].T
            fig.set('ax.rays', x=p1[0], y=p1[1], endX=p2[0], endY=p2[1])

        pynutmeg.check_errors()
        time.sleep(0.01)


def export_triangles(path, Cs, Qs, inds, ray_ind):
    F = len(inds)
    V = 3*F

    verts = zeros((V,3), np.float32)
    faces = zeros((F,3), np.uint32)

    for j, i in enumerate(inds):
        a, b, c = 3*j, 3*j + 1, 3*j + 2
        faces[j] = a, b, c

        v1, v2 = Qs[i], Qs[i+1]
        verts[a] = Cs[i]
        verts[b] = Cs[i] + 2*v1
        verts[c] = Cs[i] + 2*v2

    IO.write_ply(path, verts=verts, faces=faces)

    vert3 = zeros((3,3), np.float32)
    face3 = zeros((1,3), np.uint32)
    v1, v2 = Qs[ray_ind], Qs[ray_ind + 1]
    vert3[0] = Cs[ray_ind]
    vert3[1] = Cs[ray_ind] + 2*v1
    vert3[2] = Cs[ray_ind] + 2*v2
    face3[:] = 0,1,2

    IO.write_ply('tmp/cluster_main.ply', verts=vert3, faces=face3)


if __name__ == '__main__':
    import sys
    # seq = '../data/hat_01'
    # seq = '../data/pear_01'
    # seq = '../data/ship_01'
    seq = '../data/glass_render'
    # seq = '../data/bottle_04'
    # seq = '../data/ch_bottle_01'
    imtype = 'png'
    # visualize_soup(seq)
    # visualize_segments(seq)

    if len(sys.argv) > 1:
        seq = os.path.join('../data', sys.argv[1])

    sub = 'rays_1'
    skip = 1

    # extract_rays(seq, sub=sub, skip=skip, imtype=imtype)
    # build_voxel_grid(seq, sub)

    visualize_intersections(seq, sub=sub, skip=skip, imtype=imtype)
