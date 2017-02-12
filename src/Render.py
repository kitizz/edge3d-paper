import numpy as np
from numpy import empty, dot, cos, sin, sqrt, ones
from numpy.linalg import norm

from numba import jit, jitclass, float32, int64, uint16, uint32

from subprocess import run
import sys

from tools import Geom, Drawing, Util, IO
from tools.FastMath import cross2

import pynutmeg

_spec = [
    ('verts', float32[:, :]),
    ('normals', float32[:, :]),
    ('edges', uint32[:, :]),
    ('faces', uint32[:, :]),
    ('frames', uint32[:]),
    ('inds', uint32[:]),
    # ('rendertype', uint32),
]

RENDER_FACES = 1
RENDER_EDGES = 2

FLAT = 0
PHONG = 1
NORMAL = 2


@jitclass(_spec)
class Mesh(object):
    def __init__(self):
        self.verts = empty((0, 3), np.float32)
        self.normals = empty((0, 3), np.float32)
        self.edges = empty((0, 2), np.uint32)
        self.faces = empty((0, 3), np.uint32)
        self.frames = empty(0, np.uint32)
        self.inds = empty(0, np.uint32)
        # self.rendertype = RENDER_FACES

    def project_verts(self, R, t, cam):
        if self.verts.shape[0] == 0:
            return empty((0, 3), np.float32)

        fx, fy, cx, cy = cam
        v = dot(R, self.verts.T) + t.reshape(-1, 1)
        v[:2] /= v[2]
        v[0] *= fx
        v[0] += cx
        v[1] *= fy
        v[1] += cy
        return v.T


def save_mesh(path, mesh):
    np.savez(path, verts=mesh.verts, edges=mesh.edges, normals=mesh.normals, faces=mesh.faces)


def load_mesh(path):
    npz = np.load(path)
    mesh = Mesh()
    mesh.verts = npz['verts']
    mesh.edges = npz['edges']
    mesh.normals = npz['normals']
    mesh.faces = npz['faces']
    return mesh


def from_ply(path):
    verts, faces = IO.read_ply(path)
    mesh = Mesh()
    mesh.verts = verts.astype(np.float32)
    mesh.faces = faces.astype(np.uint32)
    return mesh


def render_frame(im, object_mesh, occlusion_mesh, poly_mesh, R, t, cam, light=None, color=None, poly_color=None):
    if light is None:
        # lazi, lalt = np.deg2rad(30), np.deg2rad(60)
        lazi, lalt = np.deg2rad(30), np.deg2rad(45)
        light = np.r_[cos(lazi)*cos(lalt), sin(lazi)*cos(lalt), sin(lalt)].astype(np.float32)
        light /= Geom.norm3(light)

        light = dot(R.T, light).astype(np.float64)
        # light = light.astype(np.float64)
    #     raise Exception("TODO: Light")

    view = -dot(R.T, t)

    pts_obj = object_mesh.project_verts(R, t, cam)

    obj_map = np.full(im.shape[:2], np.inf, np.float32)
    render_depth(obj_map, pts_obj, object_mesh)

    # --- Buffer occlusion lines ----
    if color is None:
        color = np.r_[0, 155, 255]
    pts_occl = occlusion_mesh.project_verts(R, t, cam)
    norm_proj = dot(occlusion_mesh.normals, R.T)

    if len(occlusion_mesh.edges) == 0:
        line_buf = buffer_points(pts_occl, color, alpha=0.1, size=3)
    else:
        line_buf = buffer_lines(pts_occl, norm_proj, occlusion_mesh, color, view, light, alpha=0.1, shade_method=NORMAL)

    if poly_mesh is not None:
        if poly_color is None:
            poly_color = np.r_[255, 166, 0]
        pts_poly = poly_mesh.project_verts(R, t, cam)
        norm_poly = dot(poly_mesh.normals, R.T)

        poly_buf = buffer_lines(pts_poly, norm_poly, poly_mesh, poly_color, view, light, linewidth=4, shade_method=FLAT)
        poly_buf[:, 2] = 0.001
    else:
        poly_buf = np.empty((0, line_buf.shape[1]), line_buf.dtype)

    # Make buffer out of object depth map, and append to line buffer
    I, J = np.where(obj_map != np.inf)
    depths = 1.02*obj_map[I, J]
    object_buf = empty((len(I), line_buf.shape[1]), line_buf.dtype)
    object_buf[:, 0] = I
    object_buf[:, 1] = J
    object_buf[:, 2] = depths
    object_buf[:, 3] = 0.4
    object_buf[:, 4:] = 255, 255, 255

    final_buf = np.vstack((line_buf, poly_buf, object_buf))

    render_buffer(im, final_buf, bg=255)


@jit(nopython=True)
def phong_shade(color, L, N, V, ks=0.8, kd=1.0, ka=0.5, shiny=0.3):
    R = 2*dot(L, N)*N - L
    rgb = ka*color + kd*abs(dot(L, N))*color + ks*abs(dot(R, V))**shiny*np.ones(3)
    threshold = 255.999
    m = rgb.max()
    if m <= threshold:
        return np.floor(rgb)
    total = rgb.sum()
    if total >= 3 * threshold:
        rgb[:] = np.floor(threshold)
        return rgb
    x = (3 * threshold - total) / (3 * m - total)
    gray = threshold - x * m
    return np.floor(gray + x * rgb)


@jit(nopython=True)
def normal_shade(light, normal_g, normal_l):
    out = empty(3)

    # intensity = abs(dot(light, normal_g)) + 0.5
    intensity = 1
    sgn = 1
    if normal_l[2] < 0:
        sgn = -1
    # sgn = np.signbit(normal_l[2])

    red = (-sgn * normal_l[0] + 1) * 127.5
    green = (sgn * normal_l[1] + 1) * 127.5
    blue = abs(normal_l[2]) * 127 + 128

    out[0] = min(max(red * intensity, 0), 255)
    out[1] = min(max(green * intensity, 0), 255)
    out[2] = min(max(blue * intensity, 0), 255)

    return out


@jit(nopython=True)
def buffer_points(pts, color, alpha, size=1):
    N = len(pts)

    M = size*size
    P = N * size * size

    out = empty((P, 7), np.float32)

    for n in range(N):
        p = pts[n]
        z = p[2]
        out_ind = M*n

        sz = size//2

        for dx in range(-sz, size - sz):
            for dy in range(-sz, size - sz):
                x = p[0] + dx
                y = p[1] + dy

                # print(x, y)
                out[out_ind, 0] = round(y)
                out[out_ind, 1] = round(x)
                out[out_ind, 2] = z
                out[out_ind, 3] = alpha
                out[out_ind, 4:] = color

                out_ind += 1

    return out


@jit(nopython=True)
def buffer_lines(pts, norms_proj, mesh, color, view, light, alpha=1.0, shade_method=PHONG, linewidth=1):
    '''
    Use the depth buffer, dbuf, to figure out whether the object needs to be "washed out"
    when drawing.
    Use the normal information to make clear the normal direction of the line.
    '''
    edges = mesh.edges
    normals = mesh.normals
    E = edges.shape[0]

    # Output: [i, j, d, alpha, r, g, b]
    size = 1024
    P = 7
    out = empty((size, P), np.float32)
    N = 0

    for e in range(E):
        edge = edges[e]
        x1, y1, z1 = pts[edge[0]]
        x2, y2, z2 = pts[edge[1]]

        view_local = (view - mesh.verts[edge[0]]).astype(np.float64)

        # n1, n2 = normals[edge[0]], normals[edge[1]]
        normal = normals[e].astype(np.float64)
        normal_l = norms_proj[e].astype(np.float64)
        normal_l /= Geom.norm3(normal_l)
        # light_angle = abs(dot(normal, light))*1.5 + 0.3
        # print(light_angle)

        I, J, a = Drawing.line_aa(x1, y1, x2, y2, linewidth)
        M = len(I)
        depths = np.linspace(z1, z2, M)

        # Resize if necessary
        resize = False
        while size < N + M:
            size *= 2
            resize = True
        if resize:
            newout = empty((size, P), np.float32)
            newout[:N] = out[:N]
            out = newout

        if shade_method == PHONG:
            out_color = phong_shade(color, light, normal, view_local)
        elif shade_method == NORMAL:
            out_color = normal_shade(light, normal, normal_l)
        else:
            out_color = color.astype(np.float64)

        # Copy it in
        for i, n in enumerate(range(N, N + M)):
            out[n, 0] = I[i]
            out[n, 1] = J[i]
            out[n, 2] = depths[i]
            out[n, 3] = a[i] * alpha
            out[n, 4:] = out_color

        N += M

    return out[:N]


def draw_normal_map(path, size):
    '''
    Draw a spherical normal map for visualizations
    '''
    # White backdrop
    im = np.full((size, size, 3), 255, np.uint8)
    center = size//2

    light = np.r_[0,0,1.0]
    ng = np.r_[0,0,1.0]

    for i in range(size):
        di = i - center
        for j in range(size):
            dj = j - center
            r = sqrt(di*di + dj*dj)
            if r > center:
                continue

            r_norm = r/center
            phi = np.arccos(r_norm)
            theta = -np.arctan2(dj, di)
            nx = cos(theta)*cos(phi)
            ny = sin(theta)*cos(phi)
            nz = sin(phi)
            mag = sqrt(nx*nx + ny*ny + nz*nz)
            normal = np.r_[nx/mag, -ny/mag, nz/mag]

            color = normal_shade(light, ng, normal)

            im[i,j] = color

    IO.imshow(im)
    pynutmeg.wait_for_nutmeg()

    IO.imwrite(path, im)

    return im


@jit(nopython=True)
def _render_buffer_jit(im, buf, inds, bg):
    N = len(inds)
    H = im.shape[0]
    W = im.shape[1]
    D = im.shape[2]

    n = 0
    while n < N:
        ind = inds[n]
        i0, j0 = int(buf[ind, 0]), int(buf[ind, 1])
        if i0 < 0 or i0 >= H or j0 < 0 or j0 >= W:
            n += 1
            continue

        i, j = i0, j0

        color = empty(D, buf.dtype)
        color[:] = im[i0, j0]

        while n < N and i == i0 and j == j0:
            alpha = buf[ind, 3]
            rgb = buf[ind, 4:]
            color[:] = (1 - alpha)*color + alpha*rgb

            n += 1
            if n < N:
                ind = inds[n]
                i, j = int(buf[ind, 0]), int(buf[ind, 1])

        im[i0, j0] = color


def render_buffer(im, buf, bg=1):
    '''
    Order by depth and mix alpha values at the same pixel

    buf: (Nx7) array with each element as [i, j, d, alpha, r, g, b]
    '''
    inds = Util.lex_order(buf[:, :3].T, maxs=(720, 1280, 20))[::-1]
    _render_buffer_jit(im, buf, inds, bg)


@jit(nopython=True)
def render_depth(im, pts_proj, mesh):
    '''
    pts_proj: (Nx3) array where each element is (x,y,z) projected
    '''
    faces = mesh.faces
    F = faces.shape[0]

    for f in range(F):
        i, j, k = faces[f]
        draw_tri(im, pts_proj[i], pts_proj[j], pts_proj[k])


@jit(nopython=True)
def fill_tri(im, x1, y1, x2, y2, i, j, a, b, c, ibxc):
    N = len(x1)
    M = len(x2)
    H = im.shape[0]
    W = im.shape[1]

    if i >= N:
        return i, j, y1[-1]

    y = y1[i]

    ab = b - a
    ac = c - a

    while i < N and j < M:
        y = y1[i]
        if y < 0:
            i += 1
            continue
        if y >= H:
            return i, j, y

        xmin = min(x1[i], x2[j])
        xmax = max(x1[i], x2[j])
        while j < M and y2[j] <= y:
            xmin = min(x2[j], xmin)
            xmax = max(x2[j], xmax)
            j += 1
        while i < N and y1[i] <= y:
            xmin = min(x1[i], xmin)
            xmax = max(x1[i], xmax)
            i += 1

        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W - 1, xmax))

        qx1 = xmin - a[0]
        qx2 = xmax - a[0]
        qy = y - a[1]
        s1 = (qx1*ac[1] - qy*ac[0]) * ibxc
        t1 = (qy*ab[0] - qx1*ab[1]) * ibxc
        u1 = 1 - s1 - t1
        s2 = (qx2*ac[1] - qy*ac[0]) * ibxc
        t2 = (qy*ab[0] - qx2*ab[1]) * ibxc
        u2 = 1 - s2 - t2

        d1 = u1*a[2] + s1*b[2] + t1*c[2]
        d2 = u2*a[2] + s2*b[2] + t2*c[2]
        # print(xmin, xmax)
        ds = np.linspace(d1, d2, xmax - xmin + 1)

        for n, x in enumerate(range(xmin, xmax + 1)):
            im[y, x] = min(im[y, x], ds[n])

    return i, j, y


@jit(nopython=True)
def draw_tri(im, a, b, c):
    # Work out the order of the triangle verts
    if a[1] < b[1]:
        if a[1] < c[1]:
            top = a
            if b[1] < c[1]:
                mid = b
                bot = c
            else:
                mid = c
                bot = b
        else:
            top = c
            mid = a
            bot = b
    else:
        if b[1] < c[1]:
            top = b
            if a[1] < c[1]:
                mid = a
                bot = c
            else:
                mid = c
                bot = a
        else:
            top = c
            mid = b
            bot = a
    # top = a
    # mid = b
    # bot = c
    # print("Top:", top)
    # print("Mid:", mid)
    # print("Bot:", bot)

    tm = mid - top
    tb = bot - top
    mxb = (tm[0]*tb[1] - tm[1]*tb[0])
    if mxb == 0:
        return
    imxb = 1/mxb

    # One side goes top->bot, the other goes via the mid
    y_tb, x_tb = Drawing.line(top[0], top[1], bot[0], bot[1])
    y_tm, x_tm = Drawing.line(top[0], top[1], mid[0], mid[1])
    y_mb, x_mb = Drawing.line(mid[0], mid[1], bot[0], bot[1])

    i, j, y = fill_tri(im, x_tb, y_tb, x_tm, y_tm, 0, 0, top, mid, bot, imxb)

    j = 0
    while j < len(y_mb) and y_mb[j] <= y:
        j += 1
    fill_tri(im, x_tb, y_tb, x_mb, y_mb, i, j, top, mid, bot, imxb)


@jit(nopython=True)
def draw_tri_bar(im, a, b, c):
    H = im.shape[0]
    W = im.shape[1]

    minx = min(a[0], min(b[0], c[0]))
    maxx = max(a[0], max(b[0], c[0]))
    miny = min(a[1], min(b[1], c[1]))
    maxy = max(a[1], max(b[1], c[1]))

    v1 = b - a
    v2 = c - a
    ivxv = 1/(v1[0]*v2[1] - v1[1]*v2[0])

    da = a[2]
    db = b[2]
    dc = c[2]

    for x in range(int(minx), int(np.ceil(maxx))):
        for y in range(int(miny), int(np.ceil(maxy))):
            qx = x - a[0]
            qy = y - a[1]
            s = (qx*v2[1] - qy*v2[0]) * ivxv
            t = (qy*v1[0] - qx*v1[1]) * ivxv

            if s < 0 or t < 0 or s + t > 1:
                continue

            if x >= 0 and y >= 0 and x < W and y < H:
                u = 1 - s - t
                d = s*db + t*dc + u*da
                im[y, x] = min(im[y, x], d)


def get_cam_mesh(l, w, h):
    rl = l/2
    rw = w/2

    verts = empty((6,3), np.float32)
    edges = empty((10,2), np.uint32)

    verts[0] = 0, 0, 0
    verts[1] = -rl, -rw, h
    verts[2] = rl, -rw, h
    verts[3] = -rl, rw, h
    verts[4] = rl, rw, h
    verts[5] = 0, -rw*1.5, h

    edges[0] = 0, 1
    edges[1] = 0, 2
    edges[2] = 0, 3
    edges[3] = 0, 4
    edges[4] = 1, 2
    edges[5] = 2, 4
    edges[6] = 4, 3
    edges[7] = 3, 1
    edges[8] = 1, 5
    edges[9] = 2, 5

    mesh = Mesh()
    mesh.verts = verts
    mesh.edges = edges

    return mesh


def get_cube_mesh(r):
    verts = empty((8, 3), np.float32)
    edges = empty((12, 2), np.uint32)
    faces = empty((12, 3), np.uint32)

    normals = empty((12, 3), np.float32)

    v = 0
    for z in (-r, r):
        for y in (-r, r):
            for x in (-r, r):
                verts[v] = x, y, z
                v += 1

    edges[0] = 0, 1
    edges[1] = 1, 3
    edges[2] = 3, 2
    edges[3] = 2, 0
    edges[4] = 0, 4
    edges[5] = 1, 5
    edges[6] = 2, 6
    edges[7] = 3, 7
    edges[8] = 4, 5
    edges[9] = 5, 7
    edges[10] = 7, 6
    edges[11] = 6, 4

    # Edge normals
    alpha = 1/sqrt(2)
    normals[0] = 0, -alpha, -alpha
    normals[1] = alpha, 0, -alpha
    normals[2] = 0, alpha, -alpha
    normals[3] = -alpha, 0, -alpha
    normals[4] = -alpha, -alpha, 0
    normals[5] = alpha, -alpha, 0
    normals[6] = alpha, alpha, 0
    normals[7] = -alpha, alpha, 0
    normals[8] = 0, -alpha, alpha
    normals[9] = alpha, 0, alpha
    normals[10] = 0, alpha, alpha
    normals[11] = -alpha, 0, alpha

    print(norm(normals, axis=1))

    # Bottom
    faces[0] = 0, 2, 1
    faces[1] = 3, 1, 2
    # X-
    faces[2] = 0, 4, 2
    faces[3] = 6, 2, 4
    # X+
    faces[4] = 1, 3, 5
    faces[5] = 7, 5, 3
    # Y-
    faces[6] = 0, 1, 4
    faces[7] = 5, 4, 1
    # Y+
    faces[8] = 2, 6, 3
    faces[9] = 7, 3, 6
    # Top
    faces[10] = 4, 5, 6
    faces[11] = 7, 6, 5

    mesh = Mesh()
    mesh.verts = verts
    mesh.edges = edges
    mesh.faces = faces
    mesh.normals = normals

    return mesh


def test_render_cube():
    import pynutmeg
    import time

    fig = pynutmeg.figure('cube', 'figs/imshow.qml')

    im = empty((720, 1280, 3), np.uint8)

    azi = 0
    alt = np.deg2rad(40)
    dist = 10
    cam = Geom.cam_params(29.97, 1280, 720, 35)  # Blender defaults
    cam = cam.astype(np.float32)

    mesh = get_cube_mesh(1)
    obj_mesh = Mesh()

    n = 0
    while True:
        n += 1
        azi += np.deg2rad(2)

        tw = dist * np.array([cos(alt)*cos(azi), cos(alt)*sin(azi), sin(alt)], np.float32)

        alpha = -np.pi/2 - alt
        beta = np.pi/2 + azi
        Rx = np.array([
            [1, 0, 0],
            [0, cos(alpha), -sin(alpha)],
            [0, sin(alpha), cos(alpha)]
        ], np.float32)
        Rz = np.array([
            [cos(beta), -sin(beta), 0],
            [sin(beta), cos(beta), 0],
            [0, 0, 1]
        ], np.float32)
        Rw = dot(Rz, Rx)

        t = -dot(Rw.T, tw)
        R = Rw.T

        im[:] = 255
        render_frame(im, obj_mesh, mesh, R, t, cam)

        time.sleep(0.005)

        fig.set('ax.im', binary=im)


def test_render_seq(seq, sub, dopoly):
    import pynutmeg
    import time
    import os
    import RayCloud

    fig = pynutmeg.figure('cube', 'figs/imshow.qml')

    W, H = 1920, 1080
    im = empty((H, W, 3), np.uint8)
    cam = Geom.cam_params(29.97, W, H, 35)  # Blender defaults
    cam = cam.astype(np.float32)

    print("Loading data")
    cloud = RayCloud.load(os.path.join(seq, sub))

    mesh_path = os.path.join(seq, 'occlusion_mesh.npz')
    occl_mesh = load_mesh(mesh_path)
    ply_path = os.path.join(seq, 'model.ply')
    if os.path.exists(ply_path):
        obj_mesh = from_ply(ply_path)
    else:
        obj_mesh = Mesh()

    if dopoly:
        poly_data = os.path.join(seq, sub, 'poly_data.npz')
        polynpz = np.load(poly_data)
        poly_mesh = Mesh()
        poly_mesh.verts = polynpz['verts'].astype(np.float32)
        poly_mesh.edges = polynpz['edges'].astype(np.uint32)
    else:
        poly_mesh = None

    pers_mesh_path = os.path.join(seq, 'persistent_mesh.npz')
    pers_mesh = load_mesh(pers_mesh_path)

    campath = os.path.join(seq, 'render.py')
    Rs, ts = IO.import_blender_cam(campath)

    render_out = os.path.join(seq, 'render')
    Util.try_mkdir(render_out)

    F = min(3, len(Rs))

    print("Loaded")

    for f in range(F):
        print("Frame:", f)
        # R = cloud.Rs[f]
        # t = cloud.ts[f]
        R = Rs[f].astype(np.float32)
        t = ts[f].astype(np.float32)

        im[:] = 255
        t0 = time.time()
        render_frame(im, obj_mesh, occl_mesh, None, R, t, cam)
        print("dt:", (time.time() - t0)*1000)

        # time.sleep(0.005)

        fig.set('ax.im', binary=im)
        out = os.path.join(render_out, 'frame_{}.png'.format(f))
        IO.imwrite(out, im)

        im[:] = 255
        t0 = time.time()
        render_frame(im, obj_mesh, pers_mesh, poly_mesh, R, t, cam, color=np.r_[255, 82, 82], poly_color=np.r_[0,0,0])
        print("dt:", (time.time() - t0)*1000)

        fig.set('ax.im', binary=im)
        out = os.path.join(render_out, 'frame_pers_{}.png'.format(f))
        IO.imwrite(out, im)


def render_animation(seq, sub, dopoly):
    import pynutmeg
    import time
    import os
    import RayCloud

    fig = pynutmeg.figure('cube', 'figs/imshow.qml')

    W, H = 1920, 1080
    im = empty((H, W, 3), np.uint8)
    cam = Geom.cam_params(29.97, W, H, 35)  # Blender defaults
    cam = cam.astype(np.float32)

    print("Loading data")
    cloud = RayCloud.load(os.path.join(seq, sub))

    datapath = os.path.join(seq, 'occlusion_data.npz')
    data = np.load(datapath)
    mesh_path = os.path.join(seq, 'occlusion_mesh.npz')
    occl_mesh = load_mesh(mesh_path)
    # occl_mesh = Mesh()
    # occl_mesh.verts = data['verts'].astype(np.float32)
    edges = data['edges'].astype(np.uint32)
    ply_path = os.path.join(seq, 'model.ply')
    if os.path.exists(ply_path):
        obj_mesh = from_ply(ply_path)
    else:
        obj_mesh = Mesh()

    if dopoly:
        poly_data = os.path.join(seq, sub, 'poly_data.npz')
        polynpz = np.load(poly_data)
        poly_mesh = Mesh()
        poly_mesh.verts = polynpz['verts'].astype(np.float32)
        poly_mesh.edges = polynpz['edges'].astype(np.uint32)
    else:
        poly_mesh = None

    # pers_mesh_path = os.path.join(seq, 'persistent_mesh.npz')
    # pers_mesh = load_mesh(pers_mesh_path)

    campath = os.path.join(seq, 'animation.py')
    Rs, ts = IO.import_blender_cam(campath)

    # Grab frame info
    inds = data['inds']
    frames = cloud.frames[inds]

    render_out = os.path.join(seq, 'animate')
    Util.try_mkdir(render_out)

    F = len(Rs)
    N = len(frames)

    print("Loaded", frames.max())

    end_frame = 0

    for f in range(0, F, 10):
        print("Frame:", f)
        # R = cloud.Rs[f]
        # t = cloud.ts[f]
        R = Rs[f].astype(np.float32)
        t = ts[f].astype(np.float32)

        while end_frame < N and frames[end_frame] < f:
            end_frame += 1

        occl_mesh.edges = edges[:end_frame]

        im[:] = 255
        if len(occl_mesh.edges) > 0:
            t0 = time.time()
            render_frame(im, obj_mesh, occl_mesh, None, R, t, cam)
            print("Render time: {} ms".format( int((time.time() - t0)*1000) ))

        # time.sleep(0.005)

        fig.set('ax.im', binary=im)
        out = os.path.join(render_out, 'frame_{:05d}.png'.format(f))
        IO.imwrite(out, im)


def render_lsd(seq):

    plypath = os.path.join(seq, 'render_me.ply')
    verts, edges = IO.read_ply(plypath)
    lsd_mesh = Mesh()
    lsd_mesh.verts = verts.astype(np.float32)

    campath = os.path.join(seq, 'render.py')
    Rs, ts = IO.import_blender_cam(campath)

    W, H = 1920, 1080
    im = empty((H, W, 3), np.uint8)
    cam = Geom.cam_params(29.97, W, H, 35)  # Blender defaults
    cam = cam.astype(np.float32)

    render_out = os.path.join(seq, 'render')
    Util.try_mkdir(render_out)

    im[:] = 255

    pts = lsd_mesh.project_verts(Rs[0].astype(np.float32), ts[0].astype(np.float32), cam)
    pt_buf = buffer_points(pts, np.r_[255, 82, 82], alpha=0.2, size=2)
    render_buffer(im, pt_buf, bg=255)

    out = os.path.join(render_out, 'frame_0.png')
    IO.imwrite(out, im)


def make_video(path, rotate=0):
    # def convert(inpath, outpath, rotate=0):
    outpath = path + '.mp4'
    cmd = ['ffmpeg', '-pattern_type', 'glob', '-i', '{}/*.png'.format(path)]
    if rotate == 180:
        cmd.extend( ['-vf', 'transpose=2,transpose=2'] )
    cmd.extend( ['-pix_fmt', 'yuv420p', outpath] )

    run(cmd, stdout=sys.stdout, check=True)


if __name__ == '__main__':
    seq = '../data/glass_render'
    sub = 'rays_2'

    dopoly = 0

    import sys
    import os
    if len(sys.argv) > 1:
        seq = os.path.join('../data', sys.argv[1])
        if len(sys.argv) > 2:
            dopoly = int(sys.argv[2])

    # test_render_cube()
    test_render_seq(seq, sub, dopoly)
    # render_lsd(seq)
    # render_animation(seq, sub, dopoly)
