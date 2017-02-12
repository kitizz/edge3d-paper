import numpy as np
from numpy import dot

import os
import glob
import skimage.io
import skimage.color as color

import plyfile

import pynutmeg

from subprocess import call
import json
import ruamel.yaml as yaml

from . import Geometry as Geom
from .import Quat

import time


# Remember the package directory to refer to helper files
_pkg_dir = os.path.dirname(os.path.abspath(__file__))


def imread(path):
    # raise Exception("Not Implemented")
    return skimage.io.imread(path)


def imwrite(path, im):
    skimage.io.imsave(path, im, quality=100)


_imshows = {}
def imshow_nutmeg(im, t=0, name='imshow', newfig=False, wait=None):
    if im.dtype == float:
        im = (im*255).astype(np.uint8)

    if name not in _imshows or newfig:
        print("Creating figure..")
        for i in range(2):
            _imshows[name] = pynutmeg.figure(name, 'figs/imshow.qml')
            _imshows[name].set_gui('figs/imshow_gui.qml')

    btn_next = _imshows[name].parameter('next')
    btn_skip = _imshows[name].parameter('skip')
    if wait is not None:
        value = btn_next.read()

    _imshows[name].set('ax.im.binary', im)

    skipped = False
    if wait is not None:
        t0 = time.time()
        while wait < 0 or time.time() - t0 < wait:
            if btn_next.changed or btn_skip.changed:
                break
        skipped = btn_skip.changed
        btn_skip.read()

    return skipped


def color_gradient(gx, gy, max_grad=None):
    H = gx.shape[0]
    W = gx.shape[1]

    hsv = np.empty((H,W,3))
    hsv[:,:,0] = (np.arctan2(gy, gx) % (2*np.pi)) * (1/(2*np.pi))
    mag = np.sqrt(gx*gx + gy*gy)
    if max_grad is not None:
        hsv[:,:,1] = np.clip(mag * (1/max_grad), 0, 1)
    else:
        hsv[:,:,1] = mag * (1/mag.max())

    hsv[:,:,2] = 1

    return hsv, color.hsv2rgb(hsv)


def imshow(im, name='imshow', reset=False, wait=None):
    return imshow_nutmeg(im, name=name, newfig=reset, wait=wait)


def import_blender_cam(path):
    poses = parse_cam_3d(path)

    euler = poses[:, :3]
    tw = poses[:, 3:]

    Rw = Geom.blender_to_R(euler)

    return Geom.global_to_local(Rw, tw)


def parse_cam_2d(path):
    with open(path, 'r') as file:
        thetas = []
        ts = []
        for line in file:
            if line.startswith('obj.location'):
                _, nums = line.split('=')
                x, y, z = [ float(num.strip()) for num in nums.split(',') ]
                ts.append( np.r_[x, y])

            elif line.startswith('obj.rotation_euler'):
                _, nums = line.split('=')
                x, y, z = [ float(num.strip()) for num in nums.split(',') ]
                thetas.append(z)

        N = len(thetas)
        if len(ts) != N:
            raise(Exception("Trouble parsing camera poses. Mismatch in lengths"))
        poses = np.empty((N,3))
        poses[:,0] = thetas
        poses[:,1:] = np.array(ts)

        return poses

    print("WARNING! No camera poses found...")
    return np.zeros((0,3))


def parse_cam_3d(path):
    with open(path, 'r') as file:
        angles = []
        ts = []
        l = 0
        for line in file:
            l += 1
            try:
                if line.startswith('obj.location'):
                    _, nums = line.split('=')
                    x, y, z = [ float(num.strip()) for num in nums.split(',') ]
                    ts.append( np.r_[x, y, z])

                elif line.startswith('obj.rotation_euler'):
                    _, nums = line.split('=')
                    x, y, z = [ float(num.strip()) for num in nums.split(',') ]
                    angles.append(np.r_[x, y, z])
            except:
                print("Exception at line", l)
                print(line)
                raise

        N = len(angles)
        if len(ts) != N:
            raise(Exception("Trouble parsing camera poses. Mismatch in lengths"))
        poses = np.empty((N,6))
        poses[:,:3] = angles
        poses[:,3:] = ts

        return poses

    print("WARNING! No camera poses found...")
    return np.zeros((0,6))


def export_cam(path, Rs, ts, focal):
    header = open('cam_header_template.py').read()
    body = open('cam_frame_template.py').read()

    F = len(Rs)

    Rw, tw = Geom.global_to_local(Rs, ts)
    Rw[:,:,1] *= -1
    Rw[:,:,2] *= -1
    Qs = Quat.R2quats(Rw)

    with open(path, 'w') as file:
        file.write(header.format(focal=focal, cx=0, cy=0))
        for f in range(F):
            t = tw[f]
            q = Qs[f]
            file.write(body.format(
                frame=f,
                tx=t[0], ty=t[1], tz=t[2],
                qw=q[0], qx=q[1], qy=q[2], qz=q[3]
            ))


def save_point_cloud(path, P, edges=None, normals=None):
    data = []

    if edges is not None:
        new_edges = np.empty(edges.shape, np.uint32)
        new_edges[:] = edges
        edges = new_edges.view(type=np.recarray, dtype=[('vertex1', 'u4'),('vertex2', 'u4')])[:,0]
        eel = plyfile.PlyElement.describe(edges, 'edge')
        data.append(eel)

    if normals is not None:
        vertices = np.empty((len(P),6), np.float32)
        vertices[:,:3] = P
        vertices[:,3:] = normals.astype(np.float32)

        vertices = vertices.view(type=np.recarray, dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4')])[:,0]
        vel = plyfile.PlyElement.describe(vertices, 'vertex')
        data.append(vel)

    else:
        vertices = np.empty((len(P),3), np.float32)
        vertices[:] = P
        vertices = vertices.view(type=np.recarray, dtype=[('x','f4'),('y','f4'),('z','f4')])[:,0]

        vel = plyfile.PlyElement.describe(vertices, 'vertex')
        data.append(vel)

    plyfile.PlyData(data).write(path)


def save_for_poission(path, verts, normals, scale, confidence):
    N = verts.shape[0]
    array = np.empty((N,8), dtype=np.float32)
    # array = np.recarray(N, dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('value','f4')])

    array[:,0] = verts[:,0]
    array[:,1] = verts[:,1]
    array[:,2] = verts[:,2]
    array[:,3] = normals[:,0]
    array[:,4] = normals[:,1]
    array[:,5] = normals[:,2]
    array[:,6] = confidence
    array[:,7] = scale

    with open(path, 'wb') as ply:
        with open('ply_header.ply', 'r') as headply:
            header = headply.read()

        ply.write( bytes(header.format(size=N), 'utf-8') )

    # with open(path, 'ab') as ply:
        ply.write(array.tobytes())

    # data = [ plyfile.PlyElement.describe(array, 'vertex') ]
    # plyfile.PlyData(data).write(path)


def write_ply(path, verts, normals=None, faces=None, edges=None):
    V = verts.shape[0]
    hasnormals = normals is not None
    vertsize = 3 + 3*hasnormals

    # Fill the vert array
    vertarray = np.empty((V, vertsize), dtype=np.float32)
    vertarray[:,:3] = verts
    if hasnormals:
        vertarray[:,3:] = normals
        headpath = os.path.join(_pkg_dir, 'ply_header_normals.ply')

    else:
        headpath = os.path.join(_pkg_dir, 'ply_header.ply')

    with open(headpath, 'r') as headply:
        header = headply.read()

    if faces is not None:
        F = faces.shape[0]
        facesize = faces.shape[1]
        facearray = np.empty((F, 1 + 4*facesize), np.uint8)
        facearray[:,0] = facesize
        facearray[:,1:] = (faces.astype(np.int32)).view(np.uint8)
    else:
        F = 0

    if edges is not None:
        E = edges.shape[0]
        edgearray = np.empty((E, 2), np.int32)
        edgearray[:] = edges
    else:
        E = 0

    headerout = header.format(V=V, F=F, E=E)

    with open(path, 'wb') as ply:
        ply.write( bytes(headerout, 'utf-8') )

        ply.write(vertarray.tobytes())

        if F > 0:
            ply.write(facearray.tobytes())

        if E > 0:
            ply.write(edgearray.tobytes())


def read_ply(path, return_normals=False):
    plydata = plyfile.PlyData.read(path)

    vert_data = plydata['vertex']
    N = len(vert_data['x'])

    vertices = np.empty((N,3))
    # normals = np.empty((N,3))
    # faces = np.empty((F,3))

    vertices[:,0] = vert_data['x']
    vertices[:,1] = vert_data['y']
    vertices[:,2] = vert_data['z']

    if 'face' in plydata:
        face_data = plydata['face']
        face_inds = face_data['vertex_indices']
        F = len(face_inds)
        faces = np.empty((F,3), int)
        for f in range(F):
            faces[f] = face_inds[f]
    else:
        faces = np.empty((0,3),int)

    res = [vertices, faces]
    if return_normals:
        normals = np.empty((N,3))
        normals[:,0] = vert_data['nx']
        normals[:,1] = vert_data['ny']
        normals[:,2] = vert_data['nz']
        res.append(normals)

    return res


def read_Rt(path, get_valid=False):
    F = np.load(path)
    if get_valid:
        R, t = F['R'], F['t']
        if 'valid' in F:
            return R, t, F['valid']
        else:
            valid = np.arange(0, len(R), dtype=int)
            return R, t, valid
    else:
        return F['R'], F['t']


def read_lsdslam_poses(path, normalize=True):
    # path = os.path.join(path, 'seq', 'poses.dat')
    with open(path, 'rb') as file:
        ba = np.array(bytearray(file.read()))

    F, I, J = ba[:12].view(np.int32)
    assert I == 3 and J == 4, "Not poses?"

    T = ba[12:].view(float).reshape(F,I,J)

    Rs = np.empty((F,3,3))
    ts = np.empty((F,3))
    for f in range(F):
        R = T[f,:,:3]
        s = np.linalg.norm(R[0])
        R /= s
        Rs[f] = R.T
        ts[f] = -dot(R.T, T[f,:,3])

    return Rs, ts


def write_tracking(path, Rs, ts, times, valid=None):
    '''
    Write local coord poses to CSV according to the following spec:
    `timestamp,valid,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22`
    where rij is the element at the ith row, and jth column in the rotation matrix;
    and valid is 1 or 0, to indicate validity of track.
    '''
    with open(path, 'w') as file:
        file.write('timestamp,valid,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n')

        F = len(Rs)
        valid_int = np.zeros(F, int)
        if valid is not None:
            valid_int[valid] = 1
        else:
            valid_int[:] = 1

        base = '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'

        for f in range(F):
            values = np.r_[ts[f], Rs[f].ravel()]
            line = base.format(times[f], valid_int[f], *values)
            file.write(line)


def read_tracking(path):
    with open(path, 'r') as file:
        file.readline()  # header

        times = []
        valid_int = []
        ts = []
        Rs = []
        ind = 0

        while True:
            ind += 1

            line = file.readline().strip()
            if len(line) == 0:
                break

            values = line.split(',')

            if len(values) != 14:
                raise ValueError("Error on line {}. 14 values expected per line in pose CSV.".format(ind))

            times.append(float(values[0]))

            valid_int.append(int(values[1]))

            t = np.array([ float(v) for v in values[2:5] ])
            ts.append(t)

            R = np.array([ float(v) for v in values[5:] ]).reshape(3, 3)
            Rs.append(R)

        return np.array(Rs), np.array(ts), np.array(times), np.where(valid_int)[0]


def write_obj(path, vertices, faces, uvs=None, im=None):
    dr, fl = os.path.split(path)
    name, ext = os.path.splitext(fl)

    mtlpath = os.path.splitext(path)[0] + '.mtl'
    rel_mtlpath = name + '.mtl'
    rel_path = name + '.jpg'

    faces = faces + 1  # .obj is 1-indexed

    with open(path, 'w') as obj:
        obj.write('# Generated using IO.py\n\n')
        obj.write('o {}\n'.format(name))

        if im is not None:
            obj.write('mtllib {}\n'.format(rel_mtlpath))
        obj.write('\n')

        for v in vertices:
            obj.write('v {} {} {}\n'.format(*v))
        obj.write('\n')

        if uvs is not None:
            for u, v in uvs:
                obj.write('vt {} {}\n'.format(u, v))
            obj.write('\n')

            if im is not None:
                impath = os.path.join(dr, rel_path)
                imwrite(impath, im)

                with open(mtlpath, 'w') as mtl:
                    mtl.write('newmtl Colormap\n')
                    mtl.write('Ka 1.0 1.0 1.0\n')
                    mtl.write('Kd 1.0 1.0 1.0\n')
                    mtl.write('Ks 0.0 0.0 0.0\n')
                    mtl.write('d 1.0\n')
                    mtl.write('illum 1\n')
                    mtl.write('map_Ka {}\n'.format(rel_path))
                    mtl.write('map_Kd {}\n'.format(rel_path))

            obj.write('g {}\n'.format(name))
            obj.write('usemtl Colormap\n'.format(name))
            for f in faces:
                obj.write('f {}/{} {}/{} {}/{}\n'.format( *(f.repeat(2)) ))

        else:
            obj.write('g {}\n'.format(name))
            for f in faces:
                obj.write('f {} {} {}\n'.format(*f))


def get_video_timestamps(videoFile):
    ''' Get the video's timestamps for each frame.
    Uses ffprobe, ensure that ffmpeg is installed and in the environment path.

    return: 1D `numpy.ndarray` of timestamps.
    '''
    print("Getting frame times for:", videoFile)
    videoFile = os.path.abspath(videoFile)
    infoPath, ext = os.path.splitext(videoFile)
    infoPath += '_frames.txt'

    # Call ffprobe to extract the timeframe info
    FNULL = open(os.devnull, 'w')
    if not os.path.exists(infoPath):
        with open(infoPath, 'w') as F:
            args = ['ffprobe', '-i', videoFile, '-show_frames', '-print_format', 'json']
            call(args, stdout=F, stderr=FNULL)

    timestamps = []
    with open(infoPath, 'r') as F:
        J = json.load(F)
        if 'frames' not in J:
            raise IOError('Unable to find video: %s' % videoFile)
        frames = J['frames']
        timestamps = [ float(frame['pkt_pts_time']) for frame in frames
                       if frame['media_type'] == 'video']

    return np.array(timestamps)


def read_cam_param(path):
    with open(path) as file:
        J = yaml.safe_load(file)
        return np.r_[J['fx'], J['fy'], J['cx'], J['cy']]


def write_cam_param(path, cam):
    J = dict(fx=cam[0], fy=cam[1], cx=cam[2], cy=cam[3])
    with open(path, 'w') as file:
        yaml.dump(J, file)


def read_rects(path):
    with open(path) as file:
        J = yaml.safe_load(file)
        rects = J['rects']
        N = len(rects)
        frames = np.empty(N, np.int64)
        boxes = np.empty((N,4))
        for i in range(N):
            frames[i] = rects[i]['frame']
            boxes[i] = rects[i]['bounds']

        return frames, boxes
