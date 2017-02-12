from __future__ import print_function

import numpy as np
from numpy import zeros, ones, array, r_, c_, dot, eye, linalg, sqrt
from numpy.linalg import norm

import scipy.io as io
import scipy.interpolate as sciint
from scipy import linalg as splinalg
from scipy import sparse
from scipy import ndimage
# from sparsesvd import sparsesvd
# from pypropack import svdp

import cv2
# from cv2 import cv

# import DeviceInfo
# import Util

import os
import glob
import time
import re
from subprocess import call

# Plotting
# import Plotting
import matplotlib.pyplot as plt
# from progressbar import ProgressBar

from scipy import signal

import json

import IO
import Geometry as Geom
import SolvePnP

import pynutmeg


class TrackingException(BaseException):
    pass


def _makeDeltaR(w, I=False):
    if I:
        dR = eye(3)
    else:
        dR = zeros((3,3))

    dR[1,2], dR[2,1] = -w[0], w[0]
    dR[2,0], dR[0,2] = -w[1], w[1]
    dR[0,1], dR[1,0] = -w[2], w[2]
    return dR


def _makeLowPass(x, T, cutoff, axis=-1):
    '''
    :param x: The signal to be filtered (1D or 2D)
    :param T: Sampling rate of the signal, x
    :param cutoff: Cutoff frequency in Hertz
    '''
    # Construct the filter
    Fs = 1./T
    ntaps = np.ceil(Fs/cutoff)
    # Make sure its odd
    ntaps += 1 - (ntaps % 2)
    b = signal.firwin(ntaps, cutoff, window=('kaiser', 10), nyq=0.5*Fs)
    return signal.convolve(x, b)


def rod2quat(rvecs):
    ''' Convert array of rvecs to quaternions [w, x, y, z]

    :param rvecs: m x 3 numpy.ndarray of rvecs.
    :return: m x 4 numpy.ndarray of quaternions.
    '''
    single = (rvecs.ndim == 1)

    if single:
        theta = np.linalg.norm(rvecs)
        if theta == 0:
            v = np.zeros(3)
        else:
            v = rvecs/theta
    else:
        theta = np.linalg.norm(rvecs, axis=1)
        v = (rvecs.T / theta).T
        v[theta == 0] = [0.,0.,0.]

    q1 = np.cos(0.5*theta)
    qv = (v.T*np.sin(0.5*theta))

    if single:
        quats = np.hstack( (q1,qv) )
    else:
        quats = np.vstack( (q1,qv) ).T

    return quats


class PoseTracker(object):
    '''
    Given a camera calibration matrix and a video file, track the pose of
    the camera in some vision frame. This base class is extended for
    different methods of pose estimation.
    '''

    def __init__(self, videoFile, cameraMatrix=None, distCoeffs=None, deviceTag=None):
        self.distCoeffs = distCoeffs
        self.videoFile = videoFile
        # self.vidCap = cv2.VideoCapture(videoFile)
        self.imagePaths = glob.glob( os.path.join(videoFile, '*.jpg') )

        # Create a directory to keep all the files for this video
        d, f = os.path.split( os.path.realpath(videoFile) )
        name, ext = os.path.splitext(f)
        self.videoDir = os.path.split(videoFile)[0]

        # self.nFrames = int(self.vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
        # w = int(self.vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(self.vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nFrames = len(self.imagePaths)
        imtest = IO.imread(self.imagePaths[0])
        h, w = imtest.shape[:2]
        self.size = (h,w)
        print("Frames:", self.nFrames)

        if cameraMatrix is None and deviceTag is not None:
            cameraMatrix = DeviceInfo.loadCameraMatrix(deviceTag, (h,w))[0]
        # else:
        self.cameraMatrix = cameraMatrix

        # if cameraMatrix is not None:
        #     self.cameraMatrix = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w,h))

        # Keep track of plotting
        self.ax = None

    def _getVideoFrames(self, start=0):
        for f in range(start, self.nFrames):
            path = self.imagePaths[f]
            im = IO.imread(path)

            yield im
        # im, imRect = zeros(self.size + (3,), np.uint8), zeros(self.size + (3,), np.uint8)
        # while self.vidCap.grab():
        #     success = self.vidCap.retrieve(im)[0]
        #     cv2.undistort(im, self.cameraMatrix, self.distCoeffs, imRect)

        #     if not success:
        #         print("Failed to retrieve frame...")
        #         break
        #     yield im

        # self.vidCap.release()

    def _solvePnP(self, objectPoints, imagePoints, cameraMatrix, distCoeffs=None, skipped=[]):
        N = len(imagePoints)
        Ts = np.tile( np.eye(4,4), (N,1,1) )
        q_list = np.zeros((N,4))
        # pbar = ProgressBar(self.nFrames).start()
        i = -1
        useExtrinsic = False
        # lastRvec = np.array([0,0,0],np.float32).T
        # lastP = np.array([0.,0.,1.], np.float32).T
        lastRvec, lastP = None, None
        lastR = None
        firstRvec, firstTvec = None, None
        firstGood = 0

        if type(objectPoints) == list:
            objectPointsMean = np.array(objectPoints).mean(axis=0)
        else:
            objectPointsMean = objectPoints

        skipped = set(skipped)
        invalid = set()
        # flags = cv2.SOLVEPNP_EPNP
        # flags = cv2.SOLVEPNP_DLS

        # fig = pynutmeg.figure('checker', 'figs/segments.qml')
        # fx, fy, cx, cy = cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2]

        goodFrames = []
        for points in imagePoints:
            i += 1
            # pbar.update(i)
            if i in skipped or points.shape[0] == 0:
                print("\tSkipped {}".format(i))
                if i > 0:
                    Ts[i] = Ts[i-1]
                continue

            if not useExtrinsic:
                op = objectPointsMean
            elif type(objectPoints) == list:
                op = objectPoints[i]
            else:
                op = objectPoints

            ret, rvec, P = cv2.solvePnP(
                op, points,
                cameraMatrix, distCoeffs,
                useExtrinsicGuess=useExtrinsic, rvec=lastRvec, tvec=lastP)

            # ret, R, P = SolvePnP.solvepnp(op, points, cameraMatrix, lastR, lastP)
            print("\nFrame {}:".format(i), ret, useExtrinsic)
            # print(P)

            # if i > 1240:
            #     exit()

            if P[2] < 0:
                invalid.add(i)
                continue

            if not useExtrinsic:
                # firstRvec, firstTvec = rvec.copy(), P.copy()
                firstGood = i
                print("First good frame: %d" % i)

            # Set useExtrinsic to true once we have a valid estimation
            useExtrinsic = True
            lastRvec = rvec.copy()
            lastP = P.copy()
            # lastR = R.copy()

            # Build the transformation matrix
            R,j = cv2.Rodrigues(rvec)
            # R = Geom.w_to_R(rvec)

            # u, v = points.reshape(-1, 2).T
            # fig.set('ax.P0', x=u, y=v)
            # proj = dot(R, op.T) + P.reshape(-1,1)
            # up, vp = proj[:2]/proj[2]
            # up = fx*up + cx
            # vp = fy*vp + cy
            # fig.set('ax.P1', x=up, y=vp)
            # fig.set('ax.seg', x=u, y=v, endX=up, endY=vp)
            # time.sleep(0.1)

            # q_list[i] = rod2quat(rvec.ravel())
            # q_list[i] = rod2quat(rvec)

            Ts[i, :3, :3] = R.T
            Ts[i, :3, 3] = -R.T.dot(P.ravel())

            goodFrames.append(i)

        print("Processed %d frames" % i)
        # pbar.finish()

        # Now back propogate the good frames to the ones that were dropped at
        # the start of the sequence due to invalid solvePnP solutions
        # lastRvec, lastP = firstRvec, firstTvec
        # redo = list(range(firstGood))
        # redo.reverse()

        # for i in redo:
        #     # TODO: Clean up this boilerplate...
        #     print("Redoing %d" % i)
        #     points = imagePoints[i]
        #     if i in skipped or points.shape[0] == 0:
        #         continue

        #     if type(objectPoints) == list:
        #         op = objectPoints[i]
        #     else:
        #         op = objectPoints

        #     ret, rvec, P = cv2.solvePnP(op, points,
        #         cameraMatrix, None,
        #         useExtrinsicGuess=True, rvec=lastRvec, tvec=lastP)

        #     lastRvec, lastP = rvec.copy(), P.copy()

        #     # Build the transformation matrix
        #     R = cv2.Rodrigues(rvec)[0]

        #     q = rod2quat((rvec.T)[0])
        #     q_list[i] = q

        #     T = eye(4)
        #     T[:3, :3], T[:3, 3] = R, P.flatten()
        #     Ts[i] = linalg.inv(T)

        #     if i in invalid:
        #         invalid.remove(i)

        #     goodFrames.append(i)

        skipped = list( skipped.union(invalid) )
        skipped.sort()

        return Ts, np.array(list(skipped))

    def _solvePnPGN(self, objectPoints, imagePoints, cameraMatrix, init,
                    imu=None, solveScale=True, modelFile=None, smooth=0,
                    skipped=[], it=50, plot=False):
        '''
        Solve the Perspective n-Point problem using Gauss-Newton.
        Needs a good initialization, init.

        :param objectPoints: Nx3 array of 3D points
        :param imagePoints: Nx2 array of corresponding 2D image observations
        :param init: 3x4 transformation matrix of the form [R t]
        :param modelFile: A file with dictionary representing the possible eigen
            deformations of the 3D object. If none, don't attempt to solve for
            the 3D structure.
        '''
        if plot:
            Nutmeg.init(timeout=10000)
            fig = Nutmeg.figure('Gauss-Newton', 'tripleFig.qml')
            figR = Nutmeg.figure('GN Rot', 'tripleFig.qml')
            global accFig
            accFig = None

        F = len(imagePoints)  # Frames
        N = len(objectPoints)  # Points

        # For readability
        pts3d = objectPoints

        smooth, lam = smooth > 0, smooth

        # Load the model
        model = None
        q = None
        if modelFile is not None:
            data = np.load(modelFile)
            model = {'mean': data['M'], 'dict': data['D'], 'eigen': data['v'] }
            Q = len(data['v'])
            q = zeros(Q)  # model['eigen']
        useModel = model is not None

        # Load the IMU data
        # imu = None
        useImu = imu is not None
        if useImu:
            print("Using IMU...")
            s = imu['scale']
            # gt: 0.0035
        else:
            print("Not using IMU...")
            s = 1.
            solveScale = False

        pts3d_hom = c_[ objectPoints, ones(N) ]
        pts2d = imagePoints

        if init.ndim == 3:
            # Pose estimates exist for each frame
            Ts = init.copy()
            for f in range(F):
                Ts[f] = linalg.inv(Ts[f])
                # Rnoise = cv2.Rodrigues( 0.1*np.random.standard_normal(3) )[0]
                # Ts[f, :3, :3] = Rnoise.dot(Ts[f, :3, :3])
        else:
            raise TrackingException("Bad dimensions for init pose. %s" % (init.shape,))

        if plot:
            fig.set('ax.red', y=init[:, :3, 3].T)
            figR.set('ax.red', y=init[:, 2, :3].T)

        K = cameraMatrix

        def project(points, T):
            # Project 3D points to 2D with perspective matrix, T
            x, y, z = dot(cameraMatrix, dot(T[:3], points.T))
            return c_[x/z, y/z]

        converged = False

        try:
            solveFrameByFrame = False
            if solveFrameByFrame:
                for f in range(F):
                    print("Frame: %d" % f)
                    for i in range(it):
                        J = self._jacobian(pts3d, K, pose=Ts[f])
                        H = J.dot(J.T)

                        proj = project(pts3d_hom, Ts[f])
                        g = J.dot( (pts2d[f] - proj).flatten() )

                        u = solveSVD(H, g)

                        Ts[f, :3, 3] += u[3:]

                        w = u[:3]
                        w_len = norm(w)
                        if w_len > 0.99*np.pi:
                            w = (0.99*np.pi/w_len)*w
                        dR = splinalg.expm( _makeDeltaR(w) )

                        Ts[f, :3, :3] = Ts[f, :3, :3].dot(dR)

                        if norm(u) < 1e-3:
                            break

            else:
                tol = sqrt(F)*1e-4
                for i in range(it):
                    print("Iteration: %d" % i)
                    H, g = self._formJacobian(pts3d, pts2d, K, pose=Ts,
                                              model=model, localPose=q,
                                              imu=imu, s=s, lam=lam,
                                              solveScale=solveScale, smooth=smooth)
                    # H, g = self._formHessianJacobian(pts3d, pts2d, K, pose=Ts)

                    # Due to the way the Jacobian is formed the update, u, is made up of:
                    # [wx wy wz tx ty tz]*F where wx..z parametrized delta angles,
                    # tx..z are translation for each frame
                    print("SolveSVD...")
                    # u = -solveSVD(H, g, isSparse=True)
                    u = -sparse.linalg.spsolve(H, g)
                    # Reshape so that rows alternate [w1; t1; w2; t2; ..; q]
                    u_pose = u[:6*F].reshape(-1, 3)

                    # Update structure
                    if useModel:
                        q += u[6*F:6*F + Q]
                    if solveScale:
                        s += u[-1]
                        print("Scale update: %g (%g)" % (s, u[-1]))

                    # Update translation
                    Ts[:, :3, 3] += u_pose[1::2]

                    # Update rotation and be sure to reproject correctly
                    # TODO: Make this quaternion...
                    print("Updating pose... %g" % norm(u_pose))
                    # print(u_pose)
                    T_plot = Ts[:, :3, 3].copy()
                    for f in range(F):
                        # print("Updating: %d" % f)
                        w = u_pose[2*f]
                        w_len = norm(w)
                        if w_len > 0.99*np.pi:
                            w = (0.99*np.pi/w_len)*w
                        dR = splinalg.expm( _makeDeltaR(w) )
                        # dR = _makeDeltaR(w, I=True)
                        # dR = Util.makeSO3(dR)

                        Ts[f, :3, :3] = Ts[f, :3, :3].dot(dR)
                        T_plot[f] = -Ts[f, :3, :3].T.dot(T_plot[f])

                    if plot:
                        print("Send fig..")
                        fig.set('ax.blue', y=T_plot.T)
                        figR.set('ax.blue', y=Ts[:, :3, 2].T)
                        print("Sent fig")

                    if norm(u) < tol:
                        print("norm(u): %g" % norm(u))
                        converged = True
                        break
                    print("End iteration")

        except KeyboardInterrupt:
            print("Interrupted...")
            pass

        # Ts[f, :3, :3] = R.T
        # Ts[f, :3, 3] = -R.T.dot(t)
        for f in range(F):
            Ts[f] = linalg.inv(Ts[f])

        if not converged:
            print("Warning: _solvePnPGN, unable to converge in %d iterations for frame %d" % (it, F))

        face3D = None
        if useModel:
            face3D = model['mean'] + q.dot(model['dict'].transpose(1,2,0)).T

        return Ts, face3D

    # For first derivative:
    Mx, My, Mz = [ _makeDeltaR(v) for v in eye(3) ]
    # For second derivative for each possible pair (note the symmetry):
    Mww = [ 0.5*(M1.dot(M2) + M2.dot(M1))
            for M1, M2
            in ((Mx, Mx), (Mx, My), (Mx, Mz),
                (My, My), (My, Mz),
                (Mz, Mz)) ]

    def _formJacobian(self, objectPoints, imagePoints, cameraMatrix, pose,
                      model=None, localPose=None, imu=None, s=1., lam=1.,
                      solveScale=True, smooth=False):
        '''
        WARNING: "smooth" not compatible with imu currently
        '''
        N = objectPoints.shape[0]
        F = imagePoints.shape[0]

        K = cameraMatrix  # Improve readability (it gets messy)
        useModel = model is not None and localPose is not None
        if useModel:
            q = localPose  # (len Q)
            Q = len(q)
            D = model['dict']  # (Px3xQ)
            pts3d = model['mean'] + q.dot(D.transpose(1,2,0)).T

        else:
            Q = 0
            pts3d = objectPoints

        useImu = imu is not None and 'acc' in imu and 'grav' in imu
        if useImu and smooth:
            raise TrackingException("'smooth' and 'imu' are currently incompatible")

        # solveScale = True
        if useImu:
            global accFig
            if accFig is None:
                accFig = Nutmeg.figure('GN acc', 'tripleFig.qml')
            camAcc = zeros((3,F))
            imuAcc = zeros((3,F))
            acc = imu['acc'] - imu['bias']
            grav = imu['grav']
            h = 1./imu['period']
            # coeff = h**2*r_[1., -2, 1]
            coeff = _makeLowPass(h**2*r_[1, -2, 1], imu['period'], 1.2)
            C = len(coeff)
            if C % 2 == 0:
                raise Exception("Bad filter length: %d" % C)
            print("Coeff length: %s" % C)
            Chalf = (C-1)//2
        else:
            C = 0

        # Unknowns
        U = 6*F + Q + 1*(solveScale)
        print("Unknowns: %d" % U)
        # Jacobian
        data, rows, cols = [], [], []
        # J = zeros((2*N*F, U))
        # Equations
        E = 2*N*F + 3*(F-C+1)*useImu + 3*(F-1)*smooth  # Projection error + IMU error + smooth error
        # Residuals
        res = zeros(E)

        Rs, ts, Ps, Ras = [], [], [], []
        for f in range(F):
            Rtmp, ttmp = pose[f, :3, :3], pose[f, :3, 3]
            Rs.append(Rtmp)
            ts.append(ttmp)
            Ps.append(-Rtmp.T.dot(ttmp))
            Ras.append(array([ Rtmp.dot(M).T for M in (self.Mx, self.My, self.Mz) ]))

        for f in range(F):
            # print("Frame: %d" % f)
            tm = []
            tm.append(time.clock())
            R, t = pose[f, :3, :3], pose[f, :3, 3]
            KR = K.dot(R)
            Kt = K.dot(t)

            # First derivative for Jacobian
            KRa = array([ KR.dot(M) for M in (self.Mx, self.My, self.Mz) ])

            # tm.append(time.clock())
            # tm2 = zeros(5)

            for n in range(N):
                tm2b = zeros(6)
                tm2b[0] = time.clock()
                p2d = imagePoints[f][n]
                p3d = pts3d[n]
                x, y, z = KR.dot(p3d) + Kt

                w = 1./z
                u, v = p2d  # Observations
                # Derive x,y,z w.r.t. wx, wy, wz
                if useModel:
                    dx, dy, dz = c_[ KRa.dot(p3d).T, K, KR.dot(D[n]) ]
                else:
                    dx, dy, dz = c_[ KRa.dot(p3d).T, K ]

                r = 2*(n + f*N)
                cs = 6*f + r_[0:6]
                Jx = w**2*(dx*z - x*dz)
                Jy = w**2*(dy*z - y*dz)
                # Take care of the transform at this frame (block diag in Jacobian)
                rows.extend([r]*6)
                cols.extend(cs)
                data.extend(Jx[:6])
                rows.extend([r + 1]*6)
                cols.extend(cs)
                data.extend(Jy[:6])

                # Then the structure (last Q columns of the Jacobian)
                rows.extend( [r]*Q )
                cols.extend( r_[0:Q] + 6*F )
                data.extend( Jx[6:] )
                rows.extend( [r + 1]*Q )
                cols.extend( r_[0:Q] + 6*F )
                data.extend( Jy[6:] )

                res[r] = w*x - u
                res[r + 1] = w*y - v

            if smooth and f >= 1:
                # TODO: WARNING: Not compatible with useImu currently
                rs = 2*N*F + 3*(f-1) + r_[0:3]
                cs = 6*(f-1) + r_[0:12]
                J = lam*c_[ -Ras[f-1].dot(ts[f-1]).T,  -Rs[f-1].T,
                            Ras[f].dot(ts[f]).T,       Rs[f].T ]

                j, i = np.meshgrid(cs, rs)
                rows.extend( i.flatten() )
                cols.extend( j.flatten() )
                data.extend( J.flatten() )

                res[rs] = lam*( Rs[f].T.dot(ts[f]) - Rs[f-1].T.dot(ts[f-1]) )

            if useImu and f >= Chalf and f < F - Chalf:
                # TODO: Write accompanying LaTeX/markdown explaining math
                C = len(coeff)

                rs = 2*N*F + 3*(f-Chalf) + r_[0:3]
                cs = 6*(f-Chalf) + r_[0:C*6]

                Ja = zeros((3, C*6))
                Js = zeros(3)
                # Jtmp = zeros(3)
                for c, fc in zip(range(C), f + r_[0:C] - Chalf):
                    cc = r_[0:3] + c*6  # Ahhh, running out of short var names..

                    Jtmp = -s*coeff[c]*Ras[fc].dot(ts[fc]).T
                    if f == fc:
                        Jtmp -= Ras[fc].dot(acc[f]).T

                    Ja[:,cc] = Jtmp
                    Ja[:,cc+3] = -s*coeff[c]*Rs[fc].T
                    Js += coeff[c]*Ps[fc]

                # Reshape the data and add to the indices
                j, i = np.meshgrid(cs, rs)
                rows.extend( i.flatten() )
                cols.extend( j.flatten() )
                data.extend( lam*Ja.flatten() )
                # Scale term
                if solveScale:
                    rows.extend(rs)
                    cols.extend([U - 1]*3)
                    data.extend(lam*Js)

                # print(rs-2*N*F, rs, E)
                res[rs] = lam*(s*Js + grav - R.T.dot(acc[f]))
                # res[rs] = lam*(R.dot(s*Js + grav) - acc[f])

                camAcc[:,f] = R.dot( s*Js + grav )
                imuAcc[:,f] = ( acc[f] )

        if useImu:
            accFig.set('ax[:].red', y=imuAcc)
            accFig.set('ax[:].blue', y=camAcc)

        J = sparse.coo_matrix((data, (rows, cols)), shape=(E, U)).tocsr()
        # plt.spy(J[:2*N*3][:,:18], precision=0.1)
        # Jd = J[2*N:2*N*2][:,6:12].todense()
        # J2 = self._jacobian(pts3d, K, pose[1]).T
        # print((Jd - J2).max())
        # plt.matshow(Jd - J2)
        # plt.show()
        # print("Res1:", res[:2*N])
        H = J.T.dot(J)
        g = J.T.dot(res)

        return H, g

    def _formHessianJacobian(self, objectPoints, imagePoints, cameraMatrix, pose, lam=1.):
        '''
        From: http://www.seas.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR94b.pdf
        "Minimization on the Lie Group SO(3) and Related Manifolds", C. Tayler & D. Kriegman, 1994
        '''
        N = objectPoints.shape[0]
        F = imagePoints.shape[0]
        g = zeros(6*F)
        # H = sparse.lil_matrix((6*F, 6*F))

        data, rows, cols = [], [], []

        K = cameraMatrix  # Improve readability (it gets messy)

        for f in range(F):
            # print("Frame: %d" % f)
            tm = []
            tm.append(time.clock())
            R, t = pose[f, :3, :3], pose[f, :3, 3]
            KR = K.dot(R)
            Kt = K.dot(t)

            # First derivative for Jacobian
            KRa = array([ KR.dot(M) for M in (self.Mx, self.My, self.Mz) ])

            # Second derivative for Hessain
            KRxx, KRxy, KRxz, KRyy, KRyz, KRzz = [ KR.dot(M) for M in self.Mww ]

            KRww = array([[KRxx, KRxy, KRxz],
                          [KRxy, KRyy, KRyz],
                          [KRxz, KRyz, KRzz]])
            tm.append(time.clock())

            tm2 = zeros(5)

            for n in range(N):
                tm2b = zeros(6)
                tm2b[0] = time.clock()
                p3d, p2d = objectPoints[n], imagePoints[f][n]
                x, y, z = KR.dot(p3d) + Kt

                w = 1./z
                u, v = p2d  # Observations
                # Derive x,y,z w.r.t. wx, wy, wz
                dx, dy, dz = c_[ KRa.dot(p3d).T, K ]

                ind = 6*f + r_[0:6]
                g[ind] += 2*w**3*(u*z - x)*(x*dz - dx*z)
                g[ind] += 2*w**3*(v*z - y)*(y*dz - dy*z)

                # g[ind[:3]] += lam*(2*)

                tm2b[1] = time.clock()
                # continue

                # Second order derivatives!
                # The following produces a 3x3x3 array where the 1st and 2nd dimensions are wx,wy,wz
                # And the 3rd dimension is made of vectors [x y z] each derived twice w.r.t.
                # the respective variables (better shown by this table...)
                # All second derivatives w.r.t. tx,ty,tz are zero
                #
                # x,y,z | wx wy wz tx ty tz
                # ------+------------------
                #    wx | xx xy xz  0  0  0
                #    wy | yx yy yz  0 ..
                #    wz | zx zy zz  0
                #    tx |  0  0  0  0 ..
                #    .. | ..       ..

                ddw = zeros((6, 6, 3))
                ddw[:3, :3] = KRww.dot(p3d)
                ddx, ddy, ddz = ddw[:,:,0], ddw[:,:,1], ddw[:,:,2]

                dxx, dyy, dzz = np.tile(dx, (6,1)), np.tile(dy, (6,1)), np.tile(dz, (6,1))
                tm2b[2] = time.clock()

                w4 = 2*w**4
                z2 = z**2
                dzzTdzz = dzz.T*dzz
                H_t = w4*((u*z - x)*((x*ddz + dxx*dzz.T + dxx.T*dzz)*z - 2*x*dzzTdzz - z2*ddx)
                          + (x*dzz - dxx*z)*(x*dzz.T - dxx.T*z))
                H_t += w4*((v*z - y)*((y*ddz + dyy*dzz.T + dyy.T*dzz)*z - 2*y*dzzTdzz - z2*ddy)
                           + (y*dzz - dyy*z)*(y*dzz.T - dyy.T*z))

                indR, indC = np.meshgrid(ind, ind)
                data.extend( H_t.flatten() )
                rows.extend( indR.flatten() )
                cols.extend( indC.flatten() )

                tm2b[3] = time.clock()

                if f >= 1 and f < F - 1:
                    # Smoothness term
                    g[6*f + 3: 6*f + 6] += lam*(2*pose[f, :3, 3] - pose[f+1, :3, 3] - pose[f-1, :3, 3])

                    for i in range(3, 6):
                        fn1, f0, fp1 = 6*(f-1) + i, 6*f + i, 6*(f+1) + i
                        rows.extend([f0, f0, fp1, f0, fn1])
                        cols.extend([f0, fp1, f0, fn1, f0])
                        data.extend(lam*r_[2, -1, -1, -1, -1])
                        # H[f0, f0] += 2*lam
                        # H[f0, fp1] -= lam
                        # H[fp1, f0] -= lam
                        # H[f0, fn1] -= lam
                        # H[fn1, f0] -= lam
                tm2b[4] = time.clock()

                tm2 += np.diff(tm2b)

            tm.append(time.clock())

            # print("Frame times (ms):", (np.diff(tm)*1000).astype(int))
            # print("Points (ms):", (tm2*1000).astype(int))
        # return None, g
        return sparse.coo_matrix((data, (rows, cols)), shape=(6*F, 6*F)).tocsr(), g

    def _jacobian(self, objectPoints, cameraMatrix, pose):
        '''
        :param objectPoints: Nx3 array of N 3D points
        :param pose: A transformation matrix of the form [R t]. Can be homegenous.
        '''
        N = objectPoints.shape[0]
        J = zeros((2*N,6))

        R = pose[:3, :3]
        t = pose[:3, 3]
        K = cameraMatrix  # Improve readability

        Rx, Ry, Rz = [ _makeDeltaR(v) for v in eye(3) ]
        # Rx = array([[0,0,0], [0,0,-1], [0,1,0]], float)
        # Ry = array([[0,0,1], [0,0,0], [-1,0,0]], float)
        # Rz = array([[0,-1,0], [1,0,0], [0,0,0]], float)

        # print(K.shape, R.shape)
        M = dot(K, R)  # Projected rotation transformation
        Mx, My, Mz = dot(M, Rx), dot(M, Ry), dot(M, Rz)

        Kt = dot(K, t)  # Projected position

        for n, p in enumerate(objectPoints):
            v = dot(M, p) + Kt  # Project point, p
            w = 1./v[2]
            x, y = w*v[0], w*v[1]
            vx, vy, vz = dot(Mx, p), dot(My, p), dot(Mz, p)
            XY = array([[1,0,-x], [0,1,-y]])
            J[2*n: 2*(n+1), :] = w*dot(XY, c_[vx, vy, vz, K])

        # print("Jacobian:", J)

        return J.T

    def getFrameTimes(self):
        '''
        Get the video's timestamps for each frame.
        :return: 1D `numpy.ndarray` of timestamps.
        '''
        return IO.get_video_timestamps(self.videoFile)

    def playVideo(self):
        pbar = ProgressBar(self.nFrames).start()

        i = 0
        for frame in self._getVideoFrames():
            cv2.imshow('Video', frame)

            i += 1
            pbar.update(i)
            cv2.waitKey(10)

        pbar.finish()

    def drawCamera(self, T):
        ''' Draw the camera in a 3D frame to visualise the extrinsics '''
        # Transform the camera frustum
        verts = np.dot(T,self._verts.T).T
        # Generate a list of polygons based on _tris
        poly3d = verts[self._tris,0:3]

        if self.ax == None:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.draw()

            self.cam = Poly3DCollection(poly3d, facecolors=['r','b','b','g'])
            self.ax.add_collection3d( self.cam )

        else:
            self.cam.set_verts(poly3d)
            plt.draw()


class PoseChessboard(PoseTracker):

    def track(self,
              chessboardSize=(8,6),
              squareSize=(1,1),
              forceRetrack=False,
              force=False,
              showTrack=False,
              smooth=80.0,
              calibrate=0,
              transpose=False):
        '''Tracks the chessboard in the video.

        Estimates the 4x4 affine transformation that expresses the 6DOF
        position of the camera in relation to the chessboard.

        :param chessboardSize: The number of corners in the chessboard (length, width). These corners are where black squares touch.
        :param squareSize: The metric size of the squares (length, width).
        :param forceRecalculate: The tracker keeps a record of the tracking for each video file. By default, if a recorded track sequence is on file, it will be loaded to save time. If this is set to true, this function will recalculate the tracking sequence.

        :return: A list of 4x4 affine transformations to express the 6dof camera pose. This is not guaranteed to be the same length as the video sequence provided as "bad" frames at the beginning and end will be truncated.
        '''
        trackFile = self.videoDir + '/chessTrack.npz'

        if os.path.exists(trackFile) and not forceRetrack:
            print("Loading tracked points from", trackFile)
            compressed = np.load(trackFile)
            imagePoints = compressed['imagePoints']
            skipped = compressed['skipped']
            firstFrame = compressed['firstFrame']
            lastFrame = compressed['lastFrame']
        else:
            imagePoints, skipped, firstFrame, lastFrame = \
                self._trackChessboard(chessboardSize, transpose=transpose, show=showTrack)
            np.savez(trackFile, imagePoints=imagePoints, skipped=skipped,
                firstFrame=firstFrame, lastFrame=lastFrame)

        objectPoints = self._genChessboardWorldPoints(
                chessboardSize, squareSize)

        if calibrate > 0:
            K = self.cameraMatrix.copy()
            kept = ones(len(imagePoints), bool)
            if len(skipped) > 0:
                kept[skipped] = False
            sel = np.random.choice(np.where(kept[::120])[0], calibrate, replace=False)
            N = len(sel)

            calPoints = list( imagePoints[sel] )
            calObject = [objectPoints]*N
            self.distCoeffs = zeros(4)

            flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO
            print("Calibrating....")
            result = cv2.calibrateCamera(calObject, calPoints, (720,1280), cameraMatrix=self.cameraMatrix, distCoeffs=self.distCoeffs, flags=flags)
            ret, self.cameraMatrix, self.distCoeffs, rvecs, tvecs = result

            np.set_printoptions(suppress=True)
            print("Calibration:\n", self.cameraMatrix)
            print("vs:\n", K)

            print("Dist:", self.distCoeffs)
            exit()

        transformsFile = self.videoDir + '/cameraTransforms.npz'

        if os.path.exists(transformsFile) and not (force or forceRetrack):
            print("Loading Camera Transforms from", transformsFile)
            compressed = np.load(transformsFile)
            transforms = compressed['transforms']
            times = compressed['times']
        else:

            objectPoints = objectPoints.astype(float)
            imagePoints = np.array(imagePoints)
            shp = imagePoints.shape
            imagePoints = imagePoints.reshape(shp[0], -1, 2)
            print(imagePoints.shape)

            transforms, skipped = self._solvePnP(objectPoints,
                imagePoints, self.cameraMatrix,
                skipped=skipped)

            # Using PnPGN, a smoothness constraint can be applied to the trajectory
            imagePoints = np.array(imagePoints)
            shp = imagePoints.shape
            imagePoints = imagePoints.reshape(shp[0], -1, 2)
            # transforms, _ = self._solvePnPGN(objectPoints, imagePoints,
            #                               self.cameraMatrix, transforms,
            #                               smooth=smooth, plot=False)

            base = os.path.split(self.videoFile)[0]
            vid = os.path.join(base, 'video.mp4')
            times = IO.get_video_timestamps(vid)
            if len(transforms) != len(times):
                print('Transforms and Timestamps are not the same length!')
            # np.savez(transformsFile, transforms=transforms, times=times, skipped=skipped)
            Rw, tw = transforms[:,:3,:3], transforms[:,:3,3]
            R, t = Geom.global_to_local(Rw, tw)
            valid = ones(len(R), bool)
            if len(skipped) > 0:
                valid[skipped] = False

            np.savez(transformsFile, R=R, t=t, times=times, valid=np.where(valid)[0])
            lsdFile = self.videoDir + '/lsdslam_Rt.npz'
            np.savez(lsdFile, R=R, t=t, times=times, valid=np.where(valid)[0])

        return transforms, times, skipped

    def _trackChessboard(self, chessboardSize=(8,6), transpose=False, show=False):
        print("Processing", self.nFrames, "frames")
        # pbar = ProgressBar(self.nFrames).start()
        i = 0
        firstFrame = -1
        lastFrame = 0

        nPoints = np.prod(chessboardSize)
        skipped = []
        imagePoints = []
        corners_old = None  # Suppress warning...
        flip_thresh = 50

        H, W = self.size
        gray = zeros(self.size, np.uint8)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH #+ cv2.CALIB_CB_NORMALIZE_IMAGE

        crop_row = np.s_[0:H]
        crop_col = np.s_[0:W]

        for frame in self._getVideoFrames():
            print("Frame:", i, end=' ')
            # if i > 10:
            #     break
            if True or i == 0:
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, dst=gray)
                success, corners = cv2.findChessboardCorners(gray[crop_row, crop_col], chessboardSize, flags=flags)

                if success:
                    delta = 7
                    corners[:,:,0] += crop_col.start
                    corners[:,:,1] += crop_row.start

                    if corners_old is not None:
                        error = np.abs(corners - corners_old).max()
                        print("error:", error, end=' ')
                        if error > flip_thresh:
                            corners_flip = np.empty_like(corners)
                            # J = len(corners)
                            # for j in range(J):
                                # corners_flip[j] = corners[J - j - 1]
                            corners_flip[:] = corners[::-1]
                            if np.abs(corners_flip - corners_old).max() < error:
                                print("Flipped", end=' ')
                                corners = corners_flip

                else:
                    corners = corners_old
                    delta = 7

                    # grow = 2
                    # crop_row = np.s_[ max(0, crop_row.start - grow): min(H - 1, crop_row.stop + grow) ]
                    # crop_col = np.s_[ max(0, crop_col.start - grow): min(W - 1, crop_col.stop + grow) ]

                if corners is not None:
                    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 0.1)
                    cv2.cornerSubPix(gray, corners, (delta, delta), (-1,-1), criteria)

                    minx, miny = corners.min(axis=0)[0]
                    maxx, maxy = corners.max(axis=0)[0]

                    border = 60
                    crop_row = np.s_[ max(0, int(miny - border)): min(H - 1, int(maxy + border)) ]
                    crop_col = np.s_[ max(0, int(minx - border)): min(W - 1, int(maxx + border)) ]

                    # delta = 3 #max(2, int(np.ceil(2*np.abs(corners - corners_old).max())))
                    corners_old = corners.copy()

            imagePoints.append(corners_old)
            if not success or len(corners_old) < nPoints:
                # Occlusions/detection problems
                skipped.append(i)
            else:
                if firstFrame < 0:
                    firstFrame = i
                lastFrame = i+1
            print('')

            # if i <= self.nFrames:
            #     pbar.update(i)
            i += 1
            # Just for show
            if show:
                cv2.drawChessboardCorners(frame, chessboardSize, imagePoints[i-1], success)
                cv2.imshow('Corners', frame)
                cv2.waitKey(2)
        self.nFrames = i
        # pbar.finish()

        return imagePoints, skipped, firstFrame, lastFrame

    def _genChessboardWorldPoints(self, dim, size):
        ''' Generate the world points of a chessboard from the given parameters

        :param dim: (cols, rows) size of the chessboard
        :param size: The length of the chessboard rect edges (scalar or 2 dim vector)
        '''
        if hasattr(size, '__len__'):
            l,w = size
        else:
            l,w = size,size

        N = dim[0] * dim[1]
        pts = np.empty((N,3))
        i = 0
        for v in range(dim[1]):
            for u in range(dim[0]):
                pts[i,0] = u*l
                pts[i,1] = -v*w
                pts[i,2] = 0
                i += 1

        return pts.astype(np.float32)

        # pnts = [[float(j)*l,dim[0]+1-float(i)*w,0.] for i in range(dim[1]) for j in range(dim[0])]
        # return np.array(pnts, dtype=np.float32)

    def _loadFromMatlab(self, filename):
        trackFile = self.videoDir + '/chessTrack_matlab.npz'
        matlabData = io.loadmat(filename)
        np.savez(trackFile, imagePoints=imagePoints, skipped=skipped,
                firstFrame=firstFrame, lastFrame=lastFrame)


def _quickTest():
    from CameraVisualizer import CameraVisualizer
    cameraMatrix = DeviceInfo.loadCameraMatrix('ipad', (360,480))[0]
    # cameraMatrix = DeviceInfo.loadCameraMatrix('ipad', (480,720))

    tracker = PoseChessboard('data/test_motions/ipad16.mp4', cameraMatrix)
    transforms, timestamps, skipped = tracker.track((8,6), forceRecalculate=True)

    camVis = CameraVisualizer(windowSize=(500,500))

    n = min(1000, len(transforms))
    # camVis.startVideoCapture(videoName="ipad09")
    print(n, 'Starting visualisation loop')
    for i in range(n):
        # print('Drawing:')
        camVis.setCameraTransform(transforms[i])
        time.sleep(0.01)

    print('End vis loop')
    # camVis.endVideoCapture()


def _getMatlab():
    cameraMatrix = DeviceInfo.loadCameraMatrix('ipad', (360,480))
    tracker = PoseChessboard('data/test_motions/ipad09.mp4', cameraMatrix)
    matlabFile = '../../matlab/results/motion_tests/trackMotion09_points_noframes.mat'
    tracker._loadFromMatlab(matlabFile)


def _frameCount():
    for i in range(1,10):
        trackfile = 'data/test_motions/ipad%02d/chessTrack.npz' % i
        compressed = np.load(trackfile)
        print('For', trackfile)
        print('Frame Count:', len(compressed['imagePoints']), '\n')


if __name__ == '__main__':
    import Geometry as Geom
    import sys

    seq = '../data/ch_glass_01'

    # iPhone
    K = zeros((3,3))
    fx, fy, cx, cy = IO.read_cam_param( os.path.join(seq, 'cam.yaml') )
    # mult = 1920/1280
    K[0,0] = K[1,1] = fx
    K[0,2] = cx
    K[1,2] = cy
    K[2,2] = 1
    if len(sys.argv) > 1:
        videoFile = os.path.join('../data', sys.argv[1], 'seq')
    else:
        videoFile = os.path.join(seq, 'seq')

    distCoeffs = np.c_[0.0, 0, 0, 0, 0].T
    tracker = PoseChessboard(videoFile, cameraMatrix=K, distCoeffs=distCoeffs)
    tracker.track(chessboardSize=(8,6),
                  squareSize=(0.0239,0.0239),
                  forceRetrack=False,
                  force=True,
                  showTrack=True,
                  calibrate=0,
                  transpose=True)
