"""Full 3D camera (fixed camera) from the ground homography plus the net.
Court frame: X along the court 0-18 (net at X=9), Y across 0 (far sideline) - 9 (near sideline), Z down (height = -Z).
Fits f, k1, R, t to: ground grid points from homography_px_to_m.npy, the near net-post top and the net top tape
(seen almost edge-on, running from the near post top to the far post), both at NET_H.
-> ball/camera.npz (K, dist, rvec, tvec)
usage: python ball/calib.py [frame.png]   (optional: draws the reprojected court + net on that frame)"""
import os, sys
import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_H = 2.2                                            # m, men's footvolley
TAPE = np.array([[1013, 645], [1008, 500], [1003, 370]], float)   # px on the net top tape (first = near post top)
POST = np.array([[1022, 900], [1020, 780]], float)                # px on the near post below its top


def load():
    c = np.load(os.path.join(ROOT, 'ball', 'camera.npz'))
    return c['K'], c['dist'], c['rvec'], c['tvec']


def fit():
    G = np.linalg.inv(np.load(os.path.join(ROOT, 'homography_px_to_m.npy')))
    gp = np.array([[x, y, 0.] for x in np.arange(0, 18.1, 1) for y in np.arange(0, 9.1, 1)])
    px = cv2.perspectiveTransform(gp[:, None, :2].copy(), G)[:, 0]
    m = (px[:, 0] > 0) & (px[:, 0] < 1920) & (px[:, 1] < 1080)
    gp, px = gp[m], px[m]

    def proj(p, X):
        K = np.array([[p[0], 0, 960], [0, p[0], 540], [0, 0, 1.]])
        return cv2.projectPoints(X, p[2:5], p[5:8], K, np.array([p[1], 0, 0, 0.]))[0][:, 0]

    W = 3   # weight of the net points vs. the homography grid
    def res(p):
        yn, y2, y3, z1, z2 = p[8:13]
        T = np.array([[9, yn, -NET_H], [9, y2, -NET_H], [9, y3, -NET_H]])
        P = np.array([[9, yn, z1], [9, yn, z2]])
        return np.concatenate([(proj(p, gp) - px).ravel(), W * (proj(p, T) - TAPE).ravel(), W * (proj(p, P) - POST).ravel()])

    K0 = np.array([[1600, 0, 960], [0, 1600, 540], [0, 0, 1.]])
    _, r0, t0 = cv2.solvePnP(gp, px, K0, None)
    p = least_squares(res, np.r_[1600, 0, r0.ravel(), t0.ravel(), 9.5, 6, 2, -0.5, -1.2], x_scale='jac').x
    e = res(p); ng = 2 * len(gp)
    print(f'f={p[0]:.0f} k1={p[1]:.3f}  ground rms {np.sqrt(np.mean(e[:ng] ** 2)):.1f}px  net rms {np.sqrt(np.mean((e[ng:] / W) ** 2)):.1f}px')
    K = np.array([[p[0], 0, 960], [0, p[0], 540], [0, 0, 1.]])
    return K, np.array([p[1], 0, 0, 0.]), p[2:5], p[5:8]


class Camera:
    def __init__(self, K, dist, rvec, tvec):
        self.K, self.dist, self.rvec, self.tvec = K, dist, rvec, tvec
        self.R = cv2.Rodrigues(rvec)[0]
        self.C = -self.R.T @ tvec
        self.f = K[0, 0]

    def project(self, X):
        return cv2.projectPoints(np.asarray(X, float).reshape(-1, 3), self.rvec, self.tvec, self.K, self.dist)[0][:, 0]

    def rays(self, uv):
        n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), self.K, self.dist)[:, 0]
        d = (self.R.T @ np.c_[n, np.ones(len(n))].T).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ground(self, uv):
        d = self.rays(uv)
        return self.C + (-self.C[2] / d[:, 2])[:, None] * d


if __name__ == '__main__':
    K, dist, rvec, tvec = fit()
    np.savez(os.path.join(ROOT, 'ball', 'camera.npz'), K=K, dist=dist, rvec=rvec, tvec=tvec)
    cam = Camera(K, dist, rvec, tvec)
    print('camera position (m):', np.round(cam.C * [1, 1, -1], 2))
    if len(sys.argv) > 1:
        im = cv2.imread(sys.argv[1])
        P = lambda X: tuple(int(v) for v in cam.project(X)[0])
        for x in range(0, 19):
            cv2.line(im, P([x, 0, 0]), P([x, 9, 0]), (0, 255, 255), 1)
        for y in range(0, 10):
            cv2.line(im, P([0, y, 0]), P([18, y, 0]), (0, 255, 255), 1)
        for y in (0, 9):
            cv2.line(im, P([9, y, 0]), P([9, y, -NET_H]), (0, 0, 255), 2)
        cv2.line(im, P([9, 0, -NET_H]), P([9, 9, -NET_H]), (0, 0, 255), 2)
        for u, v in np.r_[TAPE, POST]:
            cv2.circle(im, (int(u), int(v)), 5, (255, 0, 255), 2)
        cv2.imwrite(sys.argv[1].replace('.png', '_calib.png'), im)
