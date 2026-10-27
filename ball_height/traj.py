"""Ballistic 3-D fits of the ball from one calibrated camera.

World frame: X along the court, Y toward the camera, Z down (height = -Z).
Between two touches the ball follows dp/dt = v, dv/dt = g - k|v|v.
Gravity plus the known frame rate fix the metric scale of each flight.
"""
import numpy as np, cv2
from scipy.optimize import least_squares

G = 9.81
FPS = 25.0


class Camera:
    def __init__(self, path='ball_height/camera.npz'):
        c = np.load(path)
        self.K, self.dist, self.R, self.t, self.C = c['K'], c['dist'], c['R'], c['t'], c['C']
        self.rv = cv2.Rodrigues(self.R)[0]

    def project(self, X):  # X: (n,3) -> (n,2), with lens distortion
        u, _ = cv2.projectPoints(np.asarray(X, float).reshape(-1, 3), self.rv, self.t, self.K, self.dist)
        return u[:, 0, :]

    def ray(self, u):  # unit world directions through pixels (n,2)
        n = cv2.undistortPoints(np.asarray(u, float).reshape(-1, 1, 2), self.K, self.dist)[:, 0, :]
        d = (self.R.T @ np.vstack([n.T, np.ones(len(n))])).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def ground(self, u):  # pixel -> point on the sand (Z = 0)
        d = self.ray(u)
        return self.C + (-self.C[2] / d[:, 2])[:, None] * d


def simulate(p0, v0, t, g=G, k=0.0, sub=4):
    """Positions at times t (seconds from the segment start, sorted)."""
    if k == 0.0:
        a = np.array([0, 0, g])
        return p0 + np.outer(t, v0) + 0.5 * np.outer(t ** 2, a)
    out, p, v, now = [], p0.astype(float), v0.astype(float), 0.0
    for ti in t:
        while now < ti - 1e-9:
            h = min(1 / FPS / sub, ti - now)
            acc = lambda vv: np.array([0, 0, g]) - k * np.linalg.norm(vv) * vv
            # RK2 step
            a1 = acc(v); vm = v + 0.5 * h * a1
            p = p + h * vm; v = v + h * acc(vm); now += h
        out.append(p.copy())
    return np.array(out)


def fit(cam, t, uv, init=None, k=0.0, free_g=False, w=None):
    """Least-squares flight fit. Returns (params, residuals_px)."""
    t = t - t[0]
    if init is None:
        init = initial_guess(cam, t, uv)
    x0 = np.r_[init, G] if free_g else init

    def res(x):
        g = x[6] if free_g else G
        P = simulate(x[:3], x[3:6], t, g, k)
        r = cam.project(P) - uv
        if w is not None:
            r = r * w[:, None]
        return r.ravel()

    r = least_squares(res, x0, loss='soft_l1', f_scale=3.0, x_scale='jac')
    P = simulate(r.x[:3], r.x[3:6], t, r.x[6] if free_g else G, k)
    return r.x, np.linalg.norm(cam.project(P) - uv, axis=1)


def initial_guess(cam, t, uv, depth=None):
    """Seed by placing the first/last points at a guessed depth along their rays."""
    d = cam.ray(uv[[0, -1]])
    dist = 12.0 if depth is None else depth
    a, b = cam.C + dist * d[0], cam.C + dist * d[1]
    T = max(t[-1], 1e-3)
    v = (b - a) / T - 0.5 * np.array([0, 0, G]) * T
    return np.r_[a, v]
