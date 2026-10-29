"""Court/net/antenna landmarks and the camera model of calib.py, importable (calib.py runs on import).
Camera vector c = [f, cx, cy, k1, k2, rvec(3), t(3)]."""
import numpy as np, cv2

W, H_IMG, NET_H = 1920, 1080, 2.20
FAR = [(260, 419), (410, 419), (560, 418), (710, 419), (1010, 418), (1160, 421), (1310, 421), (1460, 420), (1610, 422)]
NEAR = [(260, 967), (380, 971), (500, 976), (620, 979), (740, 982), (860, 984), (980, 987), (1100, 986),
        (1220, 986), (1340, 987), (1460, 986), (1580, 984), (1700, 983)]
LEFT = [(203, 455), (145, 515), (114, 545), (89, 575), (62, 605), (32, 635)]
RIGHT = [(1825, 450), (1874, 500), (1902, 525)]
LINES = [(FAR, 1, 0.0), (NEAR, 1, 8.0), (LEFT, 0, 0.0), (RIGHT, 0, 16.0)]
CORNERS = [((232, 420), (0.0, 0.0)), ((1790, 422), (16.0, 0.0))]
NET_TAPE = [(1010, 400), (1011, 450), (1011, 500), (1012, 550), (1012, 600)]
ANTENNA = [((1018, y), hh) for y, hh in zip([486, 505, 524, 543, 561, 580, 598, 617], [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3])]


def from_camera(cam):
    return np.r_[cam.K[0, 0], cam.K[0, 2], cam.K[1, 2], cam.dist[0], cam.dist[1], cam.rv.ravel(), cam.t.ravel()]


def parts(c):
    R, _ = cv2.Rodrigues(np.asarray(c[5:8], float))
    return c[0], c[1], c[2], c[3], c[4], R, np.asarray(c[8:11], float)


def center(c):
    f, cx, cy, k1, k2, R, t = parts(c)
    return -R.T @ t


def project(c, X):
    f, cx, cy, k1, k2, R, t = parts(c)
    Xc = np.atleast_2d(X) @ R.T + t
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[f * x * d + cx, f * y * d + cy]


def rays(c, uv):
    f, cx, cy, k1, k2, R, t = parts(c)
    xd, yd = (uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(8):
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        x, y = xd / d, yd / d
    v = np.c_[x, y, np.ones_like(x)] @ R
    return -R.T @ t, v / np.linalg.norm(v, axis=1, keepdims=True)


def on_plane(c, uv, z):
    C, v = rays(c, uv)
    s = (z - C[2]) / v[:, 2]
    return C + s[:, None] * v


def residuals(c):
    """the calibration's own landmark terms (same weights as calib.py)"""
    r = []
    for pts, axis, val in LINES:
        r.append((on_plane(c, np.array(pts, float), 0.0)[:, axis] - val) * 60)
    for uv, XY in CORNERS:
        r.append(project(c, np.array([[XY[0], XY[1], 0.0]]))[0] - uv)
    r.append((on_plane(c, np.array(NET_TAPE, float), -NET_H)[:, 0] - 8.0) * 60)
    for uv, hh in ANTENNA:
        r.append(project(c, np.array([[8.0, 8.0, -hh]]))[0] - uv)
    r.append([(c[1] - W / 2) / 60.0, (c[2] - H_IMG / 2) / 60.0, c[3] / 0.3, c[4] / 0.3])
    return np.concatenate([np.ravel(a) for a in r])
