"""Ball 3D position / height per frame from the 2D ball boxes and the calibrated camera (ball/camera.npz).
Within each shot the track is split greedily into flights; each flight is fit with a ballistic curve
p(t) = p0 + v0 t + 1/2 g t^2 (gravity known -> metric scale) minimising reprojection error, with the
ball-size distance (f * BALL_D / box size) as a weak prior. Frames no flight explains fall back to the size cue.
-> ball/ball_height.csv (frame,time_sec,X,Y,height_m,method,resid_px)
   method: flight (ballistic fit) | size (box-size distance only) | missing
usage: python ball/ball_height.py [START END]"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib import Camera, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 25
G = 9.81
BALL_D = 0.22          # m, size-5 football
SIZE_SIGMA = 0.10      # relative error of the size distance
MIN_LEN = 6            # frames in a flight
MAX_GAP = 4            # missing frames a flight may bridge
ADD_TOL = 6.0          # px: a new point that the refit misses by more than this ends the flight
RMS_TOL = 3.5          # px: flight rms limit


def size_points(cam, uv, size_px):
    d = cam.rays(uv)
    dist = cam.f * BALL_D / size_px
    return cam.C + dist[:, None] * d, dist


def ballistic(q, t):
    t = t[:, None]
    return q[:3] + q[3:] * t + 0.5 * np.array([0, 0, G]) * t ** 2


def fit_flight(cam, t, uv, dist):
    """q = (p0, v0) at t[0]; returns q, per-point reprojection error"""
    t = t - t[0]
    P, _ = cam.C + dist[:, None] * cam.rays(uv), None
    # linear init from the size-cue points
    A = np.c_[np.ones_like(t), t]
    q0 = np.r_[[np.linalg.lstsq(A, P[:, k] - (0.5 * G * t ** 2 if k == 2 else 0), rcond=None)[0] for k in range(3)]].T.ravel()
    q0 = np.r_[q0[0], q0[2], q0[4], q0[1], q0[3], q0[5]]

    def res(q):
        X = ballistic(q, t)
        r = (cam.project(X) - uv).ravel()
        rng = np.linalg.norm(X - cam.C, axis=1)
        return np.r_[r, (rng - dist) / (SIZE_SIGMA * dist)]
    q = least_squares(res, q0, x_scale='jac').x
    e = np.linalg.norm(cam.project(ballistic(q, t)) - uv, axis=1)
    return q, e


def process_shot(cam, g):
    """g: rows of one shot with a ball box. returns dict frame -> (X, Y, Z, method, resid)"""
    fr = g.frame.values; t = fr / FPS
    uv = np.c_[g.center_x, g.center_y].astype(float)
    size = ((g.x2 - g.x1) + (g.y2 - g.y1)).values / 2
    Psz, dist = size_points(cam, uv, size)
    out = {f: (*Psz[i], 'size', np.nan) for i, f in enumerate(fr)}
    i, n = 0, len(fr)
    while i < n:
        j = i + MIN_LEN
        if j > n or fr[j - 1] - fr[i] > (MIN_LEN - 1) * (MAX_GAP + 1):
            i += 1; continue
        q, e = fit_flight(cam, t[i:j], uv[i:j], dist[i:j])
        if np.sqrt(np.mean(e ** 2)) > RMS_TOL:
            i += 1; continue
        while j < n and fr[j] - fr[j - 1] <= MAX_GAP + 1:
            q2, e2 = fit_flight(cam, t[i:j + 1], uv[i:j + 1], dist[i:j + 1])
            if e2[-1] > ADD_TOL or np.sqrt(np.mean(e2 ** 2)) > RMS_TOL:
                break
            q, e, j = q2, e2, j + 1
        X = ballistic(q, t[i:j] - t[i])
        for k in range(i, j):
            out[fr[k]] = (*X[k - i], 'flight', e[k - i])
        i = j
    return out


def compute(start=0, end=None):
    cam = Camera(*load())
    b = pd.read_csv(os.path.join(ROOT, 'correct_bbox', 'ball_boxes_corrected3.csv'))
    b = b[(b.frame >= start) & (b.frame < (end if end is not None else b.frame.max() + 1))]
    res = {}
    for _, g in b[b.source != 'missing'].groupby('shot'):
        res.update(process_shot(cam, g))
    rows = []
    for f, ts in zip(b.frame, b.time_sec):
        if f in res:
            X, Y, Z, m, e = res[f]
            rows.append((f, ts, X, Y, -Z, m, e))
        else:
            rows.append((f, ts, np.nan, np.nan, np.nan, 'missing', np.nan))
    return pd.DataFrame(rows, columns=['frame', 'time_sec', 'X', 'Y', 'height_m', 'method', 'resid_px'])


if __name__ == '__main__':
    a = [int(x) for x in sys.argv[1:3]]
    df = compute(*a)
    out = os.path.join(ROOT, 'ball', 'ball_height.csv')
    df.round(3).to_csv(out, index=False)
    print(df.method.value_counts().to_string()); print(df.height_m.describe().to_string())
    print('->', out)
