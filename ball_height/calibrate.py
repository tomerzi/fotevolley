"""Camera calibration from the four court lines (16 m x 8 m rectangle on the sand plane Z=0).

Model: pinhole + one radial distortion term, principal point at the image centre.
Unknowns: focal length, k1, pan/tilt/roll, camera position (8 numbers).
Data: sub-pixel samples along the far/near sidelines and left/right baselines (see lines.py).
The residual of a sample is its distance (converted to pixels) from the court line it belongs to.

usage: python calibrate.py court_line_points.csv camera.json
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera, lm

X0 = [1657.0, -0.11, -0.024, 0.493, -0.007, 7.82, -7.15, 6.77]


def load_points(path):
    d = pd.read_csv(path)
    keep = []
    for n, g in d.groupby('line'):
        a, b = (g.u.values, g.v.values) if n in ('far', 'near') else (g.v.values, g.u.values)
        k = np.ones(len(a), bool)
        for _ in range(3):                                   # drop samples off a smooth curve (sand over the tape etc.)
            p = np.polyfit(a[k], b[k], 2); r = b - np.polyval(p, a); k = np.abs(r) < 2.5
        keep.append(g[k])
    d = pd.concat(keep)
    return d[['u', 'v']].values, d.line.values


def line_residuals(cam, uv, line):
    P = cam.to_plane(uv, 0.0)
    r = np.where(line == 'far', P[:, 1] - 8, np.where(line == 'near', P[:, 1], np.where(line == 'left', P[:, 0], P[:, 0] - 16)))
    dP = np.zeros_like(P); isy = (line == 'far') | (line == 'near'); dP[isy, 1] = 0.05; dP[~isy, 0] = 0.05
    s = np.linalg.norm(cam.project(P + dP) - cam.project(P), axis=1) / 0.05
    return r * s


def fit(uv, line, fixed=None, x0=X0):
    """fixed: dict name->value of parameters held fixed (e.g. {'f': 1700})"""
    fixed = fixed or {}
    names = Camera.names[:8]
    free = [n for n in names if n not in fixed]
    w = np.ones(len(uv))
    for n in set(line):
        w[line == n] = 1 / np.sqrt((line == n).sum())       # each line counts the same

    def mk(q):
        d = dict(fixed); d.update(zip(free, q))
        return Camera(*[d[n] for n in names], d.get('cx', 960), d.get('cy', 540), d.get('k2', 0.0))
    q0 = [x0[names.index(n)] for n in free]
    q, r, _ = lm(lambda q: line_residuals(mk(q), uv, line) * w, q0, iters=200)
    cam = mk(q)
    return cam, line_residuals(cam, uv, line)


if __name__ == '__main__':
    uv, line = load_points(sys.argv[1])
    cam, r = fit(uv, line)
    print({n: round(float(v), 5) for n, v in zip(Camera.names, cam.params())})
    for n in ['far', 'near', 'left', 'right']:
        print('  %-5s n=%3d rms %.2f px' % (n, (line == n).sum(), r[line == n].std()))
    cam.to_json(sys.argv[2], extra=dict(line_rms_px=float(r.std()), note='court lines only'))
