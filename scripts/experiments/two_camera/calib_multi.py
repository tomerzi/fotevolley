"""Experiment: several replay screenshots from the camera behind the X=0 baseline.

Per screenshot (data/experiments/two_camera/shots.json): sideline/tape pixel points, players' feet
(and head top when upright) matched to their main-camera ground position at the synchronized v2 frame,
the ball centre/diameter, and the main-camera ball pixel at candidate frames.
Each screenshot gets its own rotation and focal length (the replay camera pans and zooms); the camera
position can be shared (SHARED_C=1, a fixed tripod) or free per screenshot.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_cam2 import rot, look_at, from_R, main, ROOT, D, stature

SHOTS = json.load(open(os.path.join(D, 'shots.json')))
SHARED = os.environ.get('SHARED_C', '1') == '1'
FIXC = np.array([float(v) for v in os.environ['FIXC'].split(',')]) if os.environ.get('FIXC') else None
boxes = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_corrected3.csv')).set_index('frame')
site = pd.read_csv(os.path.join(ROOT, 'data', 'analysis', 'ball_height_v2.csv')).set_index('frame')


def proj(R, C, f, cxy, P):
    c = (np.atleast_2d(P) - C) @ R.T
    return np.c_[f * c[:, 0] / c[:, 2] + cxy[0], f * c[:, 1] / c[:, 2] + cxy[1]]


def ldist(R, C, f, cxy, A, B, pts):
    a, b = proj(R, C, f, cxy, np.array([A, B]))
    n = np.array([b[1] - a[1], a[0] - b[0]]); n /= np.linalg.norm(n)
    return (np.asarray(pts) - a) @ n


def shot_res(R, C, f, s, noise=None):
    cxy = np.array(s['size']) / 2
    r = []
    L = s.get('lines', {})
    seg = {'left': ([0, 0, 0], [16, 0, 0]), 'right': ([0, 8, 0], [16, 8, 0]), 'tape': ([8, -1, -2.2], [8, 9, -2.2])}
    sig = {'left': 1.0, 'right': 1.0, 'tape': 2.0}
    for k, pts in L.items():
        pts = np.asarray(pts, float)
        if noise is not None:
            pts = pts + noise.normal(0, 1.0, pts.shape)
        r += list(ldist(R, C, f, cxy, *seg[k], pts) / sig[k])
    for p in s.get('players', []):
        g = np.array(p['ground'], float)
        if noise is not None:
            g = g + noise.normal(0, 0.15, 2)
        if p.get('feet') is not None:
            r += list((proj(R, C, f, cxy, [g[0], g[1], 0])[0] - p['feet']) / p.get('sig', 5.0))
        if p.get('head') is not None:
            st = stature[p['player']]['stature_m']
            r += list((proj(R, C, f, cxy, [g[0], g[1], -st])[0] - p['head']) / p.get('sig', 5.0))
    return r


def unpack(x, n):
    out, k = [], 0
    if FIXC is not None:
        Cs, k = FIXC, 0
    else:
        Cs = x[:3] if SHARED else None
        k = 3 if SHARED else 0
    for i in range(n):
        R = rot(x[k:k + 3]); f = x[k + 3]; k += 4
        if not SHARED and FIXC is None:
            C = x[k:k + 3]; k += 3
        else:
            C = Cs
        out.append((R, C, f))
    return out


def fit(shots, noise=None, C0=np.array([0.0, 7.3, -2.85])):
    def res(x):
        r = []
        for (R, C, f), s in zip(unpack(x, len(shots)), shots):
            r += shot_res(R, C, f, s, noise)
        return np.array(r)
    best = None
    for dc in ([0, 0, 0], [-3, 0, 0], [0, -2, 0], [-6, 1, -1]):
        C = C0 + dc
        if FIXC is not None:
            C = FIXC
        x0 = list(C) if (SHARED and FIXC is None) else []
        for s in shots:
            aim = np.array(s.get('aim', [12, 4, 0]), float)
            x0 += list(from_R(look_at(aim - C))) + [s.get('f0', 1300)]
            if not SHARED and FIXC is None:
                x0 += list(C)
        sol = least_squares(res, np.array(x0), loss='soft_l1', f_scale=2.0)
        if best is None or sol.cost < best.cost:
            best = sol
    return best, unpack(best.x, len(shots))


def tri(R, C, f, s, frame, noise=None):
    cxy = np.array(s['size']) / 2
    b = boxes.loc[frame]
    uv = np.array([b.center_x, b.center_y])
    if noise is not None:
        uv = uv + noise.normal(0, 1.5, 2)
    ball = np.array(s['ball'], float)
    if noise is not None:
        ball = ball + noise.normal(0, 1.5, 2)
    ray = main.ray(np.array([uv]))[0]

    def r(t):
        return (proj(R, C, f, cxy, main.C + t[0] * ray)[0] - ball)
    sol = min((least_squares(r, [t0]) for t0 in (8, 12, 16)), key=lambda q: q.cost)
    P = main.C + sol.x[0] * ray
    return P, float(np.sqrt(np.mean(sol.fun ** 2)))


if __name__ == '__main__':
    shots = [s for s in SHOTS if s.get('use', True)]
    sol, cams = fit(shots)
    print('shared C' if SHARED else 'free C', 'cost', round(sol.cost, 1))
    rows = []
    for (R, C, f), s in zip(cams, shots):
        r = np.array(shot_res(R, C, f, s))
        print('%-8s C=(%.2f, %.2f, h %.2f) f=%.0f  rms %.2f' % (s['name'], C[0], C[1], -C[2], f, np.sqrt(np.mean(r ** 2))))
        for fr in s['frames']:
            P, e = tri(R, C, f, s, fr)
            st = site.loc[fr]
            rows.append(dict(shot=s['name'], frame=fr, X=P[0], Y=P[1], h=-P[2], reproj=e,
                             site_X=st.court_x_m, site_Y=st.court_y_m, site_h=st.height_m))
    df = pd.DataFrame(rows)
    # Monte Carlo
    rng = np.random.default_rng(1)
    mc = []
    for it in range(int(os.environ.get('MC', 100))):
        _, cs = fit(shots, noise=rng)
        for (R, C, f), s in zip(cs, shots):
            for fr in s['frames']:
                P, _ = tri(R, C, f, s, fr, noise=rng)
                mc.append(dict(shot=s['name'], frame=fr, h=-P[2], Y=P[1]))
    if mc:
        mc = pd.DataFrame(mc).groupby(['shot', 'frame']).agg(sh=('h', 'std'), sY=('Y', 'std')).reset_index()
        df = df.merge(mc, on=['shot', 'frame'])
    df['dh'] = df.site_h - df.h
    df['dY'] = df.site_Y - df.Y
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(os.path.join(D, 'multi_result%s.csv' % ('_fixC' if FIXC is not None else ('' if SHARED else '_freeC'))), index=False)
