"""Experiment: calibrate the broadcast replay camera (behind the ISR baseline) from one
screenshot and triangulate the ball at the bicycle-kick contact of rally 36 (v2 frame ~9638)
together with the main camera.

Inputs (pixel readings in the 554x313 screenshot, see README):
  - left blue sideline (Y=0, Z=0) and right blue sideline (Y=8, Z=0): sub-pixel ridge points
  - net top tape upper edge (X=8, Z=-2.2)
  - BRA_A standing: feet + head top, ground position from the main camera, stature 1.791 m
  - ball centre and diameter in the screenshot
World frame as in the main pipeline: X along the court (0..16), Y toward the main camera,
Z down (height = -Z).
"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'ball_height'))
from traj import Camera

D = os.path.join(ROOT, 'data', 'experiments', 'two_camera')
obs = json.load(open(os.path.join(D, 'cam2_points.json')))
W, H = obs['image_size']
CX, CY = W / 2, H / 2
main = Camera(os.path.join(ROOT, 'data', 'analysis', 'camera_v2.npz'))
stature = json.load(open(os.path.join(ROOT, 'data', 'analysis', 'player_statures.json')))


def rot(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def look_at(d):
    z = d / np.linalg.norm(d)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def from_R(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def project(p, P):
    R, C, f = rot(p[:3]), p[3:6], p[6]
    c = (np.atleast_2d(P) - C) @ R.T
    return np.c_[f * c[:, 0] / c[:, 2] + CX, f * c[:, 1] / c[:, 2] + CY]


def line_dist(p, A, B, pts):
    a, b = project(p, np.array([A, B]))
    n = np.array([b[1] - a[1], a[0] - b[0]]); n /= np.linalg.norm(n)
    return (pts - a) @ n


def residuals(p, o, gp):
    r = []
    r += list(line_dist(p, [0, 0, 0], [8, 0, 0], o['left']) / o['sig_line'])
    r += list(line_dist(p, [0, 8, 0], [8, 8, 0], o['right']) / o['sig_line'])
    r += list(line_dist(p, [8, -1, -2.2], [8, 9, -2.2], o['tape']) / o['sig_tape'])
    X, Y = gp
    s = stature['BRA_A']['stature_m']
    r += list((project(p, [X, Y, 0])[0] - o['bra_feet']) / o['sig_player'])
    r += list((project(p, [X, Y, -s])[0] - o['bra_head']) / o['sig_player'])
    return np.array(r)


def calibrate(o, gp):
    best = None
    for cy in (2, 5, 8, 11):
        for ch in (2.5, 4, 6):
            C0 = np.array([-8.0, cy, -ch])
            R0 = look_at(np.array([12, 4, 0]) - C0)
            p0 = np.r_[from_R(R0), C0, 870]
            s = least_squares(residuals, p0, args=(o, gp), x_scale=np.r_[0.05, 0.05, 0.05, 1, 1, 1, 50])
            if best is None or s.cost < best.cost:
                best = s
    return best


def triangulate(p2, uv_main, uv2, diam2=None, sig2=1.0):
    ray = main.ray(np.array([uv_main]))[0]
    C1 = main.C

    def res(x):
        P = C1 + x[0] * ray
        r = list((project(p2, P)[0] - uv2) / sig2)
        if diam2 is not None:
            dist = np.linalg.norm(P - p2[3:6])
            r.append((p2[6] * 0.22 / dist - diam2[0]) / diam2[1])
        return r
    best = min((least_squares(res, [t0]) for t0 in (8, 12, 16)), key=lambda s: s.cost)
    return C1 + best.x[0] * ray, best


def load(o):
    q = dict(o)
    for k in ('left', 'right', 'tape'):
        q[k] = np.array(o[k], float)
    for k in ('bra_feet', 'bra_head', 'ball'):
        q[k] = np.array(o[k], float)
    return q


if __name__ == '__main__':
    o = load(obs)
    gp = np.array(o['bra_ground_main'])
    s = calibrate(o, gp)
    p = s.x
    r = residuals(p, o, gp)
    print('cam2 C = X %.2f Y %.2f height %.2f  f %.0f px  rms(norm) %.2f' % (p[3], p[4], -p[5], p[6], np.sqrt(np.mean(r ** 2))))
    out = {'cam2': p.tolist()}
    for f, uv in o['main_ball'].items():
        P, _ = triangulate(p, uv, o['ball'], (o['ball_diam'], o['ball_diam_sig']))
        P0, _ = triangulate(p, uv, o['ball'])
        print('frame %s: triangulated X %.2f Y %.2f height %.2f   (no size prior: X %.2f Y %.2f h %.2f)' % (f, *P[:2], -P[2], *P0[:2], -P0[2]))
        out[f] = P.tolist()

    # Monte Carlo: pixel noise, BRA_A ground position from the main camera, ball diameter
    rng = np.random.default_rng(0)
    mc = {f: [] for f in o['main_ball']}
    cams = []
    for _ in range(300):
        q = dict(o)
        for k in ('left', 'right', 'tape'):
            q[k] = o[k] + rng.normal(0, 1.0, o[k].shape)
        q['tape'] = q['tape'] + np.c_[np.zeros(len(o['tape'])), rng.normal(0, 1.0) + np.linspace(-1, 1, len(o['tape'])) * rng.normal(0, 0.7)]
        q['bra_feet'] = o['bra_feet'] + rng.normal(0, 2.0, 2)
        q['bra_head'] = o['bra_head'] + rng.normal(0, 2.0, 2)
        g = gp + rng.normal(0, 0.15, 2)
        sc = calibrate(q, g)
        cams.append(sc.x)
        b2 = o['ball'] + rng.normal(0, 1.0, 2)
        d2 = o['ball_diam'] + rng.normal(0, o['ball_diam_sig'])
        for f, uv in o['main_ball'].items():
            P, _ = triangulate(sc.x, np.array(uv) + rng.normal(0, 1.5, 2), b2, (d2, o['ball_diam_sig']))
            mc[f].append(P)
    cams = np.array(cams)
    print('cam2 MC: C X %.2f±%.2f Y %.2f±%.2f h %.2f±%.2f f %.0f±%.0f' % (cams[:, 3].mean(), cams[:, 3].std(), cams[:, 4].mean(), cams[:, 4].std(), -cams[:, 5].mean(), cams[:, 5].std(), cams[:, 6].mean(), cams[:, 6].std()))
    summ = {}
    for f, L in mc.items():
        L = np.array(L)
        m, sd = L.mean(0), L.std(0)
        print('frame %s MC: X %.2f±%.2f  Y %.2f±%.2f  height %.2f±%.2f' % (f, m[0], sd[0], m[1], sd[1], -m[2], sd[2]))
        summ[f] = dict(X=m[0], Y=m[1], height=-m[2], sX=sd[0], sY=sd[1], sh=sd[2])
    out['mc'] = summ
    json.dump(out, open(os.path.join(D, 'result.json'), 'w'), indent=1)
