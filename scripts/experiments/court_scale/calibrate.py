"""EXPERIMENT (court scale): same as scripts/ball_height/calibrate.py, but the blue lines are taken as
(16 x 8) * COURT_SCALE m (env), and the outputs go to $OUT (env).

Full camera calibration (with lens distortion) for the main camera.

Court: 16 x 8 m. World frame: X along the court (left baseline = 0),
Y toward the camera (far sideline = 0, near sideline = 8), Z down, so a
height h is Z = -h.

Constraints:
  * pixels measured on the blue court lines (median frame, see notes below),
  * the two far corners,
  * the net: its top edge is 2.20 m where it meets the near antenna,
  * upright players: ankles on the sand, neck at NECK_RATIO * stature.
    The Israeli players' statures are known; the Brazilian players' are
    fitted as two extra unknowns and written to data/analysis/player_statures.json.
The near sideline bows by ~20 px, so radial distortion is fitted too.
Writes data/analysis/camera_v2.npz.
"""
import os
import numpy as np, pandas as pd, cv2
from scipy.optimize import least_squares

W, H_IMG = 1920, 1080
import json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)  # all paths below are relative to the repo root


# Known statures (given). BRA_A / BRA_B are solved for.
STATURE = {'ISR_A': 1.86, 'ISR_B': 1.86}   # Ron Ben Ishay, Maor Haas
FREE = ['BRA_A', 'BRA_B']
NET_H = 2.20
NET_TOP_NEAR = (1008, 622)  # net top at the near antenna (just above the post cap), X = 8, Y = 8 (x scale)
NECK_RATIO = 0.82  # neck (shoulder midpoint) height / stature, upright adult

# Blue-line pixels picked on the median of frames 0..720 (peak blueness per column/row).
FAR = [(260, 419), (410, 419), (560, 418), (710, 419), (1010, 418), (1160, 421),
       (1310, 421), (1460, 420), (1610, 422)]                       # Y = 0
NEAR = [(260, 967), (380, 971), (500, 976), (620, 979), (740, 982), (860, 984),
        (980, 987), (1100, 986), (1220, 986), (1340, 987), (1460, 986),
        (1580, 984), (1700, 983)]                                   # Y = 8
LEFT = [(203, 455), (145, 515), (114, 545), (89, 575), (62, 605), (32, 635)]  # X = 0
RIGHT = [(1825, 450), (1874, 500), (1902, 525)]                     # X = 16
CORNERS = [((232, 420), (0, 0)), ((1790, 422), (16 * float(__import__('os').environ.get('COURT_SCALE', 1.0)), 0))]

SC = float(os.environ.get('COURT_SCALE', 1.0))
OUTD = os.environ.get('OUT', 'data/experiments/court_scale')
LINES = [(FAR, 1, 0.0), (NEAR, 1, 8.0 * SC), (LEFT, 0, 0.0), (RIGHT, 0, 16.0 * SC)]


def unpack(p):
    f, cx, cy, k1, k2 = p[:5]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K, np.array([k1, k2, 0, 0, 0]), p[5:8], p[8:11]


def project(p, X):
    K, dist, rv, t = unpack(p)
    u, _ = cv2.projectPoints(np.asarray(X, float).reshape(-1, 3), rv, t, K, dist)
    return u[:, 0, :]


def to_ground(p, uv):
    K, dist, rv, t = unpack(p)
    n = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, dist)[:, 0, :]
    R, _ = cv2.Rodrigues(rv)
    C = -R.T @ t
    d = (R.T @ np.vstack([n.T, np.ones(len(n))])).T
    s = -C[2] / d[:, 2]
    return C + s[:, None] * d


# Upright players from the pose file.
P = pd.read_csv('data/pose/vitpose_body25.csv')
def ang(a, b, c):
    v1 = np.stack([P[a + '_x'] - P[b + '_x'], P[a + '_y'] - P[b + '_y']], 1)
    v2 = np.stack([P[c + '_x'] - P[b + '_x'], P[c + '_y'] - P[b + '_y']], 1)
    with np.errstate(invalid='ignore'):
        cos = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))
kp = ['neck', 'mid_hip', 'r_hip', 'l_hip', 'r_knee', 'l_knee', 'r_ankle', 'l_ankle']
ok = np.all([P[k + '_s'] > 0.7 for k in kp], axis=0)
ok &= ang('r_hip', 'r_knee', 'r_ankle') > 170
ok &= ang('l_hip', 'l_knee', 'l_ankle') > 170
ok &= np.abs(np.degrees(np.arctan2(P.neck_x - P.mid_hip_x, P.mid_hip_y - P.neck_y))) < 6
ok &= (P.r_ankle_y - P.l_ankle_y).abs() < 8
S = P[ok]
feet = np.stack([(S.r_ankle_x + S.l_ankle_x) / 2, (S.r_ankle_y + S.l_ankle_y) / 2 + 6], 1)
neck = S[['neck_x', 'neck_y']].values
free_idx = np.array([FREE.index(pl) if pl in FREE else -1 for pl in S.player])


def neck_heights(p):
    st = S.player.map(STATURE).values.astype(float)
    m = free_idx >= 0
    st[m] = p[11 + free_idx[m]]
    return NECK_RATIO * st
print('upright samples', len(S))

PX_PER_M = 60.0  # turns ground-coordinate errors (m) into roughly pixel units


def resid(p):
    r = []
    for pts, axis, val in LINES:
        g = to_ground(p, pts)
        r.append((g[:, axis] - val) * PX_PER_M)
    for uv, XY in CORNERS:
        r.append(project(p, [[XY[0], XY[1], 0]])[0] - uv)
    g = to_ground(p, feet)
    g[:, 2] = -neck_heights(p)
    r.append(((project(p, g) - neck) / np.sqrt(len(S) / 30)).ravel())
    r.append((project(p, [[8.0 * SC, 8.0 * SC, -NET_H]])[0] - NET_TOP_NEAR) * 3)
    r.append([(p[1] - W / 2) / 2, (p[2] - H_IMG / 2) / 2, p[4] / 0.02])  # principal point prior
    return np.concatenate([np.ravel(x) for x in r])


def init(f0):
    K = np.array([[f0, 0, W / 2], [0, f0, H_IMG / 2], [0, 0, 1.0]])
    obj = [(0, 0, 0), (16 * SC, 0, 0), (8 * SC, 0, 0), (0, 4.78 * SC, 0), (3.6 * SC, 8 * SC, 0), (11.6 * SC, 8 * SC, 0)]
    img = [(232, 420), (1790, 422), (1010, 418), (0, 668), (380, 971), (1580, 984)]
    _, rv, t = cv2.solvePnP(np.array(obj, float), np.array(img, float), K, None)
    return np.r_[f0, W / 2, H_IMG / 2, 0, 0, rv.ravel(), t.ravel(), 1.8, 1.8]


if __name__ == '__main__':
    best = None
    for f0 in [1000, 1400, 1800, 2400]:
        r = least_squares(resid, init(f0), loss='soft_l1', f_scale=4)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    K, dist, rv, t = unpack(p)
    R, _ = cv2.Rodrigues(rv)
    C = -R.T @ t
    print(f'f={p[0]:.0f}px pp=({p[1]:.0f},{p[2]:.0f}) k1={p[3]:.3f} k2={p[4]:.3f}')
    print(f'camera at X={C[0]:.2f} Y={C[1]:.2f} height={-C[2]:.2f} m')
    for (pts, axis, val), name in zip(LINES, ['far', 'near', 'left', 'right']):
        e = to_ground(p, pts)[:, axis] - val
        print(f'  {name} line max err {np.abs(e).max() * 100:.0f} cm')
    g = to_ground(p, feet); g[:, 2] = -neck_heights(p)
    dy = project(p, g)[:, 1] - neck[:, 1]
    for pl in sorted(set(S.player)):
        m = (S.player == pl).values
        print(f'  {pl}: {m.sum()} poses, neck dy median {np.median(dy[m]):.1f}px')
    u = project(p, [[8.0 * SC, 8.0 * SC, -NET_H]])[0]
    print(f'  net top at near antenna: model y={u[1]:.0f}px, image y={NET_TOP_NEAR[1]}px')
    players = dict(STATURE, **{pl: round(float(p[11 + i]), 3) for i, pl in enumerate(FREE)})
    print('  statures:', players)
    json.dump({pl: {'stature_m': v, 'source': 'given' if pl in STATURE else 'measured'}
               for pl, v in players.items()}, open(os.path.join(OUTD, 'player_statures.json'), 'w'), indent=1)
    np.savez(os.path.join(OUTD, 'camera.npz'), K=K, dist=dist, R=R, t=t, C=C)
