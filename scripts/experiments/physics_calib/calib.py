"""EXPERIMENT: camera calibration from physics only (no player heights).

One joint least-squares fit of the camera (f, principal point, k1, k2, R, t) and of many ball flights at once.
Known quantities only:
  * the blue lines: 16 x 8 m (court lines, as given),
  * the net top: 2.20 m along its whole length (the white top tape, seen edge-on as a near-vertical line at
    x ~ 1011 px, X = 8) and where the near antenna meets it above the near sideline (X = 8, Y = 8),
  * gravity 9.81 m/s^2 in every flight,
  * ball diameter 0.22 m (box size -> distance),
  * sand landings: the ball centre is 0.11 m above the sand (data/contacts/ball_contacts_v3.csv, ball_on_sand).
Player heights are NOT used; they come out as a check (check.py), as do the labelled touches.
Outputs data/experiments/physics_calib/camera.npz.
"""
import os, sys
import numpy as np, pandas as pd, cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
OUT = 'data/experiments/physics_calib'
W, H_IMG, FPS, G, BALL_D = 1920, 1080, 25.0, 9.81, 0.22
PX, SIZE_SIG, PP_SIG = 1.5, 0.12, 60.0
N_FLIGHTS = int(os.environ.get('N_FLIGHTS', 200))

FAR = [(260, 419), (410, 419), (560, 418), (710, 419), (1010, 418), (1160, 421), (1310, 421), (1460, 420), (1610, 422)]
NEAR = [(260, 967), (380, 971), (500, 976), (620, 979), (740, 982), (860, 984), (980, 987), (1100, 986),
        (1220, 986), (1340, 987), (1460, 986), (1580, 984), (1700, 983)]
LEFT = [(203, 455), (145, 515), (114, 545), (89, 575), (62, 605), (32, 635)]
RIGHT = [(1825, 450), (1874, 500), (1902, 525)]
LINES = [(FAR, 1, 0.0), (NEAR, 1, 8.0), (LEFT, 0, 0.0), (RIGHT, 0, 16.0)]
CORNERS = [((232, 420), (0.0, 0.0)), ((1790, 422), (16.0, 0.0))]
NET_TAPE = [(1010, 400), (1011, 450), (1011, 500), (1012, 550), (1012, 600)]  # median frame, X = 8 at 2.20 m
NET_END = (1016, 620)   # antenna foot on the tape, above the near sideline: (8, 8, 2.20)
NET_H = 2.20
SAND_H = BALL_D / 2


# --- camera (world: X along the court, Y toward the camera, Z down) ---
def cam_parts(c):
    f, cx, cy, k1, k2 = c[:5]
    R, _ = cv2.Rodrigues(c[5:8])
    return f, cx, cy, k1, k2, R, c[8:11]


def project(c, X):
    f, cx, cy, k1, k2, R, t = cam_parts(c)
    Xc = X @ R.T + t
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return np.c_[f * x * d + cx, f * y * d + cy]


def rays(c, uv):
    f, cx, cy, k1, k2, R, t = cam_parts(c)
    xd, yd = (uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(8):  # invert the radial distortion
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        x, y = xd / d, yd / d
    v = np.c_[x, y, np.ones_like(x)] @ R  # R.T @ v for each row
    C = -R.T @ t
    return C, v / np.linalg.norm(v, axis=1, keepdims=True)


def on_plane(c, uv, z):
    C, v = rays(c, uv)
    s = (z - C[2]) / v[:, 2]
    return C + s[:, None] * v, s


# --- data ---
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
good = b.source.isin(['detected', 'rescued', 'manual']).values
uv_all = b[['center_x', 'center_y']].values
size_all = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)).values
T_all = b.time_sec.values
h0 = pd.read_csv('data/analysis/ball_height_v2.csv')  # flights from the main pipeline (segmentation only)
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
bc = pd.read_csv('data/contacts/ball_contacts_v3.csv', usecols=['frame', 'event_type', 'ball_on_sand'])
sand = sorted({int(fm[r.frame]) for r in bc.itertuples() if r.event_type != 'touch' and r.ball_on_sand is True
               and r.frame in fm.index})
sand = [f for f in sand if good[f]]

# long clean flights with an apex (they carry the gravity information)
flights = []
for k, g in h0[h0.flight_id >= 0].groupby('flight_id'):
    idx = g.frame.values[good[g.frame.values]]
    if len(idx) < 12 or np.nanpercentile(g.reproj_err_px, 90) > 2.5:
        continue
    vy = np.diff(uv_all[idx, 1])
    if np.ptp(vy) < 6:  # needs visible curvature to carry gravity
        continue
    flights.append(idx)
flights = sorted(flights, key=len, reverse=True)[:N_FLIGHTS]
print(f'{len(flights)} flights ({sum(map(len, flights))} detections), {len(sand)} sand landings')

c0 = np.load('data/analysis/camera_v2.npz')
rv0 = cv2.Rodrigues(c0['R'])[0].ravel()
cam0 = np.r_[c0['K'][0, 0], c0['K'][0, 2], c0['K'][1, 2], c0['dist'][0], c0['dist'][1], rv0, c0['t']]


def flight_init(c, idx):
    t = T_all[idx] - T_all[idx[0]]
    d = BALL_D * c[0] / size_all[idx]
    C, v = rays(c, uv_all[idx])
    P = C + d[:, None] * v
    A = np.c_[np.ones_like(t), t]
    p0, v0 = [], []
    for k in range(3):
        acc = G if k == 2 else 0.0
        y = P[:, k] - 0.5 * acc * t ** 2
        sol = np.linalg.lstsq(A, y, rcond=None)[0]
        p0.append(sol[0]); v0.append(sol[1])
    return np.r_[p0, v0]


x0 = np.r_[cam0, np.concatenate([flight_init(cam0, f) for f in flights])]
NC = 11
gvec = np.array([0, 0, G])


def residuals(x):
    c = x[:NC]
    r = []
    for pts, axis, val in LINES:
        P, _ = on_plane(c, np.array(pts, float), 0.0)
        r.append((P[:, axis] - val) * 60)
    for uv, XY in CORNERS:
        r.append(project(c, np.array([[XY[0], XY[1], 0.0]]))[0] - uv)
    P, _ = on_plane(c, np.array(NET_TAPE, float), -NET_H)
    r.append((P[:, 0] - 8.0) * 60)
    r.append((project(c, np.array([[8.0, 8.0, -NET_H]]))[0] - NET_END) * 2)
    r.append([(c[1] - W / 2) / PP_SIG, (c[2] - H_IMG / 2) / PP_SIG, c[3] / 0.3, c[4] / 0.3])  # weak priors
    if sand:
        uvs = uv_all[sand]
        _, s = on_plane(c, uvs, -SAND_H)
        dsz = BALL_D * c[0] / size_all[sand]
        r.append((s - dsz) / (SIZE_SIG * dsz) * 2)
    for k, idx in enumerate(flights):
        p = x[NC + 6 * k: NC + 6 * k + 6]
        t = T_all[idx] - T_all[idx[0]]
        P = p[:3] + np.outer(t, p[3:]) + 0.5 * np.outer(t ** 2, gvec)
        r.append(((project(c, P) - uv_all[idx]) / PX).ravel())
        dist = np.linalg.norm(P - (-cam_parts(c)[5].T @ c[8:11]), axis=1)
        dsz = BALL_D * c[0] / size_all[idx]
        r.append((dist - dsz) / (SIZE_SIG * dsz))
    return np.concatenate([np.ravel(a) for a in r])


# sparsity: camera columns touch everything; each flight's columns only its own residuals
r0 = residuals(x0)
n_head = len(r0) - sum(3 * len(f) for f in flights)
S = lil_matrix((len(r0), len(x0)), dtype=int)
S[:, :NC] = 1
row = n_head
for k, idx in enumerate(flights):
    S[row: row + 3 * len(idx), NC + 6 * k: NC + 6 * k + 6] = 1
    row += 3 * len(idx)
print('residuals', len(r0), 'params', len(x0))
sol = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1', f_scale=3.0, x_scale='jac', max_nfev=60, verbose=1)
c = sol.x[:NC]
f, cx, cy, k1, k2, R, t = cam_parts(c)
C = -R.T @ t
print(f'f={f:.0f} pp=({cx:.0f},{cy:.0f}) k1={k1:.3f} k2={k2:.3f} camera X={C[0]:.2f} Y={C[1]:.2f} height={-C[2]:.2f} m')
for (pts, axis, val), nm in zip(LINES, ['far', 'near', 'left', 'right']):
    P, _ = on_plane(c, np.array(pts, float), 0.0)
    print(f'  {nm} line max err {np.abs(P[:, axis] - val).max() * 100:.0f} cm')
P, _ = on_plane(c, np.array(NET_TAPE, float), -NET_H)
print(f'  net tape X err (cm): {np.round((P[:, 0] - 8) * 100).astype(int).tolist()}, net end '
      f'{np.round(project(c, np.array([[8.0, 8.0, -NET_H]]))[0] - NET_END, 1).tolist()} px')
_, s = on_plane(c, uv_all[sand], -SAND_H)
print(f'  sand landings: plane distance / size distance median {np.median(s / (BALL_D * f / size_all[sand])):.3f}')
fr = []
for k, idx in enumerate(flights):
    p = sol.x[NC + 6 * k: NC + 6 * k + 6]
    tt = T_all[idx] - T_all[idx[0]]
    P = p[:3] + np.outer(tt, p[3:]) + 0.5 * np.outer(tt ** 2, gvec)
    fr.append(np.linalg.norm(project(c, P) - uv_all[idx], axis=1))
print(f'  flight reprojection median {np.median(np.concatenate(fr)):.2f} px')
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
np.savez(f'{OUT}/camera.npz', K=K, dist=np.array([k1, k2, 0, 0, 0]), R=R, t=t, C=C)
