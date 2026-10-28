"""EXPERIMENT (improve): the chest-mark calibration with three switches (env, 0/1):
  DRAG   quadratic air drag in every flight, dv/dt = g - k|v|v, k fitted (prior 0.02 +- 0.02 1/m),
  OCCL   no ball-size term where the ball box overlaps a player box or is within 2 frames of a touch,
  CHEST  the marked ball sits at the chest from the pose (shoulder / hip midpoints) instead of above the feet.
The camera (and k) go to $OUT/camera.npz.

Based on (chest marks): scripts/experiments/physics_calib/calib.py plus the hand-marked chest touches: in the
flight that ends or starts at a marked touch, the ball must be at the touching player's horizontal position at
that frame (sigma 0.15 m). MARK_USE=even|odd|all picks the marks used; outputs in $OUT.

Camera calibration from physics only (no player heights).

One joint least-squares fit of the camera (f, principal point, k1, k2, R, t) and of many ball flights at once.
Known quantities only:
  * the blue lines: 16 x 8 m (court lines, as given),
  * the net top: 2.20 m along its whole length (the white top tape, seen edge-on as a near-vertical line at
    x ~ 1011 px, X = 8),
  * the near antenna (rules: 1.80 m rod, top 0.80 m above the net in 10 cm red / white stripes, on the sideline):
    stripe edges seen at y 486 ... 617 px, x ~ 1018 -> (8, 8, 3.00 ... 2.30 m),
  * (check only) the near post top (rules: 2.55 m, 1 m outside the sideline): cap edge (1017, 628),
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
import json, glob
os.chdir(ROOT)
OUT = os.environ['OUT']
os.makedirs(OUT, exist_ok=True)
MARK_USE = os.environ.get('MARK_USE', 'all')
DRAG, OCCL, CHEST = (os.environ.get(k, '0') == '1' for k in ('DRAG', 'OCCL', 'CHEST'))
print('DRAG', DRAG, 'OCCL', OCCL, 'CHEST', CHEST)
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
# antenna stripe edges (frame 3000, 8x zoom): heights 3.00, 2.90, ... 2.30 m above the sand
ANTENNA = [((1018, y), hh) for y, hh in zip([486, 505, 524, 543, 561, 580, 598, 617],
                                            [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3])]
POST_TOP = ((1017, 628), (8.0, 9.0, 2.55))
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
size_ok = np.ones(len(b), bool)
if OCCL:
    pbx = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
    m_ = pbx.merge(b[['frame', 'x1', 'y1', 'x2', 'y2']], on='frame', suffixes=('', '_b'))
    ov = (m_.x1 < m_.x2_b) & (m_.x2 > m_.x1_b) & (m_.y1 < m_.y2_b) & (m_.y2 > m_.y1_b)
    size_ok[m_.frame[ov].unique()] = False
    fm_ = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm_ = fm_[~fm_.deleted].set_index('v3_frame').v2_frame
    for f in pd.read_csv('data/contacts/touch_labels_final.csv').frame:
        if f in fm_.index:
            v = int(fm_[f]); size_ok[max(0, v - 2): v + 3] = False
    print(f'size term dropped on {np.sum(~size_ok & good)} of {good.sum()} detections')
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
# marked chest touches -> (flight index, frame, player ground point in pixels)
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
Hi = np.linalg.inv(np.load('data/mappings/homography_px_to_m.npy'))
mark_anchor = []  # (flight k, frame, ground pixel, torso (neck px, hip px, stature) or None)
pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player', 'neck_x', 'neck_y', 'neck_s', 'mid_hip_x', 'mid_hip_y', 'mid_hip_s']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
STAT = {k: v['stature_m'] for k, v in json.load(open('data/experiments/physics_calib/player_statures.json')).items()}
for fn in glob.glob('data/experiments/chest_marks/marks/marks/*.json'):
    m = json.load(open(fn))
    if not (m.get('chest') and m.get('feet') == 'both' and 'ball_x' in m):
        continue
    if MARK_USE != 'all' and (int(m['touch']) % 2 == 0) != (MARK_USE == 'even'):
        continue
    fr = int(m['frame'])
    try:
        x, y = pos.loc[(fr, m['player'])].values.ravel()[:2]
    except KeyError:
        continue
    q = Hi @ np.array([x, y, 1.0]); gpx = q[:2] / q[2]
    torso = None
    if CHEST:
        try:
            kp = pose.loc[(fr, m['player'])]
            if kp.neck_s > 0.4 and kp.mid_hip_s > 0.4:
                torso = (np.array([kp.neck_x, kp.neck_y]), np.array([kp.mid_hip_x, kp.mid_hip_y]), STAT[m['player']])
        except KeyError:
            pass
    fid = h0.flight_id.values
    for side in (range(fr, fr + 4), range(fr - 1, fr - 5, -1)):  # the flight leaving the chest, else the one arriving
        cand = [f for f in side if 0 <= f < len(fid) and fid[f] >= 0]
        if cand:
            k0 = fid[cand[0]]
            idx = h0.frame.values[(h0.flight_id.values == k0)]
            idx = idx[good[idx]]
            if len(idx) >= 6:
                flights.append(idx)
                mark_anchor.append((len(flights) - 1, fr, gpx, torso))
                break
print(f'{len(mark_anchor)} marked touches used as anchors')
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


NC = 12 if DRAG else 11
K0 = 0.02  # drag prior (1/m): 0.5 rho Cd A / m for a 0.22 m, ~0.4 kg ball
x0 = np.r_[cam0, [K0] if DRAG else [], np.concatenate([flight_init(cam0, f) for f in flights])]
gvec = np.array([0, 0, G])
SUB = 4  # integration substeps per frame


def fpos(p, k, times):
    """ball positions at the given times (s, relative to the flight's first detection; may be negative)."""
    times = np.asarray(times, float)
    if k == 0.0:
        return p[:3] + np.outer(times, p[3:]) + 0.5 * np.outer(times ** 2, gvec)
    out = np.empty((len(times), 3))
    for sign in (1, -1):
        sel = np.flatnonzero(times >= 0) if sign > 0 else np.flatnonzero(times < 0)
        if not len(sel):
            continue
        order = sel[np.argsort(np.abs(times[sel]))]
        P, V, now, h = p[:3].astype(float), p[3:].astype(float), 0.0, sign / FPS / SUB
        for i in order:
            T = abs(times[i])
            while now < T - 1e-9:
                a = gvec - k * np.linalg.norm(V) * V
                V = V + h * a
                P = P + h * V
                now += abs(h)
            out[i] = P
    return out


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
    for uv, hh in ANTENNA:
        r.append((project(c, np.array([[8.0, 8.0, -hh]]))[0] - uv) * 1.0)
    # the post top is reported only: its cap edge in the image is not a sure 2.55 m point
    r.append([(c[1] - W / 2) / PP_SIG, (c[2] - H_IMG / 2) / PP_SIG, c[3] / 0.3, c[4] / 0.3])  # weak priors
    if sand:
        uvs = uv_all[sand]
        _, s = on_plane(c, uvs, -SAND_H)
        dsz = BALL_D * c[0] / size_all[sand]
        r.append((s - dsz) / (SIZE_SIG * dsz) * 2)
    kd = x[11] if DRAG else 0.0
    if DRAG:
        r.append([(kd - K0) / 0.02])
    for k, fr, gpx, torso in mark_anchor:
        p = x[NC + 6 * k: NC + 6 * k + 6]
        idx = flights[k]
        P = fpos(p, kd, [T_all[fr] - T_all[idx[0]]])[0]
        if torso is not None:
            nk, hp, st = torso
            a_, _ = on_plane(c, nk[None, :], -0.82 * st)
            b_, _ = on_plane(c, hp[None, :], -0.53 * st)
            g = b_ + (0.72 - 0.53) / (0.82 - 0.53) * (a_ - b_)  # chest point between hip and neck
        else:
            g, _ = on_plane(c, gpx[None, :], 0.0)
        r.append((P[:2] - g[0, :2]) / 0.15)
    for k, idx in enumerate(flights):
        p = x[NC + 6 * k: NC + 6 * k + 6]
        t = T_all[idx] - T_all[idx[0]]
        P = fpos(p, kd, t)
        r.append(((project(c, P) - uv_all[idx]) / PX).ravel())
        dist = np.linalg.norm(P - (-cam_parts(c)[5].T @ c[8:11]), axis=1)
        dsz = BALL_D * c[0] / size_all[idx]
        r.append((dist - dsz) / (SIZE_SIG * dsz) * size_ok[idx])
    return np.concatenate([np.ravel(a) for a in r])


# sparsity: camera columns touch everything; each flight's columns only its own residuals
r0 = residuals(x0)
n_head = len(r0) - sum(3 * len(f) for f in flights)
S = lil_matrix((len(r0), len(x0)), dtype=int)
S[:, :NC] = 1
mrow = n_head - 2 * len(mark_anchor)
for j, (k, fr, gpx, torso) in enumerate(mark_anchor):
    S[mrow + 2 * j: mrow + 2 * j + 2, NC + 6 * k: NC + 6 * k + 6] = 1
row = n_head
for k, idx in enumerate(flights):
    S[row: row + 3 * len(idx), NC + 6 * k: NC + 6 * k + 6] = 1
    row += 3 * len(idx)
print('residuals', len(r0), 'params', len(x0))
sol = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1', f_scale=3.0, x_scale='jac', max_nfev=60, verbose=1)
c = sol.x[:NC]
kd = float(sol.x[11]) if DRAG else 0.0
print(f'drag k = {kd:.4f} 1/m')
f, cx, cy, k1, k2, R, t = cam_parts(c)
C = -R.T @ t
print(f'f={f:.0f} pp=({cx:.0f},{cy:.0f}) k1={k1:.3f} k2={k2:.3f} camera X={C[0]:.2f} Y={C[1]:.2f} height={-C[2]:.2f} m')
for (pts, axis, val), nm in zip(LINES, ['far', 'near', 'left', 'right']):
    P, _ = on_plane(c, np.array(pts, float), 0.0)
    print(f'  {nm} line max err {np.abs(P[:, axis] - val).max() * 100:.0f} cm')
P, _ = on_plane(c, np.array(NET_TAPE, float), -NET_H)
print(f'  net tape X err (cm): {np.round((P[:, 0] - 8) * 100).astype(int).tolist()}')
print('  antenna stripe y err (px):', [round(float(project(c, np.array([[8.0, 8.0, -hh]]))[0, 1] - uv[1]), 1) for uv, hh in ANTENNA])
print('  post top err (px):', np.round(project(c, np.array([[8.0, 9.0, -2.55]]))[0] - POST_TOP[0], 1).tolist())
print('  net top above the near sideline (2.20 m) projects to y =', round(float(project(c, np.array([[8.0, 8.0, -2.2]]))[0, 1]), 1))
_, s = on_plane(c, uv_all[sand], -SAND_H)
print(f'  sand landings: plane distance / size distance median {np.median(s / (BALL_D * f / size_all[sand])):.3f}')
fr = []
for k, idx in enumerate(flights):
    p = sol.x[NC + 6 * k: NC + 6 * k + 6]
    tt = T_all[idx] - T_all[idx[0]]
    P = fpos(p, kd, tt)
    fr.append(np.linalg.norm(project(c, P) - uv_all[idx], axis=1))
print(f'  flight reprojection median {np.median(np.concatenate(fr)):.2f} px')
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
np.savez(f'{OUT}/camera.npz', K=K, dist=np.array([k1, k2, 0, 0, 0]), R=R, t=t, C=C, k_drag=kd)
json.dump(dict(drag=DRAG, occl=OCCL, chest=CHEST, k_drag=kd, marks=MARK_USE), open(f'{OUT}/settings.json', 'w'))
