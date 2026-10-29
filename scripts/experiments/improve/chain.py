"""EXPERIMENT (improve): fit all ball flights together with continuity.

The ball cannot jump: where one flight ends and the next starts (a touch, a bounce, within MAX_GAP frames) both
must be at the same point (sigma CONT_SIG). Every flight keeps its own ballistic path (gravity 9.81; with DRAG=1 also air drag dv/dt = g - k|v|v, one k for
the whole match fitted with the flights, prior 0.02 +- 0.02 1/m, all flights integrated together, RK4) and
its pixel and ball-size terms. Optionally the hand-marked sand landings pin z = 0.11 m (SAND_USE=even|odd|all|none)
and OCCL=1 drops the size term where the ball overlaps a player or is near a touch.
NET=even|odd|all: the ball must pass over the net: where a flight crosses the net plane (X = 8) its centre must be
at least 2.20 + 0.11 m up (one-sided penalty, crossings picked by parity of their frame); plus the two labelled
net touches at X = 8, 2.0 +- 0.25 m. CHEST_MARKS=even|odd|all with POSE=vitpose|sapiens: at a hand-marked chest
touch the ball is horizontally at the chest point (between hip and neck midpoints, pose at the marked frame).
G_FIT=1: one gravity value for the whole match is fitted instead of 9.81 (prior 9.81 +- G_SIG, default 10 = free);
the ball3d benchmark (CVPRW 2026) found this the best monocular model, because it can absorb depth/scale errors.
ACC=1: every flight gets its own constant extra acceleration (3 values, prior 0 +- ACC_SIG m/s^2, default 1.5):
a spin (Magnus lift / sidespin) term, constant over the flight.
SHADOW=<shadows csv>: the ball's shadow on the sand (scripts/experiments/shadow/) is a second view: the ground point
P_xy + height * sun_xy / tan(elevation) must project onto the detected shadow (sigma SHADOW_SIG px, default 3).
The sun comes from SUN=<sun.json>. SIZE_SRC=circle: ball size from the circle fit (data/experiments/ball_size/
circle.csv) instead of the detector box, where available; SIZE_SRC=silhouette: from the background-subtracted silhouette (silhouette.csv, clean frames).
CAM_FIT=1: the camera (focal, principal point, k1, k2, rotation, position) is refined together with the flights,
keeping the calibration's own landmark terms (court lines, net tape, antenna stripes); SUN_FIT=1: the sun direction
(azimuth, elevation, drift) is refined too. Writes OUT/camera_fit.npz and OUT/sun_fit.json.
Starts from the flights of OUT/nodrag.csv (estimate_drag.py output with the same camera).
Writes OUT/chain.csv. usage: python scripts/experiments/improve/chain.py CAMERA OUT_DIR
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

cam_path, out = sys.argv[1], sys.argv[2]
SAND_USE = os.environ.get('SAND_USE', 'none')
OCCL = os.environ.get('OCCL') == '1'
CONT_SIG = float(os.environ.get('CONT_SIG', 0.15))
DRAG = os.environ.get('DRAG') == '1'
G_FIT = os.environ.get('G_FIT') == '1'
G_SIG = float(os.environ.get('G_SIG', 10.0))
ACC = os.environ.get('ACC') == '1'
ACC_SIG = float(os.environ.get('ACC_SIG', 1.5))
PER = 9 if ACC else 6
CAM_FIT = os.environ.get('CAM_FIT') == '1'
SUN_FIT = os.environ.get('SUN_FIT') == '1'
sys.path.insert(0, HERE)
import landmarks as LM
K0, SUB = 0.02, 8
MAX_GAP = 2
G, FPS, BALL_D, PX, SIZE_SIG = 9.81, 25.0, 0.22, 1.5, 0.12
gvec = np.array([0, 0, G])
cam = Camera(cam_path)
fpx = cam.K[0, 0]
h0 = pd.read_csv(f'{out}/nodrag.csv')
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
good = b.source.isin(['detected', 'rescued', 'manual']).values
uv = b[['center_x', 'center_y']].values
size = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)).to_numpy().copy()
if os.environ.get('SIZE_SRC') == 'circle':
    cc_ = pd.read_csv('data/experiments/ball_size/circle.csv')
    cc_ = cc_[(cc_.diam_px > 0.5 * cc_.box_px) & (cc_.diam_px < 1.6 * cc_.box_px)]
    size[cc_.frame.values] = cc_.diam_px.values
    print(f'ball size from the circle fit in {len(cc_)} frames')
if os.environ.get('SIZE_SRC') == 'silhouette':
    sl_ = pd.read_csv('data/experiments/ball_size/silhouette.csv'); sl_ = sl_[sl_.ok]
    size[sl_.frame.values] = sl_.diam_px.values
    print(f'ball size from the silhouette in {len(sl_)} frames')
# SIZE_FIX=even|odd|all: the detector's boxes are squeezed towards ~25 px; map box size -> ball size with a line
# fitted on the hand-marked sand landings of that parity, where the ball's distance is known (ray at z = 0.11 m)
SIZE_FIX = os.environ.get('SIZE_FIX', 'none')
if SIZE_FIX != 'none':
    pairs = []
    for fn in glob.glob('data/experiments/sand_marks/marks/landings/*.json'):
        m = json.load(open(fn))
        if m.get('landing') is not True:
            continue
        f_ = int(m['frame'])
        if np.isnan(size[f_]) or (SIZE_FIX != 'all' and (f_ % 2 == 0) != (SIZE_FIX == 'even')):
            continue
        d_ = cam.ray(uv[[f_]])[0]; s_ = (-0.11 - cam.C[2]) / d_[2]
        pairs.append((size[f_], cam.K[0, 0] * BALL_D / s_))
    pairs = np.array(pairs)
    a_, b_ = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
    print(f'size fix from {len(pairs)} landings: ball px = {b_:.2f} + {a_:.3f} x box px')
    size = b_ + a_ * size
size_ok = np.ones(len(b))
if OCCL:
    pbx = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
    m_ = pbx.merge(b[['frame', 'x1', 'y1', 'x2', 'y2']], on='frame', suffixes=('', '_b'))
    size_ok[m_.frame[(m_.x1 < m_.x2_b) & (m_.x2 > m_.x1_b) & (m_.y1 < m_.y2_b) & (m_.y2 > m_.y1_b)].unique()] = 0

# flights: frames, detections, start params from the no-drag positions
FL = []
X3 = np.c_[h0.court_x_m, h0.court_y_m, -h0.height_m].astype(float)
for fid, g in h0[h0.flight_id >= 0].groupby('flight_id'):
    fr = g.frame.values
    obs = fr[good[fr]]
    if len(obs) < 3:
        continue
    t = (fr - fr[0]) / FPS
    A = np.c_[np.ones_like(t), t]
    p = [np.linalg.lstsq(A, X3[fr, k] - (0.5 * G * t ** 2 if k == 2 else 0), rcond=None)[0] for k in range(3)]
    FL.append(dict(f0=int(fr[0]), f1=int(fr[-1]), obs=obs, x0=np.r_[[q[0] for q in p], [q[1] for q in p], [0.0] * (PER - 6)]))
FL.sort(key=lambda d: d['f0'])
links = [(i, i + 1) for i in range(len(FL) - 1) if 0 <= FL[i + 1]['f0'] - FL[i]['f1'] <= MAX_GAP]
sand = []
for fn in glob.glob('data/experiments/sand_marks/marks/landings/*.json'):
    m = json.load(open(fn))
    if SAND_USE != 'none' and m.get('landing') is True and (SAND_USE == 'all' or (m['frame'] % 2 == 0) == (SAND_USE == 'even')):
        f = int(m['frame'])
        for i, d in enumerate(FL):
            if d['f0'] - 2 <= f <= d['f1'] + 2:
                sand.append((i, f))
SHADOW = os.environ.get('SHADOW')
SHADOW_SIG = float(os.environ.get('SHADOW_SIG', 3.0))
shadows = []  # (flight i, frame, u, v)
if SHADOW:
    SUNP = json.load(open(os.environ['SUN']))
    sd_ = pd.read_csv(SHADOW)
    if 'err_px' in sd_:
        sd_ = sd_[sd_.err_px < float(os.environ.get('SHADOW_MAXERR', 25))]
    for r_ in sd_.itertuples():
        for i, d in enumerate(FL):
            if d['f0'] <= r_.frame <= d['f1']:
                shadows.append((i, int(r_.frame), r_.u, r_.v)); break


def sun_xy(f):
    a = np.radians(SUNP['az'] + SUNP['drift_deg_per_1000'] * f / 1000.0)
    return np.array([np.cos(a), np.sin(a)]) / np.tan(np.radians(SUNP['elev']))


NET = os.environ.get('NET', 'none')
CHEST_MARKS = os.environ.get('CHEST_MARKS', 'none')
POSE = os.environ.get('POSE', 'vitpose')
par = lambda f, use: use == 'all' or (f % 2 == 0) == (use == 'even')
cross = []  # (flight i, frame) where the start solution crosses X = 8 inside the flight
net_touch = []
if NET != 'none':
    fm_ = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm_ = fm_[~fm_.deleted].set_index('v3_frame').v2_frame
    tl_ = pd.read_csv('data/contacts/touch_labels_final.csv')
    nt_frames = [int(fm_[f]) for f in tl_[tl_.kind == 'net'].frame if f in fm_.index]
    for i, d in enumerate(FL):
        fr = np.arange(d['f0'], d['f1'] + 1)
        xs = d['x0'][0] + d['x0'][3] * (fr - d['f0']) / FPS - 8.0  # X is not affected by gravity
        for k_ in np.flatnonzero(np.sign(xs[:-1]) != np.sign(xs[1:])):
            f = int(fr[k_] if abs(xs[k_]) < abs(xs[k_ + 1]) else fr[k_ + 1])
            if par(f, NET) and all(abs(f - g) > 6 for g in nt_frames):
                cross.append((i, f))
        for g in nt_frames:
            if d['f0'] - 2 <= g <= d['f1'] + 2 and NET == 'all':
                net_touch.append((i, g))
chest = []  # (flight i, frame, neck px, hip px, stature)
if CHEST_MARKS != 'none':
    STAT = {k_: v['stature_m'] for k_, v in json.load(open('data/experiments/physics_calib/player_statures.json')).items()}
    marks = [json.load(open(fn)) for fn in glob.glob('data/experiments/chest_marks/marks/marks/*.json')]
    marks = [m for m in marks if m.get('chest') and m.get('feet') == 'both' and par(int(m['touch']), CHEST_MARKS)]
    if POSE == 'sapiens':
        cols = ['v2_frame', 'player'] + [f'{a}_{c}' for a in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip') for c in 'xys']
        sp = pd.read_csv('data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv', usecols=cols).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
    else:
        vp = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player', 'neck_x', 'neck_y', 'neck_s', 'mid_hip_x', 'mid_hip_y', 'mid_hip_s']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
    for m in marks:
        f = int(m['frame'])
        try:
            if POSE == 'sapiens':
                k_ = sp.loc[(f, m['player'])]
                if min(k_.left_shoulder_s, k_.right_shoulder_s, k_.left_hip_s, k_.right_hip_s) < 0.3:
                    continue
                nk = np.array([(k_.left_shoulder_x + k_.right_shoulder_x) / 2, (k_.left_shoulder_y + k_.right_shoulder_y) / 2])
                hp = np.array([(k_.left_hip_x + k_.right_hip_x) / 2, (k_.left_hip_y + k_.right_hip_y) / 2])
            else:
                k_ = vp.loc[(f, m['player'])]
                if min(k_.neck_s, k_.mid_hip_s) < 0.4:
                    continue
                nk, hp = np.array([k_.neck_x, k_.neck_y]), np.array([k_.mid_hip_x, k_.mid_hip_y])
        except KeyError:
            continue
        for i, d in enumerate(FL):
            if d['f0'] - 2 <= f <= d['f1'] + 2:
                chest.append((i, f, nk, hp, STAT[m['player']])); break


def at_height(px, z):
    d_ = cam.ray(np.array([px]))[0]
    return cam.C + ((-z - cam.C[2]) / d_[2]) * d_


CHEST_XY = []
for i, f, nk, hp, st in chest:
    A_, B_ = at_height(nk, 0.82 * st), at_height(hp, 0.53 * st)
    CHEST_XY.append((B_ + (0.72 - 0.53) / (0.82 - 0.53) * (A_ - B_))[:2])
print(f'{len(FL)} flights, {len(links)} continuity links, {len(sand)} sand pins, {len(cross)} net crossings, '
      f'{len(net_touch)} net touches, {len(chest)} chest points ({POSE}), {len(shadows)} shadows')



GCUR = [G]  # current gravity (G_FIT changes it inside the fit)


def accel(p):
    a = np.array([0.0, 0.0, GCUR[0]])
    return a + p[6:9] if ACC else a


def pos(p, f0, frames, k=0.0):
    t = (np.asarray(frames, float) - f0) / FPS
    if k == 0.0:
        return p[:3] + np.outer(t, p[3:6]) + 0.5 * np.outer(t ** 2, accel(p))
    return traj_all(p[None, :], np.array([max(1, int(np.max(np.asarray(frames) - f0)) + 1)]), k)[0][(np.asarray(frames) - f0).astype(int)]


def traj_all(P6, lens, k):
    """positions of many flights at whole frames 0..len-1 from their start, integrated together (RK4)."""
    n, L = len(P6), int(lens.max())
    P, V = P6[:, :3].copy(), P6[:, 3:6].copy()
    A0 = np.array([0.0, 0.0, GCUR[0]]) + (P6[:, 6:9] if ACC else 0.0)
    out_ = np.empty((n, L, 3)); out_[:, 0] = P
    h = 1.0 / FPS / SUB
    def acc(v):
        return A0 - k * np.linalg.norm(v, axis=1, keepdims=True) * v
    for i in range(1, L):
        for _ in range(SUB):
            a1 = acc(V); a2 = acc(V + 0.5 * h * a1); a3 = acc(V + 0.5 * h * a2); a4 = acc(V + h * a3)
            P = P + h * (V + h / 6 * (a1 + a2 + a3))
            V = V + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        out_[:, i] = P
    return out_


LENS = np.array([d['f1'] - d['f0'] + 3 for d in FL])  # +2 frames: continuity / sand points just past the end
NP = PER * len(FL)
IG = NP + (1 if DRAG else 0)  # index of the gravity parameter (G_FIT)
IC = IG + (1 if G_FIT else 0)  # camera block (CAM_FIT, 11 values)
IS = IC + (11 if CAM_FIT else 0)  # sun block (SUN_FIT: azimuth, elevation, drift)
C0 = LM.from_camera(cam)
CUR = dict(proj=cam.project, C=cam.C, f=fpx)


def all_pos(x):
    k = x[NP] if DRAG else 0.0
    if not DRAG:
        return None, 0.0
    return traj_all(x[:NP].reshape(-1, PER), LENS, k), k


def P_at(x, i, frames, T, k):
    if T is None:
        return pos(x[PER * i: PER * i + PER], FL[i]['f0'], frames)
    rel = np.asarray(frames) - FL[i]['f0']
    if rel.min() < 0:  # a point before the start: integrate backwards is not needed in practice; extrapolate
        return pos(x[PER * i: PER * i + PER], FL[i]['f0'], frames)
    return T[i][rel]


def residuals(x):
    r = []
    if G_FIT:
        GCUR[0] = x[IG]
    if CAM_FIT:
        c_ = x[IC:IC + 11]
        CUR.update(proj=lambda X, c_=c_: LM.project(c_, X), C=LM.center(c_), f=c_[0])
    if SUN_FIT:
        SUNP.update(az=x[IS], elev=x[IS + 1], drift_deg_per_1000=x[IS + 2])
    T, k = all_pos(x)
    if DRAG:
        r.append([(k - K0) / 0.02])
    for i, d in enumerate(FL):
        P = P_at(x, i, d['obs'], T, k)
        r.append(((CUR['proj'](P) - uv[d['obs']]) / PX).ravel())
        dsz = BALL_D * CUR['f'] / size[d['obs']]
        r.append((np.linalg.norm(P - CUR['C'], axis=1) - dsz) / (SIZE_SIG * dsz) * size_ok[d['obs']])
    for i, j in links:
        tb = FL[j]['f0'] if FL[j]['f0'] > FL[i]['f1'] else FL[i]['f1']
        r.append((P_at(x, i, [tb], T, k)[0] - P_at(x, j, [tb], T, k)[0]) / CONT_SIG)
    for i, f in sand:
        r.append([(-P_at(x, i, [f], T, k)[0][2] - BALL_D / 2) / 0.05])
    for i, f in cross:
        r.append([max(0.0, 2.20 + BALL_D / 2 + P_at(x, i, [f], T, k)[0][2]) / 0.05])
    for i, f in net_touch:
        P = P_at(x, i, [f], T, k)[0]
        r.append([(P[0] - 8.0) / 0.2, (-P[2] - 2.0) / 0.25])
    for (i, f, nk, hp, st), xy in zip(chest, CHEST_XY):
        r.append((P_at(x, i, [f], T, k)[0][:2] - xy) / 0.15)
    for i, f, u_, v_ in shadows:
        P = P_at(x, i, [f], T, k)[0]
        G_ = np.r_[P[:2] + (-P[2]) * sun_xy(f), 0.0]
        r.append((CUR['proj'](G_[None, :])[0] - [u_, v_]) / SHADOW_SIG)
    if ACC:
        r.append(x[:NP].reshape(-1, PER)[:, 6:9].ravel() / ACC_SIG)
    if G_FIT:
        r.append([(x[IG] - G) / G_SIG])
    if CAM_FIT:
        r.append(LM.residuals(x[IC:IC + 11]))
    return np.concatenate([np.ravel(a) for a in r])


x0 = np.concatenate([d['x0'] for d in FL] + ([[K0]] if DRAG else []) + ([[G]] if G_FIT else []) + ([C0] if CAM_FIT else [])
                    + ([[SUNP['az'], SUNP['elev'], SUNP['drift_deg_per_1000']]] if SUN_FIT else []))
r0 = residuals(x0)
S = lil_matrix((len(r0), len(x0)), dtype=int)
row = 0
if DRAG:
    S[:, NP] = 1  # the drag constant touches every residual
    S[0, NP] = 1; row = 1
for i, d in enumerate(FL):
    n = 3 * len(d['obs']); S[row: row + n, PER * i: PER * i + PER] = 1; row += n
for i, j in links:
    S[row: row + 3, PER * i: PER * i + PER] = 1; S[row: row + 3, PER * j: PER * j + PER] = 1; row += 3
for i, f in sand:
    S[row, PER * i: PER * i + PER] = 1; row += 1
for i, f in cross:
    S[row, PER * i: PER * i + PER] = 1; row += 1
for i, f in net_touch:
    S[row: row + 2, PER * i: PER * i + PER] = 1; row += 2
for c_ in chest:
    S[row: row + 2, PER * c_[0]: PER * c_[0] + PER] = 1; row += 2
for i, f, u_, v_ in shadows:
    S[row: row + 2, PER * i: PER * i + PER] = 1; row += 2
if ACC:
    for i in range(len(FL)):
        for q in range(3):
            S[row, PER * i + 6 + q] = 1; row += 1
if G_FIT:
    S[:, IG] = 1  # gravity touches every residual
    row += 1
if CAM_FIT:
    S[:, IC:IC + 11] = 1  # the camera touches every image residual (and its own landmark terms)
if SUN_FIT:
    S[:, IS:IS + 3] = 1
if DRAG and G_FIT:
    raise SystemExit('DRAG and G_FIT together are not supported')
if DRAG:  # solve without drag first (k fixed at 0), then free k from that solution
    DRAG = False
    S0 = lil_matrix(S[1:, :NP])
    sol0 = least_squares(residuals, x0[:NP], jac_sparsity=S0, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=80)
    DRAG = True
    x0 = np.r_[sol0.x, 0.0]
    print('no-drag start found, cost', round(sol0.cost, 1))
sol = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=int(os.environ.get('MAX_NFEV', 80)), verbose=0)
print('status', sol.status, 'nfev', sol.nfev, 'cost', round(sol.cost, 1))
x = sol.x
T, k = all_pos(x)
print(f'drag k = {k:.4f} 1/m' + (f'; fitted gravity {x[IG]:.3f} m/s^2' if G_FIT else ''))
if ACC:
    A_ = x[:NP].reshape(-1, PER)[:, 6:9]
    print('extra acceleration per flight (m/s^2): median |a| %.2f, p90 %.2f; median vertical %+.2f' % (np.median(np.linalg.norm(A_, axis=1)), np.percentile(np.linalg.norm(A_, axis=1), 90), np.median(-A_[:, 2])))
h = h0.copy()
gaps = []
for i, d in enumerate(FL):
    fr = np.arange(d['f0'], d['f1'] + 1)
    P = P_at(x, i, fr, T, k)
    h.loc[fr, 'court_x_m'] = P[:, 0].round(3); h.loc[fr, 'court_y_m'] = P[:, 1].round(3)
    h.loc[fr, 'height_m'] = np.clip(-P[:, 2], 0, None).round(3)
for i, j in links:
    tb = FL[j]['f0'] if FL[j]['f0'] > FL[i]['f1'] else FL[i]['f1']
    gaps.append(np.linalg.norm(P_at(x, i, [tb], T, k)[0] - P_at(x, j, [tb], T, k)[0]))
g0 = [np.linalg.norm(pos(x0[PER * i: PER * i + PER], FL[i]['f0'], [FL[j]['f0']])[0] - pos(x0[PER * j: PER * j + PER], FL[j]['f0'], [FL[j]['f0']])[0]) for i, j in links]
# per-frame height uncertainty (UNC=1): each flight's 6 start values get a covariance from the Gauss-Newton
# Hessian of the cost at the solution (its own block of the Jacobian, couplings ignored); frames outside fitted
# flights get the ball-size error (SIZE_SIG of the distance) along the viewing ray. Unscaled: calibrate the scale
# on held-out marks (scripts/experiments/shadow/unc_check.py).
if os.environ.get('UNC') == '1':
    Jc = sol.jac.tocsc()
    sig = np.full(len(h), np.nan)
    for i, d in enumerate(FL):
        B = Jc[:, PER * i: PER * i + 6]
        rows_ = np.unique(B.nonzero()[0])
        Bd = B[rows_].toarray()
        try:
            Cov = np.linalg.inv(Bd.T @ Bd + 1e-9 * np.eye(6))
        except np.linalg.LinAlgError:
            continue
        fr = np.arange(d['f0'], d['f1'] + 1); t = (fr - d['f0']) / FPS
        gz = np.zeros((len(fr), 6)); gz[:, 2] = -1; gz[:, 5] = -t
        sig[fr] = np.sqrt(np.einsum('ij,jk,ik->i', gz, Cov, gz))
    other = np.isnan(sig) & h.height_m.notna().values & b.center_x.notna().values
    idx = np.flatnonzero(other)
    if len(idx):
        dv = cam.ray(uv[idx]); dist = np.linalg.norm(np.c_[h.court_x_m.values[idx], h.court_y_m.values[idx], -h.height_m.values[idx]] - cam.C, axis=1)
        sig[idx] = SIZE_SIG * dist * np.abs(dv[:, 2])
    h['height_sigma_raw'] = np.round(sig, 4)
    print('height sigma (raw): flights median %.3f m, others median %.3f m' % (np.nanmedian(sig[~other]), np.nanmedian(sig[other]) if other.any() else np.nan))
h.to_csv(f'{out}/chain.csv', index=False)
residuals(x)  # leaves CUR / SUNP at the solution
if CAM_FIT:
    c_ = x[IC:IC + 11]; f_, cx_, cy_, k1_, k2_, R_, t_ = LM.parts(c_)
    np.savez(f'{out}/camera_fit.npz', K=np.array([[f_, 0, cx_], [0, f_, cy_], [0, 0, 1.0]]), dist=np.array([k1_, k2_, 0, 0, 0]),
             R=R_, t=t_, C=LM.center(c_), k_drag=np.array(float(k)))
    print('camera change: focal %+.1f px, principal point %+.1f/%+.1f px, k1 %+.3f, k2 %+.3f, rotation %.3f deg, position %.3f m; landmark rms %.2f (start %.2f)' % (
        c_[0] - C0[0], c_[1] - C0[1], c_[2] - C0[2], c_[3] - C0[3], c_[4] - C0[4], np.degrees(np.linalg.norm(c_[5:8] - C0[5:8])),
        np.linalg.norm(LM.center(c_) - LM.center(C0)), np.sqrt(np.mean(LM.residuals(c_) ** 2)), np.sqrt(np.mean(LM.residuals(C0) ** 2))))
if SUN_FIT:
    json.dump({k_: float(v_) for k_, v_ in SUNP.items()}, open(f'{out}/sun_fit.json', 'w'), indent=1)
    print('sun: azimuth %.2f, elevation %.2f, drift %+.3f' % (SUNP['az'], SUNP['elev'], SUNP['drift_deg_per_1000']))
pix = np.concatenate([np.linalg.norm(CUR['proj'](P_at(x, i, d['obs'], T, k)) - uv[d['obs']], axis=1) for i, d in enumerate(FL)])
json.dump(dict(drag=DRAG, k_drag=float(k), sand=SAND_USE, size_fix=SIZE_FIX, g=float(x[IG]) if G_FIT else G, acc=ACC, acc_sig=ACC_SIG if ACC else None), open(f'{out}/chain_settings.json', 'w'))
print(f'jump between consecutive flights: median {np.median(g0):.2f} -> {np.median(gaps):.2f} m; reprojection median {np.median(pix):.2f} px')
