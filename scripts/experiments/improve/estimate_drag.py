"""EXPERIMENT (improve, drag fixed): per-frame ball heights with air drag, without changing the flights.

1. The main estimator runs without drag (scripts/ball_height/estimate.py, this camera): it decides the flights.
2. Every flight is refitted with drag, dv/dt = g - k|v|v (k from the camera file or $K_DRAG), starting from its
   no-drag solution: same detections, same frames, so drag / no drag compare like for like.
   RK4 with 8 steps per frame (times are whole frames from the flight start, so every sample lands on a step).
3. OCCL=1: no size term where the ball box overlaps a player box or is within 2 frames of a touch.
Writes OUT/ball_height_v2.csv (the no-drag version stays in OUT/nodrag.csv).
usage: python scripts/experiments/improve/estimate_drag.py CAMERA OUT_DIR
"""
import os, sys, subprocess
import numpy as np, pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

cam_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
camf = np.load(cam_path)
K = float(os.environ.get('K_DRAG', camf['k_drag'] if 'k_drag' in camf.files else 0.0))
OCCL = os.environ.get('OCCL') == '1'
G, FPS, SUB, BALL_D, PX, SIZE_SIG = 9.81, 25.0, 8, 0.22, 1.5, 0.12
gvec = np.array([0, 0, G])

# 1. flights without drag
if not os.path.exists(f'{out}/nodrag.csv'):
    src = open('scripts/ball_height/estimate.py').read()
    src = src.replace("cam = Camera()", f"cam = Camera({cam_path!r})").replace("'data/analysis/ball_height_v2.csv'", f"'{out}/nodrag.csv'")
    src = src.replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))", "sys.path.insert(0, 'scripts/ball_height')")
    src = src.replace("ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))", f"ROOT = {ROOT!r}")
    open(f'{out}/_est.py', 'w').write(src)
    subprocess.run([sys.executable, '-W', 'ignore', f'{out}/_est.py'], check=True, capture_output=True)
    os.remove(f'{out}/_est.py')
h0 = pd.read_csv(f'{out}/nodrag.csv')
cam = Camera(cam_path)
f_px = cam.K[0, 0]
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
good = b.source.isin(['detected', 'rescued', 'manual']).values
uv = b[['center_x', 'center_y']].values
size = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)).values
size_ok = np.ones(len(b))
if OCCL:
    pbx = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
    m_ = pbx.merge(b[['frame', 'x1', 'y1', 'x2', 'y2']], on='frame', suffixes=('', '_b'))
    size_ok[m_.frame[(m_.x1 < m_.x2_b) & (m_.x2 > m_.x1_b) & (m_.y1 < m_.y2_b) & (m_.y2 > m_.y1_b)].unique()] = 0
    fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
    for f in pd.read_csv('data/contacts/touch_labels_final.csv').frame:
        if f in fm.index:
            v = int(fm[f]); size_ok[max(0, v - 2): v + 3] = 0


def sim(p, n_frames, k):
    """positions at frames 0..n_frames-1 after the start (RK4, SUB steps per frame)."""
    P, V = p[:3].copy(), p[3:].copy()
    h = 1.0 / FPS / SUB
    acc = lambda v: gvec - k * np.sqrt(v @ v) * v
    out_ = np.empty((n_frames, 3)); out_[0] = P
    for i in range(1, n_frames):
        for _ in range(SUB):
            a1 = acc(V); v2 = V + 0.5 * h * a1; a2 = acc(v2); v3 = V + 0.5 * h * a2; a3 = acc(v3); v4 = V + h * a3; a4 = acc(v4)
            P = P + h * (V + h / 6 * (a1 + a2 + a3))
            V = V + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        out_[i] = P
    return out_


X = h0[['court_x_m', 'court_y_m']].values.copy()
Z = -h0.height_m.values.copy()
X3 = np.c_[X, Z]
resid = np.full(len(h0), np.nan)
refit, kept = 0, 0
for fid, g in h0[h0.flight_id >= 0].groupby('flight_id'):
    fr = g.frame.values
    f0, n = fr[0], fr[-1] - fr[0] + 1
    obs = fr[good[fr]]
    if len(obs) < 4:
        continue
    rel = obs - f0
    P0 = X3[fr]
    v0 = (P0[min(2, len(P0) - 1)] - P0[0]) / (min(2, len(P0) - 1) / FPS) - 0.5 * gvec * min(2, len(P0) - 1) / FPS
    x0 = np.r_[P0[0], v0]
    dsz = BALL_D * f_px / size[obs]
    w = size_ok[obs]

    def res(x):
        P = sim(x, n, K)[rel]
        rp = (cam.project(P) - uv[obs]).ravel() / PX
        rs = (np.linalg.norm(P - cam.C, axis=1) - dsz) / (SIZE_SIG * dsz) * w
        return np.r_[rp, rs]
    # warm start: first match the no-drag positions, then the image
    r = least_squares(lambda x: (sim(x, n, K) - P0).ravel(), x0, max_nfev=50)
    r = least_squares(res, r.x, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=200)
    P = sim(r.x, n, K)
    e_new = np.linalg.norm(cam.project(P[rel]) - uv[obs], axis=1)
    e_old = np.linalg.norm(cam.project(X3[obs]) - uv[obs], axis=1)
    if np.median(e_new) <= np.median(e_old) * 1.5 + 0.5:  # keep the drag fit unless it is clearly worse
        X3[f0:f0 + n] = P; resid[obs] = e_new; refit += 1
    else:
        resid[obs] = e_old; kept += 1
h = h0.copy()
fl = h.method == 'flight_fit'
h.loc[fl, 'court_x_m'] = X3[fl, 0].round(3); h.loc[fl, 'court_y_m'] = X3[fl, 1].round(3)
h.loc[fl, 'height_m'] = np.clip(-X3[fl, 2], 0, None).round(3)
h['reproj_err_px'] = np.round(resid, 2)
h.to_csv(f'{out}/ball_height_v2.csv', index=False)
print(f'k={K:.4f}: {refit} flights refitted with drag, {kept} kept without; reprojection median {np.nanmedian(resid):.2f} px')
