"""EXPERIMENT (improve): like scripts/experiments/chest_marks/score.py, but the estimator uses the camera's fitted
drag (k_drag in camera.npz) and, with OCCL=1, no size term where the ball overlaps a player or is near a touch.
Also reports the marked height measured at the chest from the pose (geo_chest) next to the one above the feet.

Score a camera on the marked chest touches it did not use: re-estimate the ball with the main estimator
(no anchors) and compare the flight-fit height at each marked frame with the marked geometric height
(ball pixel at the touching player's horizontal position, same camera).
usage: python scripts/experiments/chest_marks/score.py CAMERA OUT_DIR [even|odd|all]   (the parity to score)"""
import os, sys, json, glob, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
cam_path, out, parity = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'all')
os.makedirs(out, exist_ok=True)
src = open('scripts/ball_height/estimate.py').read()
src = src.replace("cam = Camera()", f"cam = Camera({cam_path!r})").replace("'data/analysis/ball_height_v2.csv'", f"'{out}/ball_height_v2.csv'")
src = src.replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))", "sys.path.insert(0, 'scripts/ball_height')")
src = src.replace("ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))", f"ROOT = {ROOT!r}")
kd = float(np.load(cam_path).get('k_drag', 0.0)) if 'k_drag' in np.load(cam_path).files else 0.0
src = src.replace("P = simulate(x[:3], x[3:6], t)", f"P = simulate(x[:3], x[3:6], t, k={kd})")
src = src.replace("X[fr] = simulate(x[:3], x[3:6], T_all[fr] - T_all[o[0]])", f"X[fr] = simulate(x[:3], x[3:6], T_all[fr] - T_all[o[0]], k={kd})")
if os.environ.get('OCCL') == '1':
    src = src.replace("rs = (dist - dist_size) / (SIZE_SIGMA * dist_size)", "rs = (dist - dist_size) / (SIZE_SIGMA * dist_size) * SIZE_OK[idx]")
    src = src.replace("def size_point(i):", '''pbx_ = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
m_ = pbx_.merge(b[['frame', 'x1', 'y1', 'x2', 'y2']], on='frame', suffixes=('', '_b'))
SIZE_OK = np.ones(len(b))
SIZE_OK[m_.frame[(m_.x1 < m_.x2_b) & (m_.x2 > m_.x1_b) & (m_.y1 < m_.y2_b) & (m_.y2 > m_.y1_b)].unique()] = 0.0
fm_ = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm_ = fm_[~fm_.deleted].set_index('v3_frame').v2_frame
for f_ in pd.read_csv('data/contacts/touch_labels_final.csv').frame:
    if f_ in fm_.index:
        v_ = int(fm_[f_]); SIZE_OK[max(0, v_ - 2): v_ + 3] = 0.0


def size_point(i):''')
assert src.count(f'k={kd}') == 2, 'simulate calls not patched'
open(f'{out}/_est.py', 'w').write(src)
subprocess.run([sys.executable, '-W', 'ignore', f'{out}/_est.py'], check=True, capture_output=True)
os.remove(f'{out}/_est.py')
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera(cam_path)
h = pd.read_csv(f'{out}/ball_height_v2.csv').set_index('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
Hi = np.linalg.inv(np.load('data/mappings/homography_px_to_m.npy'))
pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player', 'neck_x', 'neck_y', 'mid_hip_x', 'mid_hip_y']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
STAT = {k: v['stature_m'] for k, v in json.load(open('data/experiments/physics_calib/player_statures.json')).items()}
rows = []
for fn in glob.glob('data/experiments/chest_marks/marks/marks/*.json'):
    m = json.load(open(fn))
    if not (m.get('chest') and m.get('feet') == 'both'):
        continue
    if parity != 'all' and (int(m['touch']) % 2 == 0) != (parity == 'even'):
        continue
    fr = int(m['frame'])
    try:
        x, y = pos.loc[(fr, m['player'])].values.ravel()[:2]
    except KeyError:
        continue
    q = Hi @ np.array([x, y, 1.0]); g = cam.ground(np.array([q[:2] / q[2]]))[0]
    d = cam.ray(np.array([[m['ball_x'], m['ball_y']]]))[0]
    s = np.dot(g[:2] - cam.C[:2], d[:2]) / np.dot(d[:2], d[:2]); geo = -(cam.C + s * d)[2]
    geo_c = np.nan
    try:
        kp = pose.loc[(fr, m['player'])]
        st = STAT[m['player']]
        Cc = cam.C
        def at_h(px, z):
            dd = cam.ray(np.array([px]))[0]; s_ = (-z - Cc[2]) / dd[2]; return Cc + s_ * dd
        A = at_h([kp.neck_x, kp.neck_y], 0.82 * st); B = at_h([kp.mid_hip_x, kp.mid_hip_y], 0.53 * st)
        ch = B + (0.72 - 0.53) / (0.82 - 0.53) * (A - B)
        s2 = np.dot(ch[:2] - cam.C[:2], d[:2]) / np.dot(d[:2], d[:2]); geo_c = -(cam.C + s2 * d)[2]
    except KeyError:
        pass
    if h.method.get(fr) == 'flight_fit':
        P = h.loc[fr]
        rows.append(dict(touch=m['touch'], geo_h=geo, geo_chest=geo_c, fit_h=P.height_m, dxy=float(np.hypot(P.court_x_m - g[0], P.court_y_m - g[1]))))
d = pd.DataFrame(rows); d['err'] = d.fit_h - d.geo_h; d['err_chest'] = d.fit_h - d.geo_chest
d.to_csv(f'{out}/score.csv', index=False)
fit = h[h.method == 'flight_fit'].height_m
print(f'n={len(d)} ({parity}): fit - marked height median {d.err.median():+.2f} m, MAE {d.err.abs().median():.2f} m '
      f'(vs chest point: {d.err_chest.median():+.2f} / {d.err_chest.abs().median():.2f}); '
      f'ball-player distance {d.dxy.median():.2f} m; marked height {d.geo_h.median():.2f} m; '
      f'ball in flight median {fit.median():.2f} / p95 {fit.quantile(.95):.2f} m')
