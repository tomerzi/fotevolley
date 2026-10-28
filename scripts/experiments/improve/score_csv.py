"""EXPERIMENT (improve): score an existing per-frame ball height file on the marked chest touches (the same
metrics as score.py, without running an estimator).
usage: python scripts/experiments/improve/score_csv.py CAMERA BALL_HEIGHT_CSV [even|odd|all]"""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
cam_path, csv, parity = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'all')
out = os.path.dirname(csv)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera(cam_path)
h = pd.read_csv(csv).set_index('frame')
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
