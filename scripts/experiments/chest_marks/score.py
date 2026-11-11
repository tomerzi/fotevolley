"""Score a camera on the marked chest touches it did not use: re-estimate the ball with the main estimator
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
open(f'{out}/_est.py', 'w').write(src)
subprocess.run([sys.executable, '-W', 'ignore', f'{out}/_est.py'], check=True, capture_output=True)
os.remove(f'{out}/_est.py')
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera(cam_path)
h = pd.read_csv(f'{out}/ball_height_v2.csv').set_index('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
Hi = np.linalg.inv(np.load('data/mappings/homography_px_to_m.npy'))
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
    if h.method.get(fr) == 'flight_fit':
        P = h.loc[fr]
        rows.append(dict(touch=m['touch'], geo_h=geo, fit_h=P.height_m, dxy=float(np.hypot(P.court_x_m - g[0], P.court_y_m - g[1]))))
d = pd.DataFrame(rows); d['err'] = d.fit_h - d.geo_h
d.to_csv(f'{out}/score.csv', index=False)
fit = h[h.method == 'flight_fit'].height_m
print(f'n={len(d)} ({parity}): fit - marked height median {d.err.median():+.2f} m, MAE {d.err.abs().median():.2f} m; '
      f'ball-player distance {d.dxy.median():.2f} m; marked height {d.geo_h.median():.2f} m; '
      f'ball in flight median {fit.median():.2f} / p95 {fit.quantile(.95):.2f} m')
