"""EXPERIMENT check for physics_calib: ball heights with the physics-only camera, then independent tests that the
calibration never saw: the labelled touches (ball at chest / head / shoulder height) and the players' statures
(from standing players' necks, neck = 0.82 x stature).
Writes data/experiments/physics_calib/ball_height_v2.csv and check.csv."""
import os, sys, subprocess, json
import numpy as np, pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
OUT = 'data/experiments/physics_calib'
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

# ball heights: the main estimator, pointed at this camera and output folder
src = open('scripts/ball_height/estimate.py').read()
src = src.replace("cam = Camera()", f"cam = Camera('{OUT}/camera.npz')")
src = src.replace("'data/analysis/ball_height_v2.csv'", f"'{OUT}/ball_height_v2.csv'")
src = src.replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))", "sys.path.insert(0, 'scripts/ball_height')")
src = src.replace("ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))", f"ROOT = {ROOT!r}")
open(f'{OUT}/_estimate_run.py', 'w').write(src)
subprocess.run([sys.executable, '-W', 'ignore', f'{OUT}/_estimate_run.py'], check=True, capture_output=True)
os.remove(f'{OUT}/_estimate_run.py')

cam = Camera(f'{OUT}/camera.npz')
h = pd.read_csv(f'{OUT}/ball_height_v2.csv')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
tl['v2'] = tl.frame.map(fm)

# statures from upright standing poses
P = pd.read_csv('data/pose/vitpose_body25.csv')
def ang(a, b_, c):
    v1 = np.stack([P[a + '_x'] - P[b_ + '_x'], P[a + '_y'] - P[b_ + '_y']], 1)
    v2 = np.stack([P[c + '_x'] - P[b_ + '_x'], P[c + '_y'] - P[b_ + '_y']], 1)
    with np.errstate(invalid='ignore'):
        cs = (v1 * v2).sum(1) / np.linalg.norm(v1, axis=1) / np.linalg.norm(v2, axis=1)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))
ok = np.all([P[k + '_s'] > 0.7 for k in ['neck', 'mid_hip', 'r_hip', 'l_hip', 'r_knee', 'l_knee', 'r_ankle', 'l_ankle']], 0)
ok &= (ang('r_hip', 'r_knee', 'r_ankle') > 170) & (ang('l_hip', 'l_knee', 'l_ankle') > 170)
ok &= np.abs(np.degrees(np.arctan2(P.neck_x - P.mid_hip_x, P.mid_hip_y - P.neck_y))) < 6
ok &= (P.r_ankle_y - P.l_ankle_y).abs() < 8
stature = {}
for pl, S in P[ok].groupby('player'):
    hs = []
    for r in S.itertuples():
        g = cam.ground(np.array([[(r.r_ankle_x + r.l_ankle_x) / 2, (r.r_ankle_y + r.l_ankle_y) / 2 + 6]]))[0]
        try:
            hs.append(brentq(lambda z: cam.project(np.array([[g[0], g[1], -z]]))[0, 1] - r.neck_y, 0.3, 3))
        except ValueError:
            pass
    stature[pl] = round(float(np.median(hs)) / 0.82, 3)
json.dump({k: {'stature_m': v, 'source': 'measured'} for k, v in stature.items()}, open(f'{OUT}/player_statures.json', 'w'), indent=1)

m = tl.merge(h, left_on='v2', right_on='frame')
m = m[m.method == 'flight_fit']
rows = []
for part, ratio in [('chest', 0.72), ('shoulder', 0.82), ('head', 0.95), ('foot', None)]:
    g = m[m.body_part == part]
    exp = g.player.map(stature) * ratio if ratio else None
    rows.append(dict(part=part, n=len(g), median_h=round(float(g.height_m.median()), 2),
                     expected=None if exp is None else round(float(exp.median()), 2),
                     median_err=None if exp is None else round(float((g.height_m - exp).median()), 2)))
chk = pd.DataFrame(rows)
chk.to_csv(f'{OUT}/check.csv', index=False)
fit = h[h.method == 'flight_fit'].height_m
print('statures (m):', stature)
print(chk.to_string(index=False))
print(f'ball height in flight: median {fit.median():.2f}, p95 {fit.quantile(.95):.2f}, max {fit.max():.2f}')
