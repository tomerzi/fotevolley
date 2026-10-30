"""EXPERIMENT (shadow): fit the sun direction (azimuth, elevation, azimuth drift over the match) to the detected
ball shadows, using the current ball positions. Robust (soft-L1) least squares on the pixel distance between the
detected shadow and the predicted one. Writes OUT/sun.json and OUT/shadows_fit.csv (with residuals).
usage: python scripts/experiments/shadow/sun_fit.py HEIGHT_CSV OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, out = sys.argv[1], sys.argv[2]
cam = Camera()
h = pd.read_csv(csv).set_index('frame')
s = pd.read_csv(f'{out}/shadows.csv')
s = s.join(h[['court_x_m', 'court_y_m', 'height_m']], on='frame')


def pred(p, d):
    az = np.radians(p[0] + p[2] * d.frame.values / 1000.0)
    k = d.height_m.values / np.tan(np.radians(p[1]))
    G = np.c_[d.court_x_m + k * np.cos(az), d.court_y_m + k * np.sin(az), np.zeros(len(d))]
    return cam.project(G)


def res(p, d):
    return (pred(p, d) - d[['u', 'v']].values).ravel() / 3.0


sol = least_squares(res, [-108, 43.5, 0.0], args=(s,), loss='soft_l1', f_scale=2.0)
e = np.linalg.norm(pred(sol.x, s) - s[['u', 'v']].values, axis=1)
s['err_px'] = e.round(2)
good = s[e < 12]
sol = least_squares(res, sol.x, args=(good,), loss='soft_l1', f_scale=2.0)
e = np.linalg.norm(pred(sol.x, s) - s[['u', 'v']].values, axis=1)
s['err_px'] = e.round(2)
az, el, dr = sol.x
json.dump(dict(az=az, elev=el, drift_deg_per_1000=dr), open(f'{out}/sun.json', 'w'), indent=1)
s.to_csv(f'{out}/shadows_fit.csv', index=False)
print(f'sun: azimuth {az:.1f} deg (drift {dr:+.2f} deg / 1000 frames), elevation {el:.1f} deg; '
      f'{len(s)} shadows, {np.mean(e < 12) * 100:.0f}% within 12 px, median error {np.median(e[e < 12]):.1f} px')
