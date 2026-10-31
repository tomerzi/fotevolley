"""EXPERIMENT (option 2): ball position from the ball and its shadow in a single frame, for frames outside fitted
flights (ball held, at a touch, rolling), where the height otherwise comes from the ball's size only.
On the camera ray through the ball, pick the point whose shadow (along the sun) projects onto the detected shadow.
Writes OUT/triangulated.csv (frame, X, Y, height, residual px) and OUT/<HEIGHT_CSV name>_tri.csv with those frames
replaced where the residual is small (method 'shadow_tri').
usage: python scripts/experiments/shadow/triangulate.py HEIGHT_CSV SHADOWS_CSV SUN_JSON CAMERA OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, sh_csv, sun_json, cam_path, out = sys.argv[1:6]
cam = Camera(cam_path)
sun = json.load(open(sun_json))
h = pd.read_csv(csv)
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv').set_index('frame')
sd = pd.read_csv(sh_csv)
sd = sd[sd.method == 'size_only'] if 'method' in sd else sd
rows = []
for r in sd.itertuples():
    f = int(r.frame)
    if f not in b.index or np.isnan(b.center_x[f]):
        continue
    d = cam.ray(np.array([[b.center_x[f], b.center_y[f]]]))[0]
    a = np.radians(sun['az'] + sun['drift_deg_per_1000'] * f / 1000.0)
    s = np.array([np.cos(a), np.sin(a)]) / np.tan(np.radians(sun['elev']))

    def err(t):
        P = cam.C + t * d
        if P[2] > 0:
            return 1e6
        G = np.r_[P[:2] + (-P[2]) * s, 0.0]
        return float(np.sum((cam.project(G[None])[0] - [r.u, r.v]) ** 2))
    res = minimize_scalar(err, bounds=(2.0, 40.0), method='bounded')
    P = cam.C + res.x * d
    rows.append(dict(frame=f, X=P[0], Y=P[1], height=-P[2], resid_px=np.sqrt(res.fun)))
t = pd.DataFrame(rows)
os.makedirs(out, exist_ok=True)
t.round(4).to_csv(f'{out}/triangulated.csv', index=False)
ok = t[(t.resid_px < 3) & (t.height > -0.05)]
h2 = h.set_index('frame')
h2.loc[ok.frame, 'court_x_m'] = ok.X.round(3).values
h2.loc[ok.frame, 'court_y_m'] = ok.Y.round(3).values
h2.loc[ok.frame, 'height_m'] = ok.height.clip(lower=0).round(3).values
h2.loc[ok.frame, 'method'] = 'shadow_tri'
name = os.path.splitext(os.path.basename(csv))[0]
h2.reset_index().to_csv(f'{out}/{name}_tri.csv', index=False)
print(f'{len(t)} size-only frames with a shadow, {len(ok)} triangulated (residual < 3 px)')

# --- frames at a labelled touch (+-2 frames) outside fitted flights: the ball is at the touching player's body,
# so its depth on the camera ray is the player's (closest point of the ray to the player's vertical line, 0.15 m in
# front of it towards the camera). Overrides the size-only estimate there (method 'touch_depth').
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv'); tl = tl[tl.kind == 'player'].copy()
tl['v2'] = tl.frame.map(fm); tl = tl.dropna(subset=['v2'])
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
n_t = 0
for r in tl.itertuples():
    for f in range(int(r.v2) - 2, int(r.v2) + 3):
        if f not in h2.index or h2.method[f] != 'size_only' or f not in b.index or np.isnan(b.center_x[f]):
            continue
        try:
            px, py = pos.loc[(f, r.player)].values.ravel()[:2]
        except KeyError:
            continue
        px, py = px * 16 / 18, py * 8 / 9
        d = cam.ray(np.array([[b.center_x[f], b.center_y[f]]]))[0]
        dxy = d[:2]; t_ = np.dot(np.array([px, py]) - cam.C[:2], dxy) / np.dot(dxy, dxy) - 0.15 / np.linalg.norm(dxy)
        P = cam.C + t_ * d
        if -P[2] < 0 or -P[2] > 4:
            continue
        h2.loc[f, ['court_x_m', 'court_y_m', 'height_m', 'method']] = [round(P[0], 3), round(P[1], 3), round(-P[2], 3), 'touch_depth']
        n_t += 1
h2.reset_index().to_csv(f'{out}/{name}_tri.csv', index=False)
print(f'{n_t} size-only frames at a touch placed at the player\'s depth')
