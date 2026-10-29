"""Build the shadow-marking page: frames where the automatic detector found no ball shadow (or one far from the
expected spot), plus some frames with a confident detection as a control (to measure the detector's accuracy).
For each: a strip of 7 frames (t-3..t+3), CROP x CROP px at full resolution around the predicted shadow.
Writes OUT/strips/s<frame>.jpg and OUT/shadows.json.
usage: python scripts/experiments/shadow_marks/make_tool.py HEIGHT_CSV SUN_JSON SHADOWS_FIT_CSV OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, sun_json, det_csv, OUT = sys.argv[1:5]
os.makedirs(f'{OUT}/strips', exist_ok=True)
CROP, HALF, N_MISS, N_CTRL = 200, 3, 100, 25
cam = Camera()
h = pd.read_csv(csv)
sun = json.load(open(sun_json))
det = pd.read_csv(det_csv).set_index('frame')
ral = pd.read_csv('data/points/rallies_v2.csv')
inplay = np.zeros(len(h), bool)
for r in ral.itertuples():
    inplay[r.start_frame:r.end_frame + 1] = True


def shadow_px(f):
    r = h.loc[f]
    a = np.radians(sun['az'] + sun['drift_deg_per_1000'] * f / 1000.0)
    k = r.height_m / np.tan(np.radians(sun['elev']))
    return cam.project(np.array([[r.court_x_m + k * np.cos(a), r.court_y_m + k * np.sin(a), 0.0]]))[0]


fl = h[(h.method == 'flight_fit') & (h.height_m >= 0.5) & inplay[h.frame]]
rng = np.random.default_rng(7)
miss, ctrl = [], []
for fid, g in fl.groupby('flight_id'):
    cands = []
    for f in g.frame.values[2:-2]:
        uv = shadow_px(f)
        if not (CROP < uv[0] < 1920 - CROP and CROP / 2 < uv[1] < 1080 - CROP / 2):
            continue
        d = det.loc[f] if f in det.index else None
        if d is None or d.err_px > 12:
            cands.append((f, uv, 'missing' if d is None else 'far'))
        elif d.err_px < 5:
            ctrl.append((f, uv, 'control'))
    if cands:
        miss.append(cands[len(cands) // 2])
rng.shuffle(miss)
idx = rng.choice(len(ctrl), size=min(N_CTRL, len(ctrl)), replace=False)
items = sorted(miss[:N_MISS] + [ctrl[i] for i in idx], key=lambda x: x[0])
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
out = []
for f, uv, kind in items:
    x0 = int(np.clip(uv[0] - CROP / 2, 0, 1920 - CROP)); y0 = int(np.clip(uv[1] - CROP / 2, 0, 1080 - CROP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - HALF)
    tiles = []
    for k in range(-HALF, HALF + 1):
        ok, im = cap.read()
        tiles.append(im[y0:y0 + CROP, x0:x0 + CROP] if ok else np.zeros((CROP, CROP, 3), np.uint8))
    cv2.imwrite(f'{OUT}/strips/s{f}.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    out.append(dict(id=int(f), x0=x0, y0=y0, kind=kind, pred=[round(float(uv[0] - x0), 1), round(float(uv[1] - y0), 1)],
                    h=round(float(h.height_m[f]), 2)))
json.dump(dict(crop=CROP, half=HALF, items=out), open(f'{OUT}/shadows.json', 'w'), separators=(',', ':'))
print(len(out), 'frames:', {k: sum(o['kind'] == k for o in out) for k in ('missing', 'far', 'control')})
