"""EXPERIMENT (shadow track search): find faint or partly hidden ball shadows in flights where too few were found
frame by frame. For each such flight, the whole flight is shifted along the camera rays by a depth factor s
(0.80 .. 1.20: the one thing a single camera cannot see) and, for every s, the predicted shadow path is scored by the
darkness under it summed over all the flight's frames (background = median of frames f-60..f+60; player boxes masked).
The best s gives the path; along it, each frame's shadow is the dark blob within +-8 px.
Writes OUT/tracked.csv (frame, u, v, flight_id, s, contrast).
usage: python scripts/experiments/shadow/track_shadows.py HEIGHT_CSV SUN_JSON OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, sun_json, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
cam = Camera()
sun = json.load(open(sun_json))
h = pd.read_csv(csv)
pb = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
pbf = {f: g[['x1', 'y1', 'x2', 'y2']].values for f, g in pb.groupby('frame')}
S = np.linspace(0.80, 1.20, 17)
OFFS = (-60, -40, -20, 20, 40, 60)
tgt = h[(h.method == 'flight_fit') & (h.flight_shadows < 3) & (h.height_m >= 0.35)].set_index('frame')


def sun_xy(f):
    a = np.radians(sun['az'] + sun['drift_deg_per_1000'] * f / 1000.0)
    el = sun['elev'] + sun.get('elev_drift_deg_per_1000', 0.0) * f / 1000.0
    return np.array([np.cos(a), np.sin(a)]) / np.tan(np.radians(el))


def predict(f, r):
    P = np.array([r.court_x_m, r.court_y_m, -r.height_m])
    Ps = cam.C + S[:, None] * (P - cam.C)
    G = Ps[:, :2] + (-Ps[:, 2:3]) * sun_xy(f)
    return cam.project(np.c_[G, np.zeros(len(S))]), np.linalg.norm(np.c_[G, np.zeros(len(S))] - cam.C, axis=1)


store = {}
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
buf = {}
for f in range(N):
    ok, im = cap.read()
    if not ok:
        break
    buf[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = f - 60
    for k in [k for k in buf if k < t - 60]:
        del buf[k]
    if t not in tgt.index:
        continue
    r = tgt.loc[t]
    uv, dist = predict(t, r)
    x0, y0 = int(np.floor(uv[:, 0].min())) - 16, int(np.floor(uv[:, 1].min())) - 16
    x1, y1 = int(np.ceil(uv[:, 0].max())) + 16, int(np.ceil(uv[:, 1].max())) + 16
    if x0 < 0 or y0 < 0 or x1 > 1920 or y1 > 1080 or (x1 - x0) * (y1 - y0) > 300 * 300:
        continue
    sl = (slice(y0, y1), slice(x0, x1))
    bg = np.median(np.stack([buf[t + o][sl] for o in OFFS if t + o in buf]), axis=0)
    ratio = cv2.GaussianBlur(buf[t][sl], (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(bg, (0, 0), 1.0), 1)
    for bx1, by1, bx2, by2 in pbf.get(t, []):
        ratio[max(0, int(by1) - y0):max(0, int(by2) - y0), max(0, int(bx1) - x0):max(0, int(bx2) - x0)] = 1.0
    store[t] = (x0, y0, ratio.astype(np.float16), uv - [x0, y0], cam.K[0, 0] * 0.11 / dist, int(r.flight_id))
    if t % 2000 < 1:
        print(t, len(store), flush=True)

rows, summ = [], []
by_flight = {}
for t, v in store.items():
    by_flight.setdefault(v[5], []).append(t)
for fid, frames in by_flight.items():
    if len(frames) < 4:
        continue
    E = np.zeros(len(S))
    for t in frames:
        x0, y0, rat, uvl, rpx, _ = store[t]
        for k, (u, v) in enumerate(uvl):
            iu, iv = int(round(u)), int(round(v))
            w = rat[max(0, iv - 2):iv + 3, max(0, iu - 2):iu + 3]
            if w.size:
                E[k] += max(0.0, 0.95 - float(w.min()))
    E /= len(frames)
    kb = int(np.argmax(E)); contrast = E[kb] - np.median(E)
    summ.append(dict(flight_id=fid, frames=len(frames), s=S[kb], evidence=round(E[kb], 3), contrast=round(contrast, 3)))
    if contrast < 0.04 or kb in (0, len(S) - 1):
        continue
    for t in frames:
        x0, y0, rat, uvl, rpx, _ = store[t]
        u, v = uvl[kb]; rp = rpx[kb]
        a, b = max(0, int(v) - 8), max(0, int(u) - 8)
        win = rat[a:int(v) + 9, b:int(u) + 9].astype(np.float32)
        if not win.size or win.min() > 0.88:
            continue
        m = (win < 0.88).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        iy, ix = np.unravel_index(np.argmin(win), win.shape)
        c = lab[iy, ix]
        area = st[c, cv2.CC_STAT_AREA]
        if not (0.3 * np.pi * rp ** 2 <= area <= 4 * np.pi * rp ** 2):
            continue
        ys, xs = np.nonzero(lab == c); wgt = 1 - win[ys, xs]
        rows.append(dict(frame=t, u=round(x0 + b + np.sum(xs * wgt) / wgt.sum(), 2), v=round(y0 + a + np.sum(ys * wgt) / wgt.sum(), 2),
                         flight_id=fid, s=round(S[kb], 3), contrast=round(contrast, 3)))
pd.DataFrame(summ).to_csv(f'{out}/flights.csv', index=False)
pd.DataFrame(rows).to_csv(f'{out}/tracked.csv', index=False)
sm = pd.DataFrame(summ)
print(f'{len(sm)} flights searched, {int((sm.contrast >= 0.04).sum())} with a clear shadow path; {len(rows)} shadows found along the paths')
