"""EXPERIMENT (shadow): find the ball's shadow on the sand in every flight frame.

The camera is static, so a background is the median of frames f-60..f+60 (step 20, f excluded); the moving ball
shadow is darker than it. For a ball at height h the shadow lies h / tan(elevation) along the sun's horizontal
direction from the point under the ball. With the current ball position (HEIGHT_CSV) and a sun guess, the shadow's
pixel position is predicted and the darkest compact blob of about the ball's size is searched in a window around it.
Writes OUT/shadows.csv (frame, u, v, darkness, area, expected area, distance to the prediction).
usage: python scripts/experiments/shadow/detect.py HEIGHT_CSV OUT_DIR AZ_DEG ELEV_DEG [AZ_DRIFT_DEG_PER_1000_FRAMES]
"""
import os, sys
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, out = sys.argv[1], sys.argv[2]
AZ, EL = float(sys.argv[3]), float(sys.argv[4])
DRIFT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
os.makedirs(out, exist_ok=True)
cam = Camera()
h = pd.read_csv(csv)
pb = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
pbf = {f: g[['x1', 'y1', 'x2', 'y2']].values for f, g in pb.groupby('frame')}
cand = h[(h.method == 'flight_fit') & (h.height_m >= 0.35)].set_index('frame')
WIN, OFFS = 45, (-60, -40, -20, 20, 40, 60)


def sun(f):
    a = np.radians(AZ + DRIFT * f / 1000.0)
    return np.array([np.cos(a), np.sin(a)]) / np.tan(np.radians(EL))


cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
buf = {}
rows = []
need = set(cand.index)
nxt = 0
for f in range(N):
    ok, im = cap.read()
    if not ok:
        break
    buf[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = f - 60  # the frame whose background is now complete
    for k in [k for k in buf if k < t - 60]:
        del buf[k]
    if t not in need:
        continue
    r = cand.loc[t]
    G = np.array([r.court_x_m, r.court_y_m]) + r.height_m * sun(t)
    uv = cam.project(np.array([[G[0], G[1], 0.0]]))[0]
    if not (WIN < uv[0] < 1920 - WIN and WIN < uv[1] < 1080 - WIN):
        continue
    x0, y0 = int(uv[0]) - WIN, int(uv[1]) - WIN
    sl = (slice(y0, y0 + 2 * WIN), slice(x0, x0 + 2 * WIN))
    bg = np.median(np.stack([buf[t + o][sl] for o in OFFS if t + o in buf]), axis=0)
    ratio = cv2.GaussianBlur(buf[t][sl], (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(bg, (0, 0), 1.0), 1)
    mask = (ratio < 0.80).astype(np.uint8)
    # ignore player bodies: inside player boxes the change is the player, not a shadow
    for x1, y1, x2, y2 in pbf.get(t, []):
        a, b_, c, d = int(x1) - x0, int(y1) - y0, int(x2) - x0, int(y2) - y0
        mask[max(0, b_):max(0, d), max(0, a):max(0, c)] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    dist = np.linalg.norm(np.array([G[0], G[1], 0.0]) - cam.C)
    rpx = cam.K[0, 0] * 0.11 / dist
    exp_area = np.pi * rpx ** 2 / np.sin(np.radians(EL))
    best = None
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not (0.35 * exp_area <= area <= 3.0 * exp_area):
            continue
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if area < 0.35 * w_ * h_:  # not compact
            continue
        m = lab == i
        wgt = (1 - ratio[m])
        cy, cx = np.nonzero(m)
        u, v = x0 + np.sum(cx * wgt) / wgt.sum(), y0 + np.sum(cy * wgt) / wgt.sum()
        dk = float(1 - ratio[m].mean())
        dpred = float(np.hypot(u - uv[0], v - uv[1]))
        score = dk - dpred / 200 - abs(np.log(area / exp_area)) * 0.05
        if best is None or score > best[0]:
            best = (score, u, v, dk, area, dpred)
    if best:
        rows.append(dict(frame=t, u=round(best[1], 2), v=round(best[2], 2), dark=round(best[3], 3), area=int(best[4]),
                         exp_area=round(exp_area, 1), dpred=round(best[5], 1), pred_u=round(uv[0], 1), pred_v=round(uv[1], 1),
                         ball_h=r.height_m))
    if t % 2000 < 1:
        print(t, len(rows), flush=True)
d = pd.DataFrame(rows)
d.to_csv(f'{out}/shadows.csv', index=False)
print(f'{len(d)} shadows found in {len(cand)} flight frames with the ball >= 0.35 m up')
