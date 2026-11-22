"""EXPERIMENT (ball size, 2nd try): the ball's diameter from its silhouette against the background.
The camera is static: background = median of frames f-60..f+60 (step 20, f excluded). The ball is the blob of
changed pixels (max colour difference > 28) that contains the detected centre; its diameter is taken from the
blob's second moments along the minor axis (motion blur stretches the ball along its path, not across it):
diameter = 4 * sqrt(smallest eigenvalue). Frames where the blob merges with a player are dropped.
(The first try, circle.py, locked onto the ball's inner pattern and measured ~13 px balls.)
Writes data/experiments/ball_size/silhouette.csv (frame, diam_px, area, box_px, ok).
"""
import os
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
b = b[b.source.isin(['detected', 'rescued', 'manual']) & b.center_x.notna()].set_index('frame')
pb = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
pbf = {f: g[['x1', 'y1', 'x2', 'y2']].values for f, g in pb.groupby('frame')}
OFFS = (-60, -40, -20, 20, 40, 60)
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
buf, rows = {}, []
need = set(b.index)
for f in range(N):
    ok, im = cap.read()
    if not ok:
        break
    buf[f] = im
    t = f - 60
    for k in [k for k in buf if k < t - 60]:
        del buf[k]
    if t not in need:
        continue
    r = b.loc[t]
    box = float(np.sqrt((r.x2 - r.x1) * (r.y2 - r.y1)))
    R = int(box * 1.5) + 8
    x0, y0 = int(r.center_x) - R, int(r.center_y) - R
    if x0 < 0 or y0 < 0 or x0 + 2 * R > 1920 or y0 + 2 * R > 1080:
        continue
    sl = (slice(y0, y0 + 2 * R), slice(x0, x0 + 2 * R))
    bg = np.median(np.stack([buf[t + o][sl] for o in OFFS if t + o in buf]), axis=0).astype(np.float32)
    diff = np.abs(buf[t][sl].astype(np.float32) - bg).max(axis=2)
    mask = (cv2.GaussianBlur(diff, (0, 0), 0.6) > 28).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    c = lab[R, R] if 0 <= R < lab.shape[0] else 0
    if c == 0:  # centre not on the blob: take the component nearest to the centre
        if n <= 1:
            continue
        c = 1 + int(np.argmin(np.hypot(cen[1:, 0] - R, cen[1:, 1] - R)))
    m = lab == c
    ys, xs = np.nonzero(m)
    touches_edge = xs.min() == 0 or ys.min() == 0 or xs.max() == 2 * R - 1 or ys.max() == 2 * R - 1
    near_player = any(x1 - 5 < r.center_x < x2 + 5 and y1 - 5 < r.center_y < y2 + 5 for x1, y1, x2, y2 in pbf.get(t, []))
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 4 else np.eye(2)
    lam = np.linalg.eigvalsh(cov)
    diam = 4 * np.sqrt(max(lam[0], 1e-6))
    rows.append(dict(frame=t, diam_px=round(float(diam), 2), area=int(m.sum()), box_px=round(box, 2),
                     ok=bool(not touches_edge and not near_player and 0.5 * box < diam < 1.8 * box)))
    if len(rows) % 2000 == 0:
        print(len(rows), flush=True)
d = pd.DataFrame(rows)
d.to_csv('data/experiments/ball_size/silhouette.csv', index=False)
print(len(d), 'balls,', int(d.ok.sum()), 'clean; diameter/box median', round((d.diam_px / d.box_px)[d.ok].median(), 3))
