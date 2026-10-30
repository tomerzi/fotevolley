"""EXPERIMENT (ball size): measure the ball's diameter in pixels by fitting a circle to its edge, instead of
using the detector's box size (boxes are squeezed towards ~25 px).
For every detected ball box: full-resolution crop, gradient magnitude, and a search over centre (+-3 px, 0.5 px
steps) and radius (0.55..1.45 x half the box, 0.25 px steps) for the circle with the strongest mean edge.
Writes data/experiments/ball_size/circle.csv (frame, cx, cy, diam_px, edge, box_px).
usage: python scripts/experiments/ball_size/circle.py
"""
import os
import numpy as np, pandas as pd, cv2
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
OUT = 'data/experiments/ball_size'
os.makedirs(OUT, exist_ok=True)
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
b = b[b.source.isin(['detected', 'rescued', 'manual']) & b.center_x.notna()].set_index('frame')
ANG = np.linspace(0, 2 * np.pi, 48, endpoint=False)
DC = np.arange(-3, 3.01, 0.5)
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
rows = []
f = -1
for fr, r in b.iterrows():
    while f < fr:
        ok, im = cap.read(); f += 1
    if not ok:
        break
    box = float(np.sqrt((r.x2 - r.x1) * (r.y2 - r.y1)))
    R = int(box) + 10
    cx0, cy0 = r.center_x, r.center_y
    x0, y0 = int(cx0) - R, int(cy0) - R
    if x0 < 0 or y0 < 0 or x0 + 2 * R >= im.shape[1] or y0 + 2 * R >= im.shape[0]:
        continue
    c = im[y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.float32)
    g = np.zeros(c.shape[:2], np.float32)
    for ch in range(3):  # colour edge strength
        gx = cv2.Sobel(c[..., ch], cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(c[..., ch], cv2.CV_32F, 0, 1, ksize=3)
        g = np.maximum(g, np.hypot(gx, gy))
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    radii = np.arange(0.55, 1.451, 0.05) * box / 2
    best = (-1, 0, 0, 0)
    ccx, ccy = cx0 - x0, cy0 - y0
    for dx in DC:
        for dy in DC:
            xs = ccx + dx + np.outer(radii, np.cos(ANG)); ys = ccy + dy + np.outer(radii, np.sin(ANG))
            e = map_coordinates(g, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(radii), -1)
            # robust: the ball edge is often hidden on one side (a foot, the sand): use the 70th percentile
            s = np.percentile(e, 70, axis=1)
            k = int(np.argmax(s))
            if s[k] > best[0]:
                best = (s[k], dx, dy, radii[k])
    # refine radius with 0.1 px steps around the best
    rr = np.arange(best[3] - 0.5, best[3] + 0.51, 0.1)
    xs = ccx + best[1] + np.outer(rr, np.cos(ANG)); ys = ccy + best[2] + np.outer(rr, np.sin(ANG))
    e = np.percentile(map_coordinates(g, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(rr), -1), 70, axis=1)
    rad = rr[int(np.argmax(e))]
    rows.append(dict(frame=fr, cx=round(cx0 + best[1], 2), cy=round(cy0 + best[2], 2), diam_px=round(2 * rad, 2),
                     edge=round(float(best[0]), 1), box_px=round(box, 2)))
    if len(rows) % 2000 == 0:
        print(len(rows), flush=True)
d = pd.DataFrame(rows)
d.to_csv(f'{OUT}/circle.csv', index=False)
print(len(d), 'balls measured; diameter vs box: median ratio', round((d.diam_px / d.box_px).median(), 3))
