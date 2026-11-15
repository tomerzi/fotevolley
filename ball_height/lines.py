"""Sub-pixel court-line extraction from the median background image.
Writes line samples (image points known to lie on a given court line) to court_line_points.csv"""
import numpy as np, sys, os
from PIL import Image
BG = sys.argv[1]
OUT = sys.argv[2]
im = np.asarray(Image.open(BG)).astype(float)
R, G, B = im[..., 0], im[..., 1], im[..., 2]
blue = B - R  # sand: strongly negative, blue tape: near zero / positive

def peak(profile):
    """sub-pixel position + contrast of the maximum of a 1-D profile above its local baseline"""
    base = np.median(profile)
    p = profile - base
    i = int(np.argmax(p))
    if i < 2 or i > len(p) - 3:
        return None
    w = np.clip(p[i - 3:i + 4] if i >= 3 and i <= len(p) - 4 else p[i - 2:i + 3], 0, None)
    off = np.arange(len(w)) - len(w) // 2
    if w.sum() <= 0:
        return None
    return i + (w * off).sum() / w.sum(), p[i]

rows = []
def scan_cols(name, x0, x1, yfun, half, step=4, thr=12):
    for x in range(x0, x1, step):
        yc = int(round(yfun(x)))
        prof = blue[yc - half:yc + half + 1, x - 1:x + 2].mean(1)
        r = peak(prof)
        if r and r[1] > thr:
            rows.append((name, x, yc - half + r[0], r[1]))
def scan_rows(name, y0, y1, xfun, half, step=3, thr=12):
    for y in range(y0, y1, step):
        xc = int(round(xfun(y)))
        if xc - half < 0 or xc + half >= blue.shape[1]:
            continue
        prof = blue[y - 1:y + 2, xc - half:xc + half + 1].mean(0)
        r = peak(prof)
        if r and r[1] > thr:
            rows.append((name, xc - half + r[0], y, r[1]))

scan_cols('far', 245, 1785, lambda x: 420 + (x - 232) * 3 / 1563, 14)
scan_cols('near', 4, 1916, lambda x: 962 + 24 * min(x, 1000) / 1000, 22)
scan_rows('left', 428, 664, lambda y: 232 - 0.947 * (y - 420), 22)
scan_rows('right', 428, 545, lambda y: 1795 + 1.02 * (y - 423), 22)
import csv
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['line', 'u', 'v', 'contrast']); w.writerows(rows)
import collections
print(collections.Counter(r[0] for r in rows))
