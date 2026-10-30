"""EXPERIMENT (improve, step 4): does the camera move? Template-match static scenery (the near post padding, the
far banner, the left baseline corner) against frame 0 every 100 frames and write the image shift
data/experiments/improve/drift.csv (frame, dx, dy, per-region shifts)."""
import os
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
REGIONS = {'post': (990, 700, 70, 300), 'banner': (1380, 270, 300, 70), 'corner': (180, 395, 110, 60), 'flag': (960, 160, 90, 120)}
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
ok, ref = cap.read()
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
rows = []
for f in range(0, 15586, 100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    row = dict(frame=f)
    for k, (x, y, w, h) in REGIONS.items():
        tpl = ref[y:y + h, x:x + w]
        win = g[y - 20:y + h + 20, x - 20:x + w + 20]
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        # sub-pixel peak by a parabola through the neighbours
        def sub(a, i):
            if 0 < i < len(a) - 1:
                d = a[i - 1] - 2 * a[i] + a[i + 1]
                return i + (0.5 * (a[i - 1] - a[i + 1]) / d if d != 0 else 0)
            return i
        sx, sy = sub(res[loc[1], :], loc[0]) - 20, sub(res[:, loc[0]], loc[1]) - 20
        row.update({f'{k}_dx': round(sx, 2), f'{k}_dy': round(sy, 2), f'{k}_score': round(score, 3)})
    rows.append(row)
d = pd.DataFrame(rows)
for ax in ('dx', 'dy'):
    cols = [f'{k}_{ax}' for k in REGIONS]
    sc = d[[f'{k}_score' for k in REGIONS]].values > 0.6
    v = np.where(sc, d[cols].values, np.nan)
    d[ax] = np.nanmedian(v, axis=1).round(2)
d.to_csv('data/experiments/improve/drift.csv', index=False)
print(d[['frame', 'dx', 'dy']].describe().round(2))
print(d[['frame', 'dx', 'dy']].iloc[::10].to_string(index=False))
