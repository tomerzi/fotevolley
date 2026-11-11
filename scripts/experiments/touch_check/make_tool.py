"""Build the touch verification page (an artifact with a shared database).

One entry per real touch (data/contacts/touches_master.csv, kind == player), in a fixed random order so that any
number of checked touches is an unbiased sample. For each touch: a strip of 7 frames (t-6 .. t+6, every 2nd frame)
cropped around the player and the ball, and the numbers the page draws on the touch frame:
  - the player's ground point (data/analysis/player_positions_v3.csv) and a metre ruler rising from it to the
    player's stature, projected with the camera (data/analysis/camera_v2.npz);
  - the ball's computed 3-D position (data/analysis/ball_height_v2.csv): its point on the sand and the pole up to it.
The viewer checks who touched, with what, whether the feet were on the sand, whether the ground point sits at the
feet, and whether the ball's point on the sand is at the player's depth (a wrong height moves it nearer or farther).

Writes OUT/strips/t<id>.jpg (kept as a cache), OUT/sheets/s<n>.jpg (4 strips each, what the page loads) and
OUT/touches.json. Needs ffmpeg and Pillow (no OpenCV).
usage: python scripts/experiments/touch_check/make_tool.py OUT_DIR
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
OUT = sys.argv[1]
os.makedirs(f'{OUT}/strips', exist_ok=True)
VIDEO = 'video/main_camera_clean_v3.mp4'
W, H, FPS = 1920, 1080, 25
OFFS = [-6, -4, -2, 0, 2, 4, 6]
TILE = 400            # px per strip tile
MIN_CROP = 380        # px of the source frame
PER = 4               # strips per sheet
SEED = 20261111


class Camera:
    """Same model as scripts/ball_height/traj.py (X along the court, Y toward the camera, Z down), numpy only."""

    def __init__(self, path='data/analysis/camera_v2.npz'):
        c = np.load(path)
        self.K, self.dist, self.R, self.t = c['K'], c['dist'], c['R'], c['t']

    def project(self, X):
        p = np.asarray(X, float).reshape(-1, 3) @ self.R.T + self.t
        x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.c_[self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]]


def frames(first, n):
    """n consecutive frames starting at `first` as (n, H, W, 3) uint8; missing ones are black."""
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{first / FPS:.3f}', '-i', VIDEO, '-frames:v', str(n),
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    a = np.frombuffer(out, np.uint8)
    got = len(a) // (W * H * 3)
    a = a[:got * W * H * 3].reshape(got, H, W, 3)
    return np.concatenate([a, np.zeros((n - got, H, W, 3), np.uint8)]) if got < n else a


def r1(v):
    return [round(float(x), 1) for x in v]


cam = Camera()
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[t.kind == 'player'].sample(frac=1, random_state=SEED).reset_index(drop=True)
pos = pd.read_csv('data/analysis/player_positions_v3.csv',
                  usecols=['frame', 'player', 'x', 'y', 'posture', 'jump', 'airborne']).set_index(['frame', 'player'])
box = pd.read_csv('data/boxes/player_boxes_v3video.csv',
                  usecols=['frame', 'player', 'x1', 'y1', 'x2', 'y2']).set_index(['frame', 'player'])
ballpx = pd.read_csv('data/boxes/ball_boxes_v3video.csv').set_index('frame')
stat = {k: v['stature_m'] for k, v in json.load(open('data/analysis/player_statures.json')).items()}
last = int(ballpx.index.max())

items = []
for n, r in enumerate(t.itertuples()):
    f = int(r.frame)
    pts = []                                   # full-image pixels the crop must contain
    b = box.loc[(f, r.player)] if (f, r.player) in box.index else None
    if b is not None:
        pts += [(b.x1, b.y1), (b.x2, b.y2)]
    g = pos.loc[(f, r.player)] if (f, r.player) in pos.index else None
    ruler = None
    if g is not None and not np.isnan(g.x):
        hs = list(np.arange(0, stat[r.player], 0.5)) + [stat[r.player]]
        ruler = cam.project([[g.x, g.y, -h] for h in hs])
        pts += [tuple(ruler[0]), tuple(ruler[-1])]
    pole = None
    if not np.isnan(r.ball_height_m):
        pole = cam.project([[r.ball_court_x_m, r.ball_court_y_m, 0], [r.ball_court_x_m, r.ball_court_y_m, -r.ball_height_m]])
        pts += [tuple(pole[0]), tuple(pole[1])]
    bp = ballpx.loc[f] if f in ballpx.index else None
    if bp is not None and not np.isnan(bp.center_x):
        pts.append((bp.center_x, bp.center_y))
    if not pts:
        continue
    P = np.clip(np.array(pts, float), [0, 0], [W - 1, H - 1])
    side = int(np.clip(max(MIN_CROP, np.ptp(P[:, 0]) + 140, np.ptp(P[:, 1]) + 140), MIN_CROP, H))
    cx, cy = (P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 1].min() + P[:, 1].max()) / 2
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))

    path = f'{OUT}/strips/t{int(r.touch)}.jpg'
    if not os.path.exists(path):               # delete OUT/strips to re-cut the frames
        first = f + OFFS[0]
        fr = frames(max(first, 0), OFFS[-1] - OFFS[0] + 1 - max(0, -first))
        strip = Image.new('RGB', (TILE * len(OFFS), TILE))
        for i, k in enumerate(OFFS):
            j = f + k - max(first, 0)
            if 0 <= j < len(fr):
                strip.paste(Image.fromarray(fr[j][y0:y0 + side, x0:x0 + side]).resize((TILE, TILE), Image.LANCZOS), (i * TILE, 0))
        strip.save(path, quality=80, optimize=True)
    balls = []
    for k in OFFS:
        q = ballpx.loc[f + k] if 0 <= f + k <= last and f + k in ballpx.index else None
        balls.append(None if q is None or np.isnan(q.center_x) else r1([q.center_x, q.center_y, (q.x2 - q.x1 + q.y2 - q.y1) / 4]))

    items.append(dict(
        id=int(r.touch), frame=f, rally=int(r.rally), set=int(r.set), inplay=bool(r.in_play), player=r.player,
        part=r.body_part, side=None if pd.isna(r.limb_side) else r.limb_side,
        action=None if pd.isna(r.action) else r.action, x0=x0, y0=y0, s=side,
        box=None if b is None else r1([b.x1, b.y1, b.x2, b.y2]),
        ruler=None if ruler is None else [r1(p) for p in ruler], stature=stat[r.player],
        pole=None if pole is None else [r1(p) for p in pole],
        h=None if np.isnan(r.ball_height_m) else round(float(r.ball_height_m), 2),
        hs=None if np.isnan(r.ball_height_sigma_m) else float(r.ball_height_sigma_m), hm=r.ball_method, balls=balls,
        # what the data says, kept out of sight on the page and compared with the answers afterwards
        auto=dict(airborne=None if g is None else bool(g.airborne), posture=None if g is None else g.posture,
                  part=None if pd.isna(r.auto_part) else r.auto_part)))
    if n % 40 == 0:
        print(n, len(t), flush=True)

# the page loads sheets of PER strips stacked top to bottom (fewer files to publish than one per touch)
os.makedirs(f'{OUT}/sheets', exist_ok=True)
for s in range(0, len(items), PER):
    sheet = Image.new('RGB', (TILE * len(OFFS), TILE * PER))
    for row, it in enumerate(items[s:s + PER]):
        sheet.paste(Image.open(f"{OUT}/strips/t{it['id']}.jpg"), (0, row * TILE))
        it['sheet'], it['row'] = s // PER, row
    sheet.save(f'{OUT}/sheets/s{s // PER}.jpg', quality=80, optimize=True)

json.dump(dict(tile=TILE, per=PER, offs=OFFS, touches=items), open(f'{OUT}/touches.json', 'w'), separators=(',', ':'))
print(len(items), 'touches ->', OUT)
