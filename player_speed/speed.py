"""Ground speed of every player in every frame.

Feet position: the lower ankle in the picture (the planted foot during a
kick) when it is confident, otherwise the bottom centre of the player's box. Pixels are mapped to the
sand with the calibrated camera (ball_height/camera.npz). Each track is
split at gaps and at the video's edit cuts, cleaned with a running median
(drops jumps and feet in the air), smoothed with a Savitzky-Golay filter
(0.84 s window) and differentiated.
Writes player_speed/player_speed.csv.
"""
import sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
sys.path.insert(0, 'ball_height')
from traj import Camera

FPS = 25
WIN = 21          # frames in the smoothing window
MED = 9           # frames in the running median
MAX_GAP = 5       # missing frames bridged by interpolation
CUTS = [3325, 8025, 13150]  # first frame after each removed replay (frame_mapping.csv)
MAX_KMH = 30      # faster than a sand sprint: feet in the air or a tracking slip

cam = Camera()
box = pd.read_csv('correct_bbox/player_boxes_v9.csv')
pose = pd.read_csv('pose/vitpose_body25.csv',
                   usecols=['frame', 'player', 'r_ankle_x', 'r_ankle_y', 'r_ankle_s',
                            'l_ankle_x', 'l_ankle_y', 'l_ankle_s'])
d = box[['frame', 'player', 'feet_x', 'feet_y', 'y2', 'source']].merge(pose, on=['frame', 'player'], how='left')
right_low = d.r_ankle_y.fillna(-1) >= d.l_ankle_y.fillna(-1)
ax = np.where(right_low, d.r_ankle_x, d.l_ankle_x)
ay = np.where(right_low, d.r_ankle_y, d.l_ankle_y)
use_pose = np.where(right_low, d.r_ankle_s, d.l_ankle_s) > 0.5
d['u'] = np.where(use_pose, ax, d.feet_x)
d['v'] = np.where(use_pose, ay + 6, d.feet_y)
g = cam.ground(d[['u', 'v']].values)
d['x'], d['y'] = g[:, 0], g[:, 1]
d['feet_cut'] = d.y2 >= 1075  # feet below the bottom of the picture

n = int(box.frame.max()) + 1
# More edit cuts (e.g. the side changes): frames where most players jump at once.
raw = d.sort_values('frame').groupby('player')[['frame', 'x', 'y']]
jump = []
for _, t in raw:
    t = t.drop_duplicates('frame')
    st = np.hypot(np.diff(t.x), np.diff(t.y))
    jump.append(t.frame.values[1:][(st > 1.0) & (np.diff(t.frame) == 1)])
jf, cnt = np.unique(np.concatenate(jump), return_counts=True)
CUTS = sorted(set(CUTS) | set(jf[cnt >= 3].tolist()))
print('cuts at frames', CUTS)
seg_id = np.searchsorted(CUTS, np.arange(n), side='right')
out = []
for pl, t in d.groupby('player'):
    t = t.drop_duplicates('frame').set_index('frame').reindex(range(n))
    x, y = t.x.values, t.y.values
    have = ~np.isnan(x)
    run = np.zeros(n, int)
    r, last = 0, None
    for f in range(n):
        if have[f]:
            if last is None or f - last > MAX_GAP + 1 or seg_id[f] != seg_id[last]:
                r += 1
            run[f], last = r, f
    sx, sy, vx, vy = (np.full(n, np.nan) for _ in range(4))
    for k in range(1, r + 1):
        idx = np.flatnonzero(run == k)
        if idx[-1] - idx[0] + 1 < WIN:
            continue
        fr = np.arange(idx[0], idx[-1] + 1)
        xi, yi = np.interp(fr, idx, x[idx]), np.interp(fr, idx, y[idx])
        xi = pd.Series(xi).rolling(MED, center=True, min_periods=1).median().values
        yi = pd.Series(yi).rolling(MED, center=True, min_periods=1).median().values
        sx[fr] = savgol_filter(xi, WIN, 2)
        sy[fr] = savgol_filter(yi, WIN, 2)
        vx[fr] = savgol_filter(xi, WIN, 2, deriv=1) * FPS
        vy[fr] = savgol_filter(yi, WIN, 2, deriv=1) * FPS
    sp = np.hypot(vx, vy)
    reliable = sp * 3.6 <= MAX_KMH
    out.append(pd.DataFrame({
        'frame': np.arange(n), 'time_sec': np.arange(n) / FPS, 'player': pl,
        'court_x_m': sx.round(3), 'court_y_m': sy.round(3),
        'speed_mps': sp.round(3), 'speed_kmh': (sp * 3.6).round(2),
        'reliable': reliable,
        'feet_cut': t.feet_cut.fillna(False).astype(bool).values,
        'box_source': t.source.values}))
res = pd.concat(out).sort_values(['frame', 'player'])
res.to_csv('player_speed/player_speed.csv', index=False)
for pl, t in res.groupby('player'):
    s = t.speed_kmh[t.reliable]
    step = np.hypot(np.diff(t.court_x_m), np.diff(t.court_y_m))
    print(f'{pl}: coverage {len(s) / n:.0%}  median {s.median():.1f} km/h  p99 {s.quantile(.99):.1f}  '
          f'max {s.max():.1f}  distance {np.nansum(step):.0f} m')
