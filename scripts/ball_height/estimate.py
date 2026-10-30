"""Ball height above the sand (ground = 0) for every frame.

Pipeline:
  1. Split the 2-D ball track into flights: a flight grows frame by frame
     while one ballistic 3-D trajectory still reprojects within a few px.
  2. Fit each flight with gravity plus a weak prior from the ball's
     apparent size (distance = f * D / size).
  3. Frames outside any flight fall back to the size-only depth.
Outputs data/analysis/ball_height_v2.csv.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj import Camera, simulate, initial_guess, G
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)  # all paths below are relative to the repo root


BALL_D = 0.22      # ball diameter (given)
SIZE_SIGMA = 0.12  # relative noise of a single size-based distance
PX_SIGMA = 1.5
MAX_GAP = 5        # missing frames tolerated inside a flight
MIN_OBS = 6

cam = Camera()
f = cam.K[0, 0]
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
good = b.source.isin(['detected', 'rescued', 'manual']).values
has_box = b.x1.notna().values
uv_all = b[['center_x', 'center_y']].values
size_all = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)).values
T_all = b.time_sec.values


def size_point(i):
    d = cam.ray(uv_all[[i]])[0]
    return cam.C + f * BALL_D / size_all[i] * d


def fit_flight(idx, x0=None):
    t = T_all[idx] - T_all[idx[0]]
    uv, dist_size = uv_all[idx], f * BALL_D / size_all[idx]
    if x0 is None:
        x0 = initial_guess(cam, t, uv, depth=float(np.median(dist_size)))

    def res(x):
        P = simulate(x[:3], x[3:6], t)
        rp = (cam.project(P) - uv).ravel() / PX_SIGMA
        dist = np.linalg.norm(P - cam.C, axis=1)
        rs = (dist - dist_size) / (SIZE_SIGMA * dist_size)
        return np.r_[rp, rs]

    r = least_squares(res, x0, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=200)
    P = simulate(r.x[:3], r.x[3:6], t)
    return r.x, np.linalg.norm(cam.project(P) - uv, axis=1)


obs = np.flatnonzero(good)
flights = []  # (obs indices, params)
i = 0
while i < len(obs):
    # seed: MIN_OBS consecutive-ish observations
    j = i + MIN_OBS
    if j > len(obs) or np.any(np.diff(b.frame.values[obs[i:j]]) > MAX_GAP + 1):
        i += 1
        continue
    x, r = fit_flight(obs[i:j])
    if np.percentile(r, 90) > 3.0:
        i += 1
        continue
    while j < len(obs) and b.frame.values[obs[j]] - b.frame.values[obs[j - 1]] <= MAX_GAP + 1:
        x2, r2 = fit_flight(obs[i:j + 1], x)
        if r2[-1] > 5.0 or np.sqrt(np.mean(r2 ** 2)) > 2.5:
            break
        x, r, j = x2, r2, j + 1
    flights.append((obs[i:j], x))
    i = j
print(f'{len(flights)} flights covering {sum(len(o) for o, _ in flights)} of {len(obs)} detections')

n = len(b)
X = np.full((n, 3), np.nan)
method = np.array(['missing'] * n, dtype=object)
flight_id = np.full(n, -1)
resid = np.full(n, np.nan)
for k, (o, x) in enumerate(flights):
    fr = np.arange(o[0], o[-1] + 1)  # includes gaps/interpolated frames inside the flight
    X[fr] = simulate(x[:3], x[3:6], T_all[fr] - T_all[o[0]])
    method[fr] = 'flight_fit'
    flight_id[fr] = k
    resid[o] = np.linalg.norm(cam.project(X[o]) - uv_all[o], axis=1)

rest = np.flatnonzero(has_box & (method == 'missing'))
for i in rest:
    X[i] = size_point(i)
    method[i] = 'size_only'

h = -X[:, 2]
# Size-only points are noisy: smooth them with a short running median per run.
s = pd.Series(np.where(method == 'size_only', h, np.nan))
h_s = s.rolling(5, center=True, min_periods=1).median().values
h = np.where(method == 'size_only', h_s, h)
h = np.clip(h, 0, None)

out = pd.DataFrame({
    'frame': b.frame, 'time_sec': T_all,
    'height_m': np.round(h, 3),
    'court_x_m': np.round(X[:, 0], 3), 'court_y_m': np.round(X[:, 1], 3),
    'method': method, 'flight_id': flight_id,
    'reproj_err_px': np.round(resid, 2), 'box_source': b.source,
})
out.to_csv('data/analysis/ball_height_v2.csv', index=False)
print(out.method.value_counts())
print(out.groupby('method').height_m.describe())
print('reproj err px (fit obs):', np.nanpercentile(resid, [50, 90, 99]).round(2))
