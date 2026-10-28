"""EXPERIMENT (chest marks): like scripts/experiments/court_scale/estimate.py, but the chest touches use the
hand-marked frame and a fixed 3-D point: the ball pixel put at the touching player's horizontal position
(data/experiments/chest_marks/marks). MARK_USE=even|odd|all picks which marked touches become anchors, so the
others can test the result. Camera from $CAM, statures from $STATURES, outputs in $OUT.

Ball height above the sand (ground = 0) for every frame.

Pipeline:
  1. Cut the 2-D ball track at every ball event of the project: the labelled
     touches (data/contacts/touch_labels_final.csv) and the other breaks of
     data/contacts/ball_contacts_v3.csv (sand, net, tracking). No flight
     crosses an event.
  2. Between two events a flight grows frame by frame while one ballistic
     3-D trajectory still reprojects within a few px.
  3. Each flight is fitted with gravity, a weak prior from the ball's
     apparent size (distance = f * D / size) and anchors at the events it
     starts or ends on: at a touch the ball is at the touching body part of
     that player (court position from data/analysis/player_positions_v3.csv,
     height from the body part and the player's stature), on the sand it is
     one radius up.
  4. Frames outside any flight fall back to the size-only depth.
Env HOLDOUT=<body part> leaves that part's anchors out (to test the result
against those touches). Outputs data/analysis/ball_height_v2.csv.
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj import Camera, simulate, initial_guess, G
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)  # all paths below are relative to the repo root


BALL_D = 0.22      # ball diameter (given)
SIZE_SIGMA = 0.12  # relative noise of a single size-based distance
PX_SIGMA = 1.5
MAX_GAP = 5        # missing frames tolerated inside a flight
MIN_OBS = 6

OUTD = os.environ['OUT']
os.makedirs(OUTD, exist_ok=True)
cam = Camera(os.environ['CAM'])
MARK_USE = os.environ.get('MARK_USE', 'all')
ANCHORS = os.environ.get('ANCHORS', 'all')  # 'sand': only sand landings (no body-part anchors)
SAND_USE = os.environ.get('SAND_USE', 'all')  # even|odd|all: sand landings by frame parity (hold-out)
f = cam.K[0, 0]
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
good = b.source.isin(['detected', 'rescued', 'manual']).values
has_box = b.x1.notna().values
uv_all = b[['center_x', 'center_y']].values
size_all = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)).values
T_all = b.time_sec.values

# --- ball events (v3 frames -> v2) and anchors ---
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
STATURE = {k: v['stature_m'] for k, v in json.load(open(os.environ['STATURES'])).items()}
# body part: (height / stature, height sigma m, horizontal sigma m); foot height is absolute (m)
PART = {'chest': (0.72, 0.12, 0.35), 'head': (0.95, 0.25, 0.35), 'shoulder': (0.82, 0.15, 0.4),
        'arm': (0.75, 0.35, 0.6), 'thigh': (0.45, 0.15, 0.45), 'knee': (0.29, 0.15, 0.45),
        'shin': (0.2, 0.15, 0.5), 'foot': (None, 0.6, 0.7)}
FOOT_H = 0.4
SAND_H, SAND_SIG = BALL_D / 2, 0.05
HOLDOUT = os.environ.get('HOLDOUT', '')
Hm = np.load('data/mappings/homography_px_to_m.npy')
Hi = np.linalg.inv(Hm)
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y'])
pos = pos.set_index(['v2_frame', 'player'])


def player_xy(fr, pl):
    """player's ground point at a v2 frame, in this camera's world (m)."""
    try:
        x, y = pos.loc[(fr, pl)].values.ravel()[:2]
    except KeyError:
        return None
    q = Hi @ np.array([x, y, 1.0])
    return cam.ground(np.array([q[:2] / q[2]]))[0][:2]


import glob
MARKS = {}
for fn in glob.glob('data/experiments/chest_marks/marks/marks/*.json'):
    m = json.load(open(fn))
    if m.get('chest') and m.get('feet') == 'both' and 'ball_x' in m:
        MARKS[int(m['touch'])] = m
use = set() if MARK_USE == 'none' else {k for k in MARKS if MARK_USE == 'all' or (k % 2 == 0) == (MARK_USE == 'even')}


def marked_point(m):
    xy = player_xy(int(m['frame']), m['player'])
    if xy is None:
        return None
    d = cam.ray(np.array([[m['ball_x'], m['ball_y']]]))[0]
    s_ = np.dot(xy - cam.C[:2], d[:2]) / np.dot(d[:2], d[:2])
    P = cam.C + s_ * d
    return (P[:2], float(-P[2]), 0.06, 0.15)


events = {}  # v2 frame -> anchor or None
for k in use:
    a = marked_point(MARKS[k])
    if a is not None:
        events[int(MARKS[k]['frame'])] = a
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
for r in tl.itertuples():
    fr = int(fm.get(int(r.frame), -1))
    if fr < 0 or int(r.touch) in MARKS:  # marked chest touches: handled above (anchor) or left free (test)
        continue
    anc = None
    if ANCHORS == 'all' and r.kind == 'player' and isinstance(r.body_part, str) and r.body_part in PART and r.body_part != HOLDOUT:
        xy = player_xy(fr, r.player)
        ratio, sh, sxy = PART[r.body_part]
        if xy is not None:
            anc = (xy, FOOT_H if ratio is None else ratio * STATURE.get(r.player, 1.8), sh, sxy)
    elif r.kind == 'sand':
        anc = (None, SAND_H, SAND_SIG, None)
    events[fr] = anc
bc = pd.read_csv('data/contacts/ball_contacts_v3.csv', usecols=['frame', 'event_type', 'ball_on_sand'])
for r in bc.itertuples():
    fr = int(fm.get(int(r.frame), -1))
    if fr < 0 or fr in events:
        continue
    sand_ok = SAND_USE == 'all' or (fr % 2 == 0) == (SAND_USE == 'even')
    events[fr] = (None, SAND_H, SAND_SIG, None) if (r.event_type != 'touch' and r.ball_on_sand is True and sand_ok) else None
ev_frames = np.array(sorted(events))
print(f'{len(ev_frames)} ball events, {sum(v is not None for v in events.values())} with an anchor')


def size_point(i):
    d = cam.ray(uv_all[[i]])[0]
    return cam.C + f * BALL_D / size_all[i] * d


def fit_flight(idx, x0=None, anchors=()):
    """anchors: (frame, (xy or None, height, sigma_h, sigma_xy))"""
    t = T_all[idx] - T_all[idx[0]]
    uv, dist_size = uv_all[idx], f * BALL_D / size_all[idx]
    if x0 is None:
        x0 = initial_guess(cam, t, uv, depth=float(np.median(dist_size)))
    ta = np.array([T_all[fr] - T_all[idx[0]] for fr, _ in anchors])

    def res(x):
        P = simulate(x[:3], x[3:6], t)
        rp = (cam.project(P) - uv).ravel() / PX_SIGMA
        dist = np.linalg.norm(P - cam.C, axis=1)
        rs = (dist - dist_size) / (SIZE_SIGMA * dist_size)
        ra = []
        if len(anchors):
            A = simulate(x[:3], x[3:6], ta)
            for (fr, (xy, hh, sh, sxy)), Pa in zip(anchors, A):
                ra.append((-Pa[2] - hh) / sh)
                if xy is not None:
                    ra.extend((Pa[:2] - xy) / sxy)
        return np.r_[rp, rs, ra]

    r = least_squares(res, x0, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=200)
    P = simulate(r.x[:3], r.x[3:6], t)
    return r.x, np.linalg.norm(cam.project(P) - uv, axis=1)


def anchors_for(o):
    """events a flight starts or ends on (within 2 frames of its first / last detection)."""
    out = []
    for fr in (o[0], o[-1]):
        k = np.searchsorted(ev_frames, fr)
        for e in ev_frames[max(0, k - 1):k + 1]:
            if abs(e - fr) <= 2 and events[e] is not None and all(e != a for a, _ in out):
                out.append((int(e), events[e]))
    return out


all_obs = np.flatnonzero(good)
piece = np.searchsorted(ev_frames, all_obs, side='right')  # flights never cross an event
flights = []  # (obs indices, params, anchors)
anchor_log = []
for pc in np.unique(piece):
  obs = all_obs[piece == pc]
  # the event that ends this piece also starts the next one: the ball is there for both flights
  if pc < len(ev_frames) and good[ev_frames[pc]]:
      obs = np.r_[obs, ev_frames[pc]]
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
    anc = anchors_for(obs[i:j])
    if anc:
        x0 = x
        x, r = fit_flight(obs[i:j], None, anc)  # fresh start: the unanchored solution can sit in the wrong depth
        xb, rb = fit_flight(obs[i:j], x0, anc)
        if np.sum(rb ** 2) < np.sum(r ** 2):
            x, r = xb, rb
        for fr, (xy, hh, sh, sxy) in anc:
            Pa = simulate(x[:3], x[3:6], np.array([T_all[fr] - T_all[obs[i]]]))[0]
            anchor_log.append(dict(frame=fr, target_h=round(hh, 2), fit_h=round(-Pa[2], 2),
                                   dxy=None if xy is None else round(float(np.hypot(*(Pa[:2] - xy))), 2),
                                   reproj_p90=round(float(np.percentile(r, 90)), 2), n_obs=j - i))
    flights.append((obs[i:j], x, anc))
    i = j
print(f'{len(flights)} flights covering {sum(len(o) for o, _, _ in flights)} of {len(all_obs)} detections, '
      f'{sum(bool(a) for _, _, a in flights)} anchored')

n = len(b)
X = np.full((n, 3), np.nan)
method = np.array(['missing'] * n, dtype=object)
flight_id = np.full(n, -1)
resid = np.full(n, np.nan)
for k, (o, x, anc) in enumerate(flights):
    lo = min([o[0]] + [a for a, _ in anc]); hi = max([o[-1]] + [a for a, _ in anc])
    fr = np.arange(lo, hi + 1)  # includes gaps/interpolated frames inside the flight and its anchor frames
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
out.to_csv(os.path.join(OUTD, 'ball_height_v2.csv'), index=False)
pd.DataFrame(anchor_log).to_csv(os.path.join(OUTD, 'anchors.csv'), index=False)  # anchor fit check
print(out.method.value_counts())
print(out.groupby('method').height_m.describe())
print('reproj err px (fit obs):', np.nanpercentile(resid, [50, 90, 99]).round(2))
