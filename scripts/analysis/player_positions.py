"""Smoothed court position, velocity and speed of every player in every frame (v3 video).

Ground point per frame, by posture (Sapiens2 keypoints, see ground_points): standing -> ankle midpoint, or the planted
ankle when the other foot is up and swinging for a kick; lying on the sand -> hip midpoint; upside down (bicycle kick)
-> no measurement; sitting / kneeling or no usable pose -> bottom centre of the reviewed box. Occluded players and
boxes cut by the image border count less or not at all. Pixels -> court units with the homography.
Court: x along the length (0-18, net at 9), y across (0 = far sideline, 9 = near sideline, camera beyond).

Jumps that keep an upright pose (see airborne): a point above the sand projects further from the camera, so the
distance from the camera's ground point gets a short bump; bumps that running could not make are taken out of the
measurement, which is then trusted less along the camera ray.

Smoothing: per player and camera segment (same shot, no dead-time cut in between), a constant-velocity Kalman
filter + RTS smoother. Measurement noise is given in pixels and taken through the homography, so far-side positions
(fewer pixels per metre, mostly along y) count less. A measurement far from the prediction (NIS > GATE) is dropped
as an outlier (pose flicker, feet mixed up). After MAX_REJECT drops in a row the filter is reset in place: from the
first dropped frame, from a straight-line fit of the dropped measurements with a widened covariance, and one RTS pass
then runs over the whole stretch, joining the two sides. Gaps up to MAX_GAP frames are
bridged by the smoother; longer gaps split the track.

Units: the homography maps the court lines to 18 x 9. The EFVL rules of this event (Graz 2022) say 16 x 8 m; until
that is settled (docs/ANALYSIS_PLAN.md) lengths and speeds are in these court units (x 8/9 for metres if 16 x 8).

inputs : data/boxes/player_boxes_v3video.csv, data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv,
         data/boxes/ball_boxes_v3video.csv (camera shot of each frame), data/mappings/homography_px_to_m.npy,
         data/contacts/touch_labels_final.csv
output : data/analysis/player_positions_v3.csv  one row per (frame, player) from the first to the last frame of each
         track: posture, raw ground point, jump correction, smoothed x/y, vx/vy/speed (units/s), side, 3x3 square
usage: python scripts/analysis/player_positions.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
FPS = 25
DT = 1 / FPS
W, H_IMG = 1920, 1080
S_MIN = 0.3          # keypoint score
ANKLE_LIFT = 0.04    # box heights: the ankle joint sits this far above the sole
LIFT_FRAC = 0.5      # ankles further apart than this x leg length (image height): one foot may be up ...
PLANT_DT = 2         # ... and it is, if over +-PLANT_DT frames it moves more than PLANT_SWING box heights and
PLANT_RATIO = 2.0    #     PLANT_RATIO x the lower foot, which stays within STILL_FRAC box heights (pivoting allowed;
PLANT_SWING = 0.08   #     a lunge, a stride or two feet apart in depth do not count)
STILL_FRAC = 0.1
SHARED_PX = 5        # an ankle this close to a teammate's ankle (or to the other ankle) is not this player's own
ANKLE_BOX_MARGIN = 0.1   # box heights: an ankle outside the player's own box by more than this is dropped
UPRIGHT_FRAC = 0.5   # feet at least this x torso length below the hips: standing / running
LYING_WIDE = 1.2     # box wider than this x its height: lying on the sand
INVERT_FRAC = 0.3    # both feet more than this x torso length above the hips: upside down (bicycle kick)
TOUCH_WIN = 8        # frames around a player's own touch, when the player may be jumping
TOUCH_FAR = 15       # frames: a bump with no own touch this near needs AIR_MIN_FAR
OCC_IOU = 0.4        # box overlap with the teammate's box above this: the pose may belong to the other player
EDGE_PX = 3          # box within this of the left/right/bottom image border: cut by the border
EDGE_SHORT = 0.6     # ... and shorter than this x the player's usual box height: no measurement
SIGMA_PX = 4.0       # px, ground-point noise in the image (standing, both ankles)
SIGMA_ONE_ANKLE_PX = 12.0   # one ankle seen: it may be the swinging foot
SIGMA_LYING_PX = 10.0
SIGMA_BOX_PX = 8.0
SIGMA_OCC_PX = 20.0
SIGMA_EDGE_PX = 12.0
SIGMA_JUMP_PX = 40.0 # px, image height of the point near a touch (feet may be off the sand)
SIGMA_NEAR_LYING_PX = 20.0  # box bottom within LYING_NEAR frames of a lying / airborne pose (edge of the body)
LYING_NEAR = 5
Q_ACC = 30.0         # units^2/s^3, white-noise acceleration density (how fast a player may change velocity)
AIR_WIN = 31         # frames: bumps in the distance from the camera shorter than this can be jumps
AIR_MIN_M = 0.7      # units: smallest bump counted as a jump (stepping between the feet makes ~0.4) ...
AIR_MIN_FAR = 1.0    # ... or this, with no own touch within TOUCH_FAR frames
AIR_SUPPORT = 0.3    # units: an accepted bump is taken out wherever it stays above this
BUMP_TRUST = 0.3     # sigma along the camera ray of a corrected measurement = this x its bump
AIR_K = 3.0          # ... and at least this x the noise of that distance
AIR_FWHM_MAX = 15    # frames: a jump's bump is at most this wide at half height, upside-down frames not counted
LYING_BUMP_MIN = 1.5 # units: a lying player's bump counts only above this (horizontal bicycle kick, not a dive)
ACC_MAX = 12.0       # units/s^2: a bump that real running could not make (there and back) is a jump
GATE = 13.8          # chi2(2 dof) 0.999: farther than this from the prediction = outlier
MAX_REJECT = 4       # outliers in a row before the filter is reset on them
RESET_POS = 2.0      # units, extra position sigma at a reset
RESET_VEL = 0.5      # units/s, extra velocity sigma at a reset (small: most resets come from a hard acceleration)
SEED_V_MAX = 8.0     # units/s, cap on the velocity a reset starts from
LOOKAHEAD = 3        # measurements after a rejected run: if 2 of them are back on the old track, it was a glitch
MAX_GAP = 25         # frames without a measurement that the smoother may bridge (a bicycle kick lasts ~20)
V0_SIGMA = 3.0       # units/s, velocity uncertainty at a track start

H = np.load(os.path.join(ROOT, 'data', 'mappings', 'homography_px_to_m.npy'))
F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
Q = Q_ACC * np.array([[DT ** 3 / 3, 0, DT ** 2 / 2, 0], [0, DT ** 3 / 3, 0, DT ** 2 / 2],
                      [DT ** 2 / 2, 0, DT, 0], [0, DT ** 2 / 2, 0, DT]])
HM = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
RESET = np.diag([RESET_POS ** 2, RESET_POS ** 2, RESET_VEL ** 2, RESET_VEL ** 2])
CAM_GROUND = H[:2, 1] / H[2, 1]     # court point where all image verticals meet: the camera's foot on the court plane


def to_court(u, v):
    """pixels -> court units, plus the 2x2 Jacobian d(x, y)/d(u, v) of each point."""
    w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    X = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w
    Y = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w
    J = np.empty((len(u), 2, 2))
    J[:, 0, 0] = (H[0, 0] - X * H[2, 0]) / w
    J[:, 0, 1] = (H[0, 1] - X * H[2, 1]) / w
    J[:, 1, 0] = (H[1, 0] - Y * H[2, 0]) / w
    J[:, 1, 1] = (H[1, 1] - Y * H[2, 1]) / w
    return X, Y, J


def square(x, y):
    """3x3 square per half, as in scripts/court/court_squares.py (L1-L9 / R1-R9 from the far sideline)."""
    if not (0 <= x <= 18 and 0 <= y <= 9):
        return 'out'
    s = 'L' if x < 9 else 'R'
    c = min(int((x - (0 if s == 'L' else 9)) // 3), 2)
    r = min(int(y // 3), 2)
    return f'{s}{r * 3 + c + 1}'


def motion(b, x, y, scale):
    """largest move of a point over +-PLANT_DT frames of the same player (in units of scale), NaN if unseen."""
    out = np.zeros(len(b))
    g = b.groupby('player')
    for d in range(-PLANT_DT, PLANT_DT + 1):
        if d == 0:
            continue
        fx = g[x].shift(-d).values
        fy = g[y].shift(-d).values
        ok = (g.frame.shift(-d).values - b.frame.values) == d
        m = np.where(ok, np.hypot(fx - b[x].values, fy - b[y].values), 0)
        out = np.fmax(out, np.nan_to_num(m))
    return np.where(np.isfinite(b[x].values), out / scale, np.nan)


def ground_points(boxes):
    """one raw ground point (px) and its pixel noise per box row, by posture:
    stand  feet clearly below the hips: ankle midpoint (ankles lowered to the sole), or the planted ankle when the
           other foot is up and swinging (higher by more than LIFT_FRAC leg lengths, moving PLANT_RATIO x more than
           the planted one, which stays still); one ankle if only one is seen (counted less)
    lying  feet not below the hips and the box wider than tall (on the sand after a dive/fall): hip midpoint
    air    both feet more than INVERT_FRAC torso lengths above the hips (bicycle kick), or feet not below the hips at
           the player's own touch: no measurement, the smoother bridges it (in the air the body keeps its horizontal
           velocity)
    low    feet not below the hips otherwise (sitting, kneeling, crouched): bottom centre of the box
    box    no usable pose: bottom centre of the box
    edge   box cut by the left/right/bottom image border and much shorter than usual, or feet below the image: none
    hidden box bottom of an occluded player: none
    Ankle keypoints outside the player's own box, or on a teammate's ankle (within SHARED_PX), are dropped; two
    ankles on one point count as one. Near the player's own touches (+-TOUCH_WIN frames) the image height of the
    point counts little (sigma_v = SIGMA_JUMP_PX): ordinary jumps lift the feet without the pose looking any
    different. Occluded = the teammate's box overlaps (IoU > OCC_IOU) or stands in front of the feet (its bottom
    lower in the image, the player's box bottom and centre inside it). Occluded and border-cut rows count less, and
    neither feeds the jump detector nor seeds a reset."""
    kp = [f'{s}_{j}' for s in ('left', 'right') for j in ('ankle', 'hip', 'shoulder')]
    cols = ['frame', 'player'] + [f'{k}_{a}' for k in kp for a in 'xys']
    p = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'), usecols=cols)
    b = boxes.merge(p, on=['frame', 'player'], how='left').sort_values(['player', 'frame']).reset_index(drop=True)
    mates = {'BRA_A': 'BRA_B', 'BRA_B': 'BRA_A', 'ISR_A': 'ISR_B', 'ISR_B': 'ISR_A'}
    mc = ['x1', 'y1', 'x2', 'y2'] + [f'{s}_ankle_{a}' for s in ('left', 'right') for a in 'xys']
    m = b[['frame', 'player'] + mc].assign(player=lambda d: d.player.map(mates))
    o = b[['frame', 'player']].merge(m, on=['frame', 'player'], how='left')       # same row order as b
    n_rows = len(b)
    bh = (b.y2 - b.y1).values

    def pts(j, drop=0.0, d=b):
        x, y = [], []
        for s in ('left', 'right'):
            ok = d[f'{s}_{j}_s'].values >= S_MIN
            x.append(np.where(ok, d[f'{s}_{j}_x'].values, np.nan))
            y.append(np.where(ok, d[f'{s}_{j}_y'].values + drop, np.nan))
        return np.array(x), np.array(y)

    def mid(a):
        n = np.isfinite(a).sum(0)
        return np.where(n > 0, np.nansum(a, 0) / np.maximum(n, 1), np.nan)

    ax, ay = pts('ankle')
    mx_, my_ = pts('ankle', d=o)
    mg = ANKLE_BOX_MARGIN * bh
    outside = (ax < b.x1.values - mg) | (ax > b.x2.values + mg) | (ay < b.y1.values - mg) | (ay > b.y2.values + mg)
    shared = np.zeros_like(outside)
    for k in (0, 1):
        shared |= np.hypot(ax - mx_[k], ay - my_[k]) < SHARED_PX
    bad = outside | shared
    ax, ay = np.where(bad, np.nan, ax), np.where(bad, np.nan, ay)
    same = np.hypot(ax[0] - ax[1], ay[0] - ay[1]) < SHARED_PX       # both ankle keypoints on one foot
    ax[1], ay[1] = np.where(same, np.nan, ax[1]), np.where(same, np.nan, ay[1])
    ay = ay + ANKLE_LIFT * bh                                       # ankle joint -> sole
    hx, hy = pts('hip')
    sx, sy = pts('shoulder')
    n_ank = np.isfinite(ax).sum(0)
    hipx, hipy, shy, shx = mid(hx), mid(hy), mid(sy), mid(sx)
    torso = np.hypot(shx - hipx, shy - hipy)
    leg = mid(np.hypot(hx - ax, hy - ay))
    leg = np.where(np.isfinite(leg), leg, 0.45 * bh)
    u, v = mid(ax), mid(ay)

    # planted foot: the other one is higher in the image AND swinging while this one stays still
    for s, k in (('l', 0), ('r', 1)):
        b[f'_{s}x'], b[f'_{s}y'] = ax[k], ay[k]
    mv = np.array([motion(b, '_lx', '_ly', bh), motion(b, '_rx', '_ry', bh)])
    high = np.argmin(np.nan_to_num(ay, nan=np.inf), 0)            # the higher ankle in the image
    low = 1 - high
    r_ = np.arange(n_rows)
    lifted = (n_ank == 2) & (np.abs(ay[0] - ay[1]) > LIFT_FRAC * leg) & (mv[low, r_] < STILL_FRAC) \
        & (mv[high, r_] > np.maximum(PLANT_RATIO * mv[low, r_], PLANT_SWING))
    u = np.where(lifted, ax[low, r_], u)
    v = np.where(lifted, ay[low, r_], v)
    b = b.drop(columns=['_lx', '_ly', '_rx', '_ry'])

    t = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'touch_labels_final.csv'))
    t = t[t.kind == 'player']
    tdist = np.full(n_rows, np.inf)
    for pl, g in t.groupby('player'):
        sel = (b.player == pl).values
        tdist[sel] = np.abs(b.frame.values[sel][:, None] - g.frame.values[None]).min(1)
    near = tdist <= TOUCH_WIN
    b['near_touch'], b['touch_dist'] = near, tdist

    # teammate in the way: overlapping boxes, or the teammate's box in front of this player's feet
    iw = np.clip(np.minimum(b.x2, o.x2) - np.maximum(b.x1, o.x1), 0, None)
    ih = np.clip(np.minimum(b.y2, o.y2) - np.maximum(b.y1, o.y1), 0, None)
    inter = (iw * ih).fillna(0).values
    area = ((b.x2 - b.x1) * (b.y2 - b.y1)).values
    area_m = ((o.x2 - o.x1) * (o.y2 - o.y1)).fillna(0).values
    cx = ((b.x1 + b.x2) / 2).values
    front = (o.y2.values > b.y2.values) & (b.y2.values >= o.y1.values) & (cx >= o.x1.values) & (cx <= o.x2.values)
    occluded = (inter / np.maximum(area + area_m - inter, 1) > OCC_IOU) | front

    # image border
    cut = (b.x1.values <= EDGE_PX) | (b.x2.values >= W - 1 - EDGE_PX) | (b.y2.values >= H_IMG - 1 - EDGE_PX)
    usual = b.assign(h=np.where(cut, np.nan, bh)).groupby('player').h.transform(
        lambda s: s.rolling(51, center=True, min_periods=1).median()).fillna(pd.Series(bh))
    feet_out = (b.y2.values >= H_IMG - 1 - EDGE_PX) & (n_ank == 0)
    edge_drop = cut & ((bh < EDGE_SHORT * usual.values) | feet_out)

    feet_ref = np.where(lifted, v, np.nanmax(np.where(np.isfinite(ay), ay, -np.inf), 0))
    posed = np.isfinite(hipy) & np.isfinite(torso) & (n_ank > 0)
    upright = posed & (feet_ref - hipy > UPRIGHT_FRAC * torso)
    wide = (b.x2 - b.x1).values > LYING_WIDE * bh
    lying = posed & ~upright & wide
    inverted = (n_ank == 2) & (np.nanmax(ay, 0) < hipy - INVERT_FRAC * torso)
    air = posed & ~upright & ~wide & (inverted | near)
    low_ = posed & ~upright & ~wide & ~air                         # sitting / kneeling / crouched: box bottom
    upright, lying, air, low_ = (a & ~edge_drop for a in (upright, lying, air, low_))
    hidden = occluded & ~edge_drop & ~upright & ~lying & ~air      # a box bottom behind the teammate is not the feet
    low_ = low_ & ~hidden
    b['posture'] = np.select([edge_drop, hidden, upright, lying, air, low_],
                             ['edge', 'hidden', 'stand', 'lying', 'air', 'low'], 'box')
    b['src'] = np.select([edge_drop, hidden, upright & lifted, upright & (n_ank == 2), upright, lying, air],
                         ['edge', 'hidden', 'planted', 'ankles', 'ankle', 'hips', 'air'], 'box')
    gone = edge_drop | hidden
    b['u'] = np.select([gone, upright, lying, air], [np.nan, u, hipx, np.nan], b.feet_x.values)
    b['v'] = np.select([gone, upright, lying, air], [np.nan, v, hipy, np.nan], b.feet_y.values)
    src = b.src.values
    flat = pd.Series(lying | air).groupby(b.player).transform(
        lambda s: s.rolling(2 * LYING_NEAR + 1, center=True, min_periods=1).max()).values > 0
    su = np.select([src == 'ankle', lying, (src == 'box') & flat, src == 'box'],
                   [SIGMA_ONE_ANKLE_PX, SIGMA_LYING_PX, SIGMA_NEAR_LYING_PX, SIGMA_BOX_PX], SIGMA_PX)
    su = np.where(occluded, np.maximum(su, SIGMA_OCC_PX), su)
    su = np.where(cut, np.maximum(su, SIGMA_EDGE_PX), su)
    b['occluded'], b['edge'] = occluded, cut
    b['sigma_u'] = su
    b['sigma_v'] = np.where(near & ~lying, np.maximum(su, SIGMA_JUMP_PX), su)
    b['clean'] = ~occluded & ~cut                                 # may feed the jump detector and seed a reset
    return b


def _span(d, pk, level):
    """first and last index of the run around pk where d stays above level."""
    lo, hi = pk, pk
    while lo > 0 and d[lo - 1] > level:
        lo -= 1
    while hi < len(d) - 1 and d[hi + 1] > level:
        hi += 1
    return lo, hi


def airborne(b):
    """height of the jump bump (units, along the camera ray) of every measurement, 0 when the feet are on the sand.
    A point above the sand projects onto the court further from the camera, so in a jump (or the scissor phase of a
    bicycle kick, where the pose still looks upright) the distance rho from the camera's ground point gets a bump that
    only goes up and comes back down at landing. The detector uses clean rows only (not occluded, not cut by the
    border; standing, lying or crouched, not the flickering box bottom). A morphological opening (rolling min, then
    rolling max, over AIR_WIN frames; edges padded, gaps interpolated) of the 5-frame median keeps slower movement and
    removes shorter bumps. A run more than max(AIR_MIN_M, AIR_K sigma_rho) above the opening (AIR_MIN_FAR with no own
    touch within TOUCH_FAR frames) is a jump only if its bump is at most AIR_FWHM_MAX frames wide at half height, not
    counting frames upside down (airborne anyway, e.g. in a bicycle kick), and running there and back would need more
    than ACC_MAX (about 2 pi^2 A / T^2 for a bump of height A lasting T = twice that width); otherwise it is taken as
    real movement. An accepted bump is taken out of every measurement (occluded and border rows too) wherever it
    stays above AIR_SUPPORT (median or raw), by that row's own excess over the opening (0..A); of a lying row only
    above LYING_BUMP_MIN."""
    use = b.clean.values & b.posture.isin(['stand', 'lying', 'low']).values
    rho_all = np.hypot(b.raw_x.values - CAM_GROUND[0], b.raw_y.values - CAM_GROUND[1])
    rho = np.where(use, rho_all, np.nan)
    bump = np.zeros(len(b))
    half = AIR_WIN // 2
    w = 2 * half + 1
    lying = (b.posture == 'lying').values
    measured = np.isfinite(rho_all)
    in_air = (b.posture == 'air').values
    far = b.touch_dist.values > TOUCH_FAR
    for _, g in b.groupby(['player', 'cam_seg']):
        gi = g.index.values
        ok = np.isfinite(rho[gi])
        if ok.sum() < 3:
            continue
        i, fr = gi[ok], g.frame.values[ok]
        idx = np.arange(fr[0], fr[-1] + 1)
        r = pd.Series(rho[i], index=fr).reindex(idx)
        air_f = pd.Series(in_air[gi], index=g.frame.values).reindex(idx, fill_value=False).values
        far_f = pd.Series(far[gi], index=g.frame.values).reindex(idx, fill_value=True).values
        ri = r.interpolate()
        med = ri.rolling(5, center=True, min_periods=1).median()
        pad = pd.Series(np.r_[np.full(half, med.iloc[0]), med.values, np.full(half, med.iloc[-1])])
        base = pad.rolling(w, center=True, min_periods=1).min().rolling(w, center=True, min_periods=1).max()
        base = base.values[half:len(pad) - half]
        d = np.clip(med.values - base, 0, None)
        sig = pd.Series(b.sigma_rho.values[i], index=fr).reindex(idx).interpolate().values
        cand = d > np.maximum(np.where(far_f, AIR_MIN_FAR, AIR_MIN_M), AIR_K * sig)
        top = np.zeros(len(d))                      # height of the accepted bump each frame belongs to
        edges = np.flatnonzero(np.diff(np.r_[0, cand.astype(int), 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            pk = a + int(np.argmax(d[a:e]))
            A = d[pk]
            lo, hi = _span(d, pk, A / 2)
            width = (hi - lo + 1) - air_f[lo:hi + 1].sum()   # frames upside down are airborne anyway
            T = 2 * max(width, 1) / FPS                  # a raised-cosine bump lasts twice its width at half height
            if width <= AIR_FWHM_MAX and 2 * np.pi ** 2 * A / T ** 2 > ACC_MAX:
                lo, hi = _span(np.maximum(d, ri.values - base), pk, AIR_SUPPORT)   # the median lags at take-off
                top[lo:hi + 1] = np.maximum(top[lo:hi + 1], A)
        rows = gi[measured[gi]]
        f = b.frame.values[rows]
        inside = (f >= fr[0]) & (f <= fr[-1])
        rows, f = rows[inside], f[inside]
        A_row = top[f - fr[0]]
        ex = np.where(A_row > 0, np.clip(rho_all[rows] - base[f - fr[0]], 0, A_row), 0)
        bump[rows] = np.where(lying[rows] & (ex < LYING_BUMP_MIN), 0, ex)
    return bump


def _back_on_track(z, R, x, P, k, gate):
    """do at least 2 of the next LOOKAHEAD measurements after frame k fit the old track (state x, P at k, carried
    forward with no update)?"""
    hits = seen = 0
    j = k + 1
    while j < len(z) and seen < LOOKAHEAD:
        x, P = F @ x, F @ P @ F.T + Q
        if np.isfinite(z[j, 0]):
            y = z[j] - x[:2]
            seen += 1
            hits += y @ np.linalg.inv(P[:2, :2] + R[j]) @ y <= gate
        j += 1
    return hits >= 2


def kalman_rts(z, R, clean, gate=GATE):
    """z: (n, 2) measurements (NaN = none, z[0] given), R: (n, 2, 2), clean: (n,) measurements allowed to seed a
    reset. Returns smoothed states (n, 4), outlier flags and reset flags. On MAX_REJECT outliers in a row the forward
    pass goes back to the first of them and restarts there from a straight-line fit (position and velocity, speed
    capped at SEED_V_MAX) of the rejected (clean) measurements, with the covariance widened by RESET - unless the
    measurements right after the run are back on the old track (a glitch: the run stays rejected). The RTS pass runs over everything and uses
    the prediction covariance without that widening, so it joins the two sides of a reset smoothly."""
    n = len(z)
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    Pq = np.zeros((n, 4, 4))
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))
    out, reset = np.zeros(n, bool), np.zeros(n, bool)
    xf[0] = xp[0] = np.r_[z[0], 0, 0]
    Pf[0] = Pp[0] = Pq[0] = np.block([[R[0], np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2) * V0_SIGMA ** 2]])
    rejected, first, seed_at, seed, seed_v = 0, 0, -1, None, None
    k = 1
    while k < n:
        x, P = F @ xf[k - 1], F @ Pf[k - 1] @ F.T + Q
        Pq[k] = P
        if k == seed_at:
            x = np.r_[x[:2], seed_v]
            P = P + RESET
        xp[k], Pp[k] = x, P
        if np.isfinite(z[k, 0]):
            zk = seed if k == seed_at else z[k]
            y = zk - x[:2]
            Si = np.linalg.inv(P[:2, :2] + R[k])
            if k != seed_at and y @ Si @ y > gate:
                out[k] = True
                if not rejected:
                    first = k
                rejected += 1
                if rejected >= MAX_REJECT and _back_on_track(z, R, x, P, k, gate):
                    rejected = 0                         # a glitch: the old track goes on after it
                elif rejected >= MAX_REJECT:
                    idx = [j for j in range(first, k + 1) if out[j]]
                    use = np.array([j for j in idx if clean[j]] or idx)
                    if len(use) >= 2:
                        cf = np.polyfit(use - first, z[use], 1)       # rows: slope, value at `first`
                        seed, seed_v = cf[1], cf[0] * FPS
                        sp = np.hypot(*seed_v)
                        if sp > SEED_V_MAX:
                            seed_v = seed_v * SEED_V_MAX / sp
                    else:
                        seed, seed_v = z[use[0]], xf[k - 1][2:]
                    out[first:k + 1] = False
                    reset[first] = True
                    seed_at, k, rejected = first, first, 0
                    continue
            else:
                rejected = 0
                K = P @ HM.T @ Si
                x = x + K @ y
                P = (np.eye(4) - K @ HM) @ P
        xf[k], Pf[k] = x, P
        k += 1
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pq[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - F @ xf[k])
    return xs, out, reset


def smooth(sub):
    """smooth one stretch of a track (frames may have holes <= MAX_GAP), starting and ending on a measurement."""
    f = np.arange(sub.frame.iloc[0], sub.frame.iloc[-1] + 1)
    idx = np.searchsorted(f, sub.frame.values)
    z = np.full((len(f), 2), np.nan); z[idx] = sub[['raw_x', 'raw_y']].values
    R = np.repeat(np.eye(2)[None] * 1e6, len(f), 0); R[idx] = np.stack(sub.R.values)
    clean = np.zeros(len(f), bool); clean[idx] = sub.clean.values
    if len(f) == 1:
        return pd.DataFrame({'frame': f, 'x': z[:, 0], 'y': z[:, 1], 'vx': np.nan, 'vy': np.nan, 'outlier': False,
                             'reset': False})
    xs, out, reset = kalman_rts(z, R, clean)
    return pd.DataFrame({'frame': f, 'x': xs[:, 0], 'y': xs[:, 1], 'vx': xs[:, 2], 'vy': xs[:, 3], 'outlier': out,
                         'reset': reset})


def smooth_track(g):
    """g: one player's measured rows in one camera segment, sorted by frame; gaps longer than MAX_GAP split it."""
    fr = g.frame.values
    bounds = np.r_[0, np.nonzero(np.diff(fr) > MAX_GAP + 1)[0] + 1, len(fr)]
    return pd.concat([smooth(g.iloc[a:e]) for a, e in zip(bounds[:-1], bounds[1:])])


def camera_segments(ball):
    """segment id per frame: a new one at every shot change or removed dead time; one-frame segments that continue
    the previous frame (v2 numbering) are merged into it (a shot label one frame early or late)."""
    cut = (ball.shot.diff() != 0) | (ball.v2_frame.diff() != 1)
    seg = cut.cumsum() - 1
    size = seg.map(seg.value_counts())
    lone = (size == 1) & (ball.v2_frame.diff() == 1)
    seg = seg.where(~lone, seg.shift(1))
    return pd.Series(pd.factorize(seg)[0], index=ball.index)


def main():
    os.makedirs(OUT, exist_ok=True)
    boxes = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'player_boxes_v3video.csv'),
                        usecols=['frame', 'v2_frame', 'player', 'team', 'x1', 'y1', 'x2', 'y2', 'feet_x', 'feet_y',
                                 'source'])
    ball = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_v3video.csv'), usecols=['frame', 'v2_frame', 'shot'])
    ball['cam_seg'] = camera_segments(ball)
    b = ground_points(boxes)
    b = b.merge(ball[['frame', 'cam_seg']], on='frame').sort_values(['player', 'frame']).reset_index(drop=True)
    x, y, J = to_court(b.u.values, b.v.values)
    b['raw_x'], b['raw_y'] = x, y
    dv = np.linalg.norm(J[:, :, 1], axis=1)                  # units per image pixel along the image vertical
    b['sigma_rho'] = SIGMA_PX * dv
    b['bump'] = airborne(b)
    b['jump'] = b.bump > 0
    up = -J[:, :, 1] / dv[:, None]                           # where a point above the sand projects to
    j = b.jump.values
    b.loc[j, 'raw_x'] -= b.bump[j] * up[j, 0]                # take the jump out of the measurement ...
    b.loc[j, 'raw_y'] -= b.bump[j] * up[j, 1]
    Sig = np.zeros((len(b), 2, 2)); Sig[:, 0, 0] = b.sigma_u ** 2; Sig[:, 1, 1] = b.sigma_v ** 2
    R = J @ Sig @ J.transpose(0, 2, 1)
    R[j] += ((BUMP_TRUST * b.bump.values[j]) ** 2)[:, None, None] * up[j, :, None] * up[j, None, :]   # ... a bit less
    b['R'] = list(R)
    res = []
    for (pl, seg), g in b[np.isfinite(b.raw_x)].groupby(['player', 'cam_seg']):
        res.append(smooth_track(g).assign(player=pl, cam_seg=seg))
    s = pd.concat(res)
    s['track'] = ((s.player != s.player.shift()) | (s.cam_seg != s.cam_seg.shift()) | (s.frame.diff() != 1)).cumsum()
    s = s.merge(b[['frame', 'player', 'posture', 'src', 'near_touch', 'occluded', 'edge', 'jump', 'bump', 'u', 'v',
                   'raw_x', 'raw_y']], on=['frame', 'player'], how='left')
    s['src'] = s.src.fillna('none')
    s['posture'] = s.posture.fillna('none')
    for c in ('near_touch', 'occluded', 'edge', 'jump'):
        s[c] = s[c].fillna(False).astype(bool)
    s['airborne'] = s.jump | (s.posture == 'air')
    s = s.merge(ball[['frame', 'v2_frame']], on='frame')
    s['time_sec'] = (s.frame / FPS).round(2)
    s['team'] = s.player.str[:3]
    s['speed'] = np.hypot(s.vx, s.vy)
    s['side'] = np.where(s.x < 9, 'LEFT', 'RIGHT')
    s['square'] = [square(a, c) for a, c in zip(s.x, s.y)]
    cols = ['frame', 'v2_frame', 'time_sec', 'player', 'team', 'track', 'posture', 'src', 'near_touch', 'occluded',
            'edge', 'u', 'v', 'raw_x', 'raw_y', 'jump', 'bump', 'airborne', 'outlier', 'reset',
            'x', 'y', 'vx', 'vy', 'speed', 'side', 'square']
    s = s[cols].sort_values(['frame', 'player'])
    for c in ('raw_x', 'raw_y', 'bump', 'x', 'y', 'vx', 'vy', 'speed'):
        s[c] = s[c].round(3)
    s['u'], s['v'] = s.u.round(1), s.v.round(1)
    s.to_csv(os.path.join(OUT, 'player_positions_v3.csv'), index=False)
    m = s[np.isfinite(s.raw_x)]
    print(f'{len(s)} rows ({len(s) - len(m)} bridged), {s.track.nunique()} tracks, outliers {m.outlier.mean():.1%}, '
          f'resets {int(s.reset.sum())}, jump frames {int(s.jump.sum())}')
    print('postures:', s.posture.value_counts().to_dict())
    print('speed quantiles:', s.speed.quantile([.5, .9, .99, .999]).round(2).to_dict())


if __name__ == '__main__':
    main()
