"""Smoothed court position, velocity and speed of every player in every frame (v3 video).

Ground point per frame, by posture (Sapiens2 keypoints, see ground_points): standing -> ankle midpoint, or the planted
ankle when the other foot is up for a kick; lying on the sand -> hip midpoint; in the air with the feet above the hips
(bicycle kick) -> no measurement; no usable pose -> bottom centre of the reviewed box. Pixels -> court metres with the
homography. Court: x along the length (0-18, net at 9), y across (0 = far sideline, 9 = near sideline, camera beyond).

Jumps that keep an upright pose (see airborne): a point above the sand projects further from the camera, so the
distance from the camera's ground point gets a short bump; bumps that running could not make are taken out of the
measurement, which is then trusted less along the camera ray.

Smoothing: per player and camera segment (same shot, no dead-time cut in between), a constant-velocity Kalman
filter + RTS smoother. Measurement noise is given in pixels and taken through the homography, so far-side positions
(fewer pixels per metre, mostly along y) count less. A measurement far from the prediction (NIS > GATE) is dropped
as an outlier (pose flicker, feet mixed up); after MAX_REJECT drops in a row the track restarts on the measurements.
Gaps up to MAX_GAP frames are bridged by the smoother; longer gaps split the track.

Units: the homography maps the court lines to 18 x 9. The EFVL rules of this event (Graz 2022) say 16 x 8 m; until
that is settled (docs/ANALYSIS_PLAN.md) lengths and speeds are in these court units (x 8/9 for metres if 16 x 8).

inputs : data/boxes/player_boxes_v3video.csv, data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv,
         data/boxes/ball_boxes_v3video.csv (camera shot of each frame), data/mappings/homography_px_to_m.npy
output : data/analysis/player_positions_v3.csv  one row per (frame, player) from the first to the last frame of each
         track: raw ground point, smoothed x/y (m), vx/vy/speed (m/s), side, 3x3 square
usage: python scripts/analysis/player_positions.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
FPS = 25
DT = 1 / FPS
S_MIN = 0.3          # ankle keypoint score
LIFT_FRAC = 0.5      # ankles further apart than this x leg length (image height): one foot is in the air
UPRIGHT_FRAC = 0.5   # feet at least this x torso length below the hips: standing / running
LYING_WIDE = 1.2     # box wider than this x its height: lying on the sand
INVERT_FRAC = 0.3    # feet more than this x torso length above the hips: upside down (bicycle kick)
TOUCH_WIN = 8        # frames around a player's own touch, when the player may be jumping
SIGMA_PX = 4.0       # px, ground-point noise in the image (standing)
SIGMA_LYING_PX = 10.0
SIGMA_BOX_PX = 8.0
SIGMA_JUMP_PX = 40.0 # px, image height of the point near a touch (feet may be off the sand)
Q_ACC = 30.0         # m^2/s^3, white-noise acceleration density (how fast a player may change velocity)
AIR_WIN = 31         # frames: bumps in the distance from the camera shorter than this can be jumps
AIR_MIN_M = 0.35     # m: smallest bump counted as a jump
AIR_K = 3.0          # ... and at least this x the noise of that distance
ACC_MAX = 12.0       # m/s^2: a bump that real running could not make (there and back) is a jump
BUMP_EDGE = 0.1      # a jump is taken out where its bump is above this fraction of its height
GATE = 13.8          # chi2(2 dof) 0.999: farther than this from the prediction = outlier
MAX_REJECT = 4       # outliers in a row before the track restarts on the measurements
MAX_GAP = 25         # frames without a measurement that the smoother may bridge (a bicycle kick lasts ~20)
V0_SIGMA = 3.0       # m/s, velocity uncertainty at a track start

H = np.load(os.path.join(ROOT, 'data', 'mappings', 'homography_px_to_m.npy'))
F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
Q = Q_ACC * np.array([[DT ** 3 / 3, 0, DT ** 2 / 2, 0], [0, DT ** 3 / 3, 0, DT ** 2 / 2],
                      [DT ** 2 / 2, 0, DT, 0], [0, DT ** 2 / 2, 0, DT]])
HM = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
CAM_GROUND = H[:2, 1] / H[2, 1]     # court point where all image verticals meet: the camera's foot on the court plane


def to_court(u, v):
    """pixels -> court metres, plus the 2x2 Jacobian d(x, y)/d(u, v) of each point."""
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


def ground_points(boxes):
    """one raw ground point (px) and its pixel noise per box row, by posture:
    stand  feet clearly below the hips: ankle midpoint, or the planted ankle when the other foot is lifted for a kick
           (ankles more than LIFT_FRAC leg lengths apart in image height); one ankle if only one is seen
    lying  feet not below the hips and the box wider than tall (on the sand after a dive/fall): hip midpoint
    air    feet more than INVERT_FRAC torso lengths above the hips (bicycle kick), or feet not below the hips at the
           player's own touch: no measurement, the smoother bridges it (in the air the body keeps its horizontal
           velocity)
    low    feet not below the hips otherwise (sitting, kneeling, crouched): bottom centre of the box
    box    no usable pose: bottom centre of the box
    Near the player's own touches (+-TOUCH_WIN frames) the image height of the point counts little (sigma_v =
    SIGMA_JUMP_PX): ordinary jumps lift the feet without the pose looking any different."""
    kp = [f'{s}_{j}' for s in ('left', 'right') for j in ('ankle', 'hip', 'shoulder')]
    cols = ['frame', 'player'] + [f'{k}_{a}' for k in kp for a in 'xys']
    p = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'), usecols=cols)
    b = boxes.merge(p, on=['frame', 'player'], how='left')
    n_rows = len(b)

    def pts(j):
        x, y = [], []
        for s in ('left', 'right'):
            ok = b[f'{s}_{j}_s'].values >= S_MIN
            x.append(np.where(ok, b[f'{s}_{j}_x'].values, np.nan))
            y.append(np.where(ok, b[f'{s}_{j}_y'].values, np.nan))
        return np.array(x), np.array(y)

    def mid(a):
        n = np.isfinite(a).sum(0)
        return np.where(n > 0, np.nansum(a, 0) / np.maximum(n, 1), np.nan)

    ax, ay = pts('ankle')
    hx, hy = pts('hip')
    sx, sy = pts('shoulder')
    n_ank = np.isfinite(ax).sum(0)
    hipx, hipy, shy, shx = mid(hx), mid(hy), mid(sy), mid(sx)
    torso = np.hypot(shx - hipx, shy - hipy)
    leg = mid(np.hypot(hx - ax, hy - ay))
    leg = np.where(np.isfinite(leg), leg, 0.45 * (b.y2 - b.y1).values)
    u, v = mid(ax), mid(ay)
    lifted = (n_ank == 2) & (np.abs(ay[0] - ay[1]) > LIFT_FRAC * leg)
    low = np.argmax(np.nan_to_num(ay, nan=-1), 0)                  # the lower ankle in the image stands on the sand
    u = np.where(lifted, ax[low, np.arange(n_rows)], u)
    v = np.where(lifted, ay[low, np.arange(n_rows)], v)
    feet_ref = np.where(lifted, v, np.nanmax(np.where(np.isfinite(ay), ay, -np.inf), 0))
    t = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'touch_labels_final.csv'))
    t = t[t.kind == 'player']
    near = np.zeros(n_rows, bool)
    for pl, g in t.groupby('player'):
        m = (b.player == pl).values
        f = b.frame.values[m]
        d = np.abs(f[:, None] - g.frame.values[None]).min(1)
        near[np.nonzero(m)[0][d <= TOUCH_WIN]] = True
    b['near_touch'] = near
    posed = np.isfinite(hipy) & np.isfinite(torso) & (n_ank > 0)
    upright = posed & (feet_ref - hipy > UPRIGHT_FRAC * torso)
    wide = (b.x2 - b.x1).values > LYING_WIDE * (b.y2 - b.y1).values
    lying = posed & ~upright & wide
    inverted = feet_ref < hipy - INVERT_FRAC * torso
    air = posed & ~upright & ~wide & (inverted | near)
    low = posed & ~upright & ~wide & ~air                         # sitting / kneeling / crouched: box bottom
    b['posture'] = np.select([upright, lying, air, low], ['stand', 'lying', 'air', 'low'], 'box')
    b['src'] = np.select([upright & lifted, upright & (n_ank == 2), upright, lying, air],
                         ['planted', 'ankles', 'ankle', 'hips', 'air'], 'box')
    b['u'] = np.select([upright, lying, air], [u, hipx, np.nan], b.feet_x.values)
    b['v'] = np.select([upright, lying, air], [v, hipy, np.nan], b.feet_y.values)
    su = np.select([lying, b.src.values == 'box'], [SIGMA_LYING_PX, SIGMA_BOX_PX], SIGMA_PX)
    b['sigma_u'] = su
    b['sigma_v'] = np.where(near & (b.posture.values != 'lying'), SIGMA_JUMP_PX, su)
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
    """height of the jump bump (m, along the camera ray) of every measurement, 0 when the feet are on the sand.
    A point above the sand projects onto the court further from the camera, so in a jump (or the scissor phase of a
    bicycle kick, where the pose still looks upright) the distance rho from the camera's ground point gets a bump that
    only goes up and comes back down at landing. A morphological opening (rolling min, then rolling max, over AIR_WIN
    frames; edges padded, gaps interpolated) keeps slower movement and removes shorter bumps. A run of frames more than
    max(AIR_MIN_M, AIR_K sigma_rho) above the opening is a jump only if running there and back would need more than
    ACC_MAX (about 2 pi^2 A / T^2 for a bump of height A lasting T = twice its width at half height); otherwise it
    is taken as real movement. A jump is taken out over the whole bump (down to BUMP_EDGE of its height)."""
    rho = np.hypot(b.raw_x.values - CAM_GROUND[0], b.raw_y.values - CAM_GROUND[1])
    bump = np.zeros(len(b))
    half = AIR_WIN // 2
    for _, g in b.groupby(['player', 'cam_seg']):
        i = g.index.values
        ok = np.isfinite(rho[i])
        if ok.sum() < 3:
            continue
        i, fr = i[ok], g.frame.values[ok]
        r = pd.Series(rho[i], index=fr).reindex(np.arange(fr[0], fr[-1] + 1)).interpolate()
        med = r.rolling(5, center=True, min_periods=1).median()
        pad = pd.Series(np.r_[np.full(half, med.iloc[0]), med.values, np.full(half, med.iloc[-1])])
        w = 2 * half + 1
        base = pad.rolling(w, center=True, min_periods=1).min().rolling(w, center=True, min_periods=1).max()
        d = np.clip(med.values - base.values[half:len(pad) - half], 0, None)
        sig = pd.Series(b.sigma_rho.values[i], index=fr).reindex(med.index).interpolate().values
        cand = d > np.maximum(AIR_MIN_M, AIR_K * sig)
        keep = np.zeros(len(d), bool)
        edges = np.flatnonzero(np.diff(np.r_[0, cand.astype(int), 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            pk = a + int(np.argmax(d[a:e]))
            A = d[pk]
            lo, hi = _span(d, pk, A / 2)
            T = 2 * (hi - lo + 1) / FPS                  # a raised-cosine bump lasts twice its width at half height
            if 2 * np.pi ** 2 * A / T ** 2 > ACC_MAX:
                lo, hi = _span(d, pk, BUMP_EDGE * A)
                keep[lo:hi + 1] = True
        bump[i] = np.where(keep, d, 0)[fr - fr[0]]
    return bump


def kalman_rts(z, R, gate=GATE):
    """z: (n, 2) measurements (NaN = none), R: (n, 2, 2). Returns smoothed states (n, 4) and outlier flags, or
    (None, k) when MAX_REJECT outliers in a row mean the track must restart at frame index k."""
    n = len(z)
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))
    out = np.zeros(n, bool)
    x = np.r_[z[0], 0, 0]
    P = np.zeros((4, 4)); P[:2, :2] = R[0]; P[2:, 2:] = np.eye(2) * V0_SIGMA ** 2
    rejected, first = 0, 0
    for k in range(n):
        if k:
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if k and np.isfinite(z[k, 0]):                # z[0] is already in the initial state
            y = z[k] - x[:2]
            Si = np.linalg.inv(P[:2, :2] + R[k])
            if y @ Si @ y > gate:
                out[k] = True
                if not rejected:
                    first = k
                rejected += 1
                if rejected >= MAX_REJECT:
                    return None, first
            else:
                rejected = 0
                K = P @ HM.T @ Si
                x = x + K @ y
                P = (np.eye(4) - K @ HM) @ P
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs, out


def smooth(sub, gate=GATE):
    """smooth one stretch of a track (frames may have holes <= MAX_GAP). Returns (DataFrame, None) or
    (None, frame) where the track has to restart."""
    f = np.arange(sub.frame.iloc[0], sub.frame.iloc[-1] + 1)
    idx = np.searchsorted(f, sub.frame.values)
    z = np.full((len(f), 2), np.nan); z[idx] = sub[['raw_x', 'raw_y']].values
    R = np.repeat(np.eye(2)[None] * 1e6, len(f), 0); R[idx] = np.stack(sub.R.values)
    xs, out = kalman_rts(z, R, gate)
    if xs is None:
        return None, f[out]
    return pd.DataFrame({'frame': f, 'x': xs[:, 0], 'y': xs[:, 1], 'vx': xs[:, 2], 'vy': xs[:, 3], 'outlier': out}), None


def smooth_track(g):
    """g: one player's rows in one camera segment, sorted by frame. Long gaps split it; at a restart the part before
    the rejected run is smoothed on its own (same gate: its forward pass never reached MAX_REJECT) and smoothing
    carries on from the rejected run. Each piece gets its own number (column piece)."""
    fr = g.frame.values
    bounds = np.r_[0, np.nonzero(np.diff(fr) > MAX_GAP + 1)[0] + 1, len(fr)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        while b - a >= 2:
            res, restart = smooth(g.iloc[a:b])
            if res is not None:
                rows.append(res)
                break
            cut = max(a + int(np.searchsorted(fr[a:b], restart)), a + 1)
            if cut - a >= 2:
                rows.append(smooth(g.iloc[a:cut])[0])
            a = cut
    rows = [r.assign(piece=k) for k, r in enumerate(rows) if r is not None]
    return pd.concat(rows) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    boxes = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'player_boxes_v3video.csv'),
                        usecols=['frame', 'v2_frame', 'player', 'team', 'x1', 'y1', 'x2', 'y2', 'feet_x', 'feet_y', 'source'])
    ball = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_v3video.csv'), usecols=['frame', 'v2_frame', 'shot'])
    cut = (ball.shot.diff() != 0) | (ball.v2_frame.diff() != 1)        # camera cut or removed dead time
    ball['cam_seg'] = cut.cumsum() - 1
    b = ground_points(boxes)
    b = b.merge(ball[['frame', 'cam_seg']], on='frame').sort_values(['player', 'frame']).reset_index(drop=True)
    x, y, J = to_court(b.u.values, b.v.values)
    b['raw_x'], b['raw_y'] = x, y
    dv = np.linalg.norm(J[:, :, 1], axis=1)                  # m per image pixel along the image vertical
    b['sigma_rho'] = SIGMA_PX * dv
    b['bump'] = airborne(b)
    b['jump'] = b.bump > 0
    up = -J[:, :, 1] / dv[:, None]                           # where a point above the sand projects to
    j = b.jump.values
    b.loc[j, 'raw_x'] -= b.bump[j] * up[j, 0]                # take the jump out of the measurement ...
    b.loc[j, 'raw_y'] -= b.bump[j] * up[j, 1]
    Sig = np.zeros((len(b), 2, 2)); Sig[:, 0, 0] = b.sigma_u ** 2; Sig[:, 1, 1] = b.sigma_v ** 2
    R = J @ Sig @ J.transpose(0, 2, 1)
    R[j] += (b.bump.values[j] ** 2)[:, None, None] * up[j, :, None] * up[j, None, :]   # ... trusting that part less
    b['R'] = list(R)
    res = []
    for (pl, seg), g in b[np.isfinite(b.raw_x)].groupby(['player', 'cam_seg']):
        if len(g) < 2:
            continue
        s = smooth_track(g)
        if s is not None:
            res.append(s.assign(player=pl, cam_seg=seg))
    s = pd.concat(res)
    s['track'] = ((s.player != s.player.shift()) | (s.cam_seg != s.cam_seg.shift()) | (s.piece != s.piece.shift())
                  | (s.frame.diff() != 1)).cumsum()
    s = s.merge(b[['frame', 'player', 'posture', 'src', 'near_touch', 'jump', 'bump', 'u', 'v', 'raw_x', 'raw_y']], on=['frame', 'player'],
                how='left')
    s['src'] = s.src.fillna('none')
    s['posture'] = s.posture.fillna('none')
    s['near_touch'] = s.near_touch.fillna(False).astype(bool)
    s['jump'] = s.jump.fillna(False).astype(bool)
    s = s.merge(ball[['frame', 'v2_frame']], on='frame')
    s['time_sec'] = (s.frame / FPS).round(2)
    s['team'] = s.player.str[:3]
    s['speed'] = np.hypot(s.vx, s.vy)
    s['side'] = np.where(s.x < 9, 'LEFT', 'RIGHT')
    s['square'] = [square(a, c) for a, c in zip(s.x, s.y)]
    cols = ['frame', 'v2_frame', 'time_sec', 'player', 'team', 'track', 'posture', 'src', 'near_touch', 'u', 'v',
            'raw_x', 'raw_y', 'jump', 'bump', 'outlier',
            'x', 'y', 'vx', 'vy', 'speed', 'side', 'square']
    s = s[cols].sort_values(['frame', 'player'])
    for c in ('raw_x', 'raw_y', 'bump', 'x', 'y', 'vx', 'vy', 'speed'):
        s[c] = s[c].round(3)
    s['u'], s['v'] = s.u.round(1), s.v.round(1)
    s.to_csv(os.path.join(OUT, 'player_positions_v3.csv'), index=False)
    m = s[np.isfinite(s.raw_x)]
    print(f'{len(s)} rows ({len(s) - len(m)} bridged), {s.track.nunique()} tracks, outliers {m.outlier.mean():.1%}')
    print('postures:', s.posture.value_counts().to_dict())
    print('speed m/s quantiles:', s.speed.quantile([.5, .9, .99, .999]).round(2).to_dict())


if __name__ == '__main__':
    main()
