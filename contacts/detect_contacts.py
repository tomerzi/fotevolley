"""Find every ball touch and decide which player touched it and with which body part.
inputs : ball/ball_boxes.csv, pose/vitpose_body25.csv, player_boxes.csv, homography_px_to_m.npy
outputs: contacts/ball_contacts.csv   one row per touch (frame, player, team, body_part, side, confidence, ...)
         contacts/unassigned_ball_events.csv   velocity jumps with no player within reach (sand, net, glitches)
         contacts/contact_candidates.csv   every (touch, player) pair that was considered, with its costs

1. touch frames: the ball is a free flight between touches, so a touch is where the velocity jumps.
   For every frame a quadratic is fitted to the observed ball centres on each side (detected/rescued only, never
   across a scene cut) and the jump |v_after - v_before| is measured; local peaks above V_MIN are touches.
2. who / which part: each player's BODY_25 skeleton is turned into capsules (head, chest, shoulder, thigh, knee,
   shin, foot, arm, hand). The 2D gap between the ball and each capsule, in metres at that player's depth, is the
   proximity cost.
3. 2D overlap (e.g. the head of the near player and the foot of the far player both touch the ball in the image):
   resolved by depth. The ball's size in pixels says how far from the camera it is: expected diameter =
   BALL_K * (px per metre at the player's court position, from the homography). The player whose depth matches
   the ball's size wins. A player on the other side of the net from the ball is penalised, and the same
   player touching twice in a row is penalised (footvolley rule), solved jointly per rally with Viterbi.
usage: python contacts/detect_contacts.py"""
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'contacts')
FPS = 25
OBS = ('detected', 'rescued')
FIT_W = 5            # observed points used on each side of a frame
FIT_SPAN = 9         # ...taken from at most this many frames away
V_MIN = 7.0          # px/frame velocity jump to call a touch
PEAK_SEP = 5         # frames between two touches at least
SEARCH = 2           # frames around the velocity peak searched for the touching skeleton
S_MIN = 0.3          # keypoint score
BALL_M = 0.22        # nominal ball diameter (m), BALL_K is re-fitted on clear touches
REACH_M = 0.45       # touch rejected (no player) if the nearest capsule is farther than this
SAME_PLAYER_COST = 1.5
WRONG_SIDE_COST = 2.0
NET_BAND = 45        # px around the net line where the side of the ball is uncertain
DEPTH_W = 2.5        # weight of |log(ball size / expected size)|
ARM_BIAS_M = 0.06    # arms/hands are illegal in play and the raised arm often overlaps the head in 2D
GAP_RALLY = 2 * FPS  # touches farther apart than this start a new rally (sequence constraint)

# capsules: (name, joint a, joint b or None, radius m). chest = neck .. 40% of the way to mid_hip
PARTS = [('head', 'HEAD', None, 0.12),
         ('chest', 'neck', 'CHEST', 0.14),
         ('shoulder', 'r_shoulder', None, 0.08), ('shoulder', 'l_shoulder', None, 0.08),
         ('torso', 'CHEST', 'mid_hip', 0.13),
         ('thigh', 'r_hip', 'r_knee', 0.09), ('thigh', 'l_hip', 'l_knee', 0.09),
         ('knee', 'r_knee', None, 0.07), ('knee', 'l_knee', None, 0.07),
         ('shin', 'r_knee', 'r_ankle', 0.06), ('shin', 'l_knee', 'l_ankle', 0.06),
         ('foot', 'r_ankle', 'r_big_toe', 0.07), ('foot', 'r_ankle', 'r_heel', 0.06),
         ('foot', 'l_ankle', 'l_big_toe', 0.07), ('foot', 'l_ankle', 'l_heel', 0.06),
         ('arm', 'r_shoulder', 'r_elbow', 0.05), ('arm', 'r_elbow', 'r_wrist', 0.05),
         ('arm', 'l_shoulder', 'l_elbow', 0.05), ('arm', 'l_elbow', 'l_wrist', 0.05),
         ('hand', 'r_wrist', 'R_HAND', 0.06), ('hand', 'l_wrist', 'L_HAND', 0.06)]
SIDE_OF = {'r_': 'right', 'l_': 'left'}
HEAD_J = ['nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

H = np.load(os.path.join(ROOT, 'homography_px_to_m.npy'))
Hi = np.linalg.inv(H)


def px_per_m(cx, cy):
    """horizontal image scale (px per metre) at court point(s) (cx, cy) in metres."""
    cx, cy = np.atleast_1d(cx).astype(float), np.atleast_1d(cy).astype(float)
    a = cv2.perspectiveTransform(np.c_[cx - .5, cy][None], Hi)[0]
    b = cv2.perspectiveTransform(np.c_[cx + .5, cy][None], Hi)[0]
    return np.hypot(*(b - a).T)


def net_x(y):
    """image x of the net line (court x = 9 m) at image row y; the camera sits near the net plane."""
    p = cv2.perspectiveTransform(np.array([[[9., 0.], [9., 9.]]]), Hi)[0]
    return p[0, 0] + (y - p[0, 1]) * (p[1, 0] - p[0, 0]) / (p[1, 1] - p[0, 1])


# ---------------------------------------------------------------- ball
ball = pd.read_csv(os.path.join(ROOT, 'ball', 'ball_boxes.csv'))
N = len(ball)
obs = ball.source.isin(OBS).values
bx, by, shot = ball.center_x.values, ball.center_y.values, ball.shot.values
bd = np.where(obs, ((ball.x2 - ball.x1) + (ball.y2 - ball.y1)).values / 2, np.nan)


def side_pts(t, step):
    pts = []
    g = t
    while len(pts) < FIT_W and abs(g - t) <= FIT_SPAN and 0 <= g < N and shot[g] == shot[t]:
        if obs[g]:
            pts.append(g)
        g += step
    return pts


def fit(pts, t):
    """position and velocity at frame t from a quadratic (linear if < 4 points) through pts."""
    tt = np.array(pts, float) - t
    deg = 2 if len(pts) >= 4 else 1
    px, py = np.polyfit(tt, bx[pts], deg), np.polyfit(tt, by[pts], deg)
    pos = np.array([np.polyval(px, 0), np.polyval(py, 0)])
    vel = np.array([np.polyval(np.polyder(px), 0), np.polyval(np.polyder(py), 0)])
    return pos, vel


dv = np.zeros(N)
pos_t = np.full((N, 2), np.nan)
vin = np.full((N, 2), np.nan); vout = np.full((N, 2), np.nan)
for t in range(N):
    L, R = side_pts(t, -1), side_pts(t, 1)
    if len(L) < 3 or len(R) < 3 or (not obs[t] and (t - L[0] > 4 or R[0] - t > 4)):
        continue
    (pl, vl), (pr, vr) = fit(L, t), fit(R, t)
    if np.hypot(*(pl - pr)) > 60:          # the two sides are not the same ball (tracking switch)
        continue
    dv[t] = np.hypot(*(vr - vl))
    pos_t[t] = (bx[t], by[t]) if obs[t] else (pl + pr) / 2
    vin[t], vout[t] = vl, vr

peaks = []
for t in np.argsort(-dv):
    if dv[t] < V_MIN:
        break
    if all(abs(t - p) >= PEAK_SEP for p in peaks):
        peaks.append(int(t))
peaks.sort()


def ball_size(t):
    w = [g for g in range(t - 6, t + 7) if 0 <= g < N and shot[g] == shot[t] and obs[g]]
    return float(np.nanmedian(bd[w])) if w else np.nan


# ---------------------------------------------------------------- players
pose = pd.read_csv(os.path.join(ROOT, 'pose', 'vitpose_body25.csv'))
boxes = pd.read_csv(os.path.join(ROOT, 'player_boxes.csv'))
boxes['scale'] = px_per_m(boxes.court_x.values, boxes.court_y.values)
# smooth the depth scale per player (a jump lifts the feet and fakes a farther court position)
boxes = boxes.sort_values(['player', 'frame'])
boxes['scale'] = boxes.groupby('player').scale.transform(lambda s: s.rolling(9, center=True, min_periods=1).median())
pose = pose.merge(boxes[['frame', 'player', 'court_x', 'court_y', 'court_side', 'scale', 'x1', 'y1', 'x2', 'y2']],
                  on=['frame', 'player'], how='left')
pose_by_frame = {f: g for f, g in pose.groupby('frame')}


def joint(r, j):
    if j == 'HEAD':
        p = [(r[f'{k}_x'], r[f'{k}_y']) for k in HEAD_J if r[f'{k}_s'] >= S_MIN]
        return np.mean(p, 0) if p else None
    if j == 'CHEST':
        a, b = joint(r, 'neck'), joint(r, 'mid_hip')
        return None if a is None or b is None else a + 0.4 * (b - a)
    if j in ('R_HAND', 'L_HAND'):       # hand centre: the forearm extended by 35% past the wrist
        k = j[0].lower()
        a, b = joint(r, f'{k}_elbow'), joint(r, f'{k}_wrist')
        return None if a is None or b is None else b + 0.35 * (b - a)
    return np.array([r[f'{j}_x'], r[f'{j}_y']]) if r[f'{j}_s'] >= S_MIN else None


def seg_dist(p, a, b):
    if b is None:
        return np.hypot(*(p - a))
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1)
    return np.hypot(*(p - a - u * ab))


def part_gaps(r, p, ball_r):
    """gap (m, <0 = overlap) between the ball (centre p, radius ball_r px) and each capsule of skeleton row r."""
    out = []
    for name, ja, jb, rad in PARTS:
        a = joint(r, ja)
        b = joint(r, jb) if jb else None
        if a is None or (jb and b is None):
            continue
        d = (seg_dist(p, a, b) - ball_r) / r.scale - rad + (ARM_BIAS_M if name in ('arm', 'hand') else 0)
        lr = SIDE_OF.get(ja[:2], '') if name not in ('head', 'chest', 'torso') else ''
        out.append((d, name, lr))
    return out


# ---------------------------------------------------------------- candidates per touch
rows = []
for t in peaks:
    d_ball = ball_size(t)
    side_px = pos_t[t, 0] - net_x(pos_t[t, 1])
    ball_side = 'LEFT' if side_px < -NET_BAND else 'RIGHT' if side_px > NET_BAND else 'NET'
    best = {}
    for f in range(t - SEARCH, t + SEARCH + 1):
        if f not in pose_by_frame or np.isnan(pos_t[t, 0]):
            continue
        # the touch point is where the ball is at frame t; neighbouring skeletons only absorb pose/timing jitter.
        # (the ball's own position in frame f would match whoever it flew past just before or after the touch)
        p = pos_t[t]
        for r in pose_by_frame[f].itertuples(index=False):
            r = r._asdict()
            r = pd.Series(r)
            if np.isnan(r.scale):
                continue
            gaps = part_gaps(r, p, (d_ball if not np.isnan(d_ball) else 27) / 2)
            if not gaps:
                continue
            g, part, lr = min(gaps)
            # second-best part of this player (for reporting how sure the part is)
            others = sorted(x for x in gaps if x[1] != part)
            g2, part2 = (others[0][0], others[0][1]) if others else (np.nan, '')
            cost_t = abs(f - t) * 0.15
            k = r.player
            if k not in best or g + cost_t < best[k]['gap_m'] + best[k]['dt_cost']:
                best[k] = dict(frame=t, player=k, team=r.team, court_side=r.court_side, contact_pose_frame=f,
                               body_part=part, limb_side=lr, gap_m=round(g, 3), dt_cost=cost_t,
                               part2=part2, gap2_m=round(g2, 3) if not np.isnan(g2) else np.nan,
                               player_scale=r.scale, court_x=r.court_x, court_y=r.court_y)
    for k, c in best.items():
        c.update(ball_x=round(pos_t[t, 0], 1), ball_y=round(pos_t[t, 1], 1), ball_d=d_ball, dv=round(dv[t], 1),
                 ball_side=ball_side)
        rows.append(c)
cand = pd.DataFrame(rows)

# ball size per metre of depth, fitted on touches where only one player is within reach
near = cand[cand.gap_m < 0.1]
solo = near.groupby('frame').filter(lambda g: len(g) == 1)
BALL_K = float(np.nanmedian(solo.ball_d / solo.player_scale)) if len(solo) > 10 else BALL_M
cand['expected_d'] = BALL_K * cand.player_scale
cand['depth_ratio'] = cand.ball_d / cand.expected_d
cand['prox_cost'] = np.clip(cand.gap_m, 0, None) / 0.1 + cand.dt_cost
cand['depth_cost'] = DEPTH_W * np.abs(np.log(cand.depth_ratio.fillna(1)))
wrong = (cand.ball_side != 'NET') & (cand.ball_side != cand.court_side)
cand['side_cost'] = np.where(wrong, WRONG_SIDE_COST, 0.)
cand['cost'] = cand.prox_cost + cand.depth_cost + cand.side_cost
cand.loc[cand.gap_m > REACH_M, 'cost'] = np.inf

# ---------------------------------------------------------------- sequence (Viterbi per rally)
frames = sorted(cand.frame.unique())
choice = {}
NONE = '-'
rally_id, rid, prev_f = {}, 0, -10 ** 9
for f in frames:
    if f - prev_f > GAP_RALLY:
        rid += 1
    rally_id[f] = rid; prev_f = f
for rid_, fs in pd.Series(frames).groupby(pd.Series([rally_id[f] for f in frames])):
    fs = list(fs)
    states = []
    for f in fs:
        g = cand[cand.frame == f]
        st = {r.player: r.cost for r in g.itertuples() if np.isfinite(r.cost)}
        st[NONE] = 4.0           # nobody (ground, net, tracking glitch)
        states.append(st)
    acc = [{s: (c, None) for s, c in states[0].items()}]
    for i in range(1, len(fs)):
        cur = {}
        for s, c in states[i].items():
            opts = [(acc[-1][q][0] + c + (SAME_PLAYER_COST if s == q and s != NONE else 0), q) for q in acc[-1]]
            cur[s] = min(opts)
        acc.append(cur)
    s = min(acc[-1], key=lambda k: acc[-1][k][0])
    for i in range(len(fs) - 1, -1, -1):
        choice[fs[i]] = s
        s = acc[i][s][1]

# ---------------------------------------------------------------- output
out, unassigned = [], []
for f in frames:
    g = cand[cand.frame == f].sort_values('cost')
    who = choice[f]
    fin = g[np.isfinite(g.cost)]
    alt = fin[fin.player != who]
    base = dict(frame=f, time_sec=round(f / FPS, 2), rally=rally_id[f],
                ball_x=g.ball_x.iloc[0], ball_y=g.ball_y.iloc[0], ball_side=g.ball_side.iloc[0],
                velocity_jump=g.dv.iloc[0])
    if who == NONE:
        unassigned.append(dict(base, note='no skeleton within reach (sand / net / tracking glitch)'))
        continue
    r = g[g.player == who].iloc[0]
    margin = (alt.cost.iloc[0] - r.cost) if len(alt) else 10.0
    part_margin = (r.gap2_m - r.gap_m) if not np.isnan(r.gap2_m) else 1.0
    conf = float(np.clip(1 / (1 + np.exp(-(margin - 0.5) * 2)), 0, 1) * np.exp(-max(r.gap_m, 0) / 0.2))
    note = []
    if len(alt) and alt.prox_cost.iloc[0] < 1.0:
        a = alt.iloc[0]
        note.append(f'2D overlap with {a.player} {a.body_part} (gap {a.gap_m:.2f} m); '
                    f'depth ball/expected {r.depth_ratio:.2f} vs {a.depth_ratio:.2f}')
    if r.side_cost > 0:
        note.append('ball on the other side of the net')
    if part_margin < 0.05:
        note.append(f'part close to {r.part2}')
    if r.body_part == 'hand':
        note.append('hand: ball held/caught, probably dead ball')
    out.append(dict(base, player=who, team=r.team, body_part=r.body_part, limb_side=r.limb_side,
                    confidence=round(conf, 2), gap_m=r.gap_m, depth_ratio=round(r.depth_ratio, 2),
                    second_player=alt.player.iloc[0] if len(alt) else '',
                    second_part=alt.body_part.iloc[0] if len(alt) else '',
                    second_cost_margin=round(margin, 2), note='; '.join(note)))
res = pd.DataFrame(out)
res.to_csv(os.path.join(OUT, 'ball_contacts.csv'), index=False)
pd.DataFrame(unassigned).to_csv(os.path.join(OUT, 'unassigned_ball_events.csv'), index=False)
cand.drop(columns=['dt_cost']).round(3).to_csv(os.path.join(OUT, 'contact_candidates.csv'), index=False)
print(f'BALL_K={BALL_K:.3f} m-equivalent, {len(peaks)} velocity peaks, {len(res)} touches, {len(unassigned)} unassigned')
print(res.body_part.value_counts().to_dict())
print(res.player.value_counts().to_dict())
