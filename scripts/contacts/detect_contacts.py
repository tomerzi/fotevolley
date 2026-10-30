"""Who touched the ball, with which body part, in which frame (v3 video, Sapiens2 pose).

inputs : data/boxes/ball_boxes_v3video.csv, data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv,
         data/boxes/player_boxes_v3video.csv, data/points/rallies_v3video.csv, data/mappings/homography_px_to_m.npy,
         data/contacts/manual_overrides.csv (optional, decisions made by watching the video)
outputs: data/contacts/ball_contacts_v3.csv   one row per ball event (touch, ground, net...), with player and part
         data/contacts/contact_frames_v3.csv  one row per video frame: who touches the ball with what (empty if nobody)
         data/contacts/contact_candidates_v3.csv  every (event, player) pair considered, with every cost

1. events: the ball path is split into free-flight pieces (scripts/contacts/ballflight.py); each break is an event,
   at the sub-frame time where the two pieces meet.
2. candidates: each player's Sapiens2 skeleton becomes capsules (head, chest, shoulder, torso, thigh, knee, shin, foot,
   arm, hand). Gap ball <-> capsule is measured in metres at the player's depth.
3. 2D overlap is resolved with physics, not only proximity:
   - depth: the ball's size in pixels must match the player's distance from the camera (homography scale);
   - impulse: the velocity change of the ball points away from the touching part (contact normal);
   - limb motion: a kicking foot / heading head moves the way the ball is pushed;
   - net side: the toucher stands on the side of the net where the ball is.
4. rally rules, solved jointly per rally (Viterbi): no player touches twice in a row, at most 3 touches per team, the
   team changes when the ball crossed the net. Arms/hands are faults in footvolley, so they cost extra in play.
5. dead ball: touches before the serve (bouncing the ball) and after the ball reached the sand are kept but marked.
usage: python scripts/contacts/detect_contacts.py"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ballflight import segment, events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'contacts')
FPS = 25
OBS = ('detected', 'rescued', 'manual')
DV_MIN = 6.0         # px/frame velocity change to keep a break as an event (smaller = curve-fit artefact)
S_MIN = 0.3          # keypoint score
REACH_M = 0.45       # no player if the nearest capsule is farther than this
POSE_DT = 1          # skeletons from t-1..t+1 are tried (pose / timing jitter)
DEPTH_W = 3.0        # weight of |log(ball size / size expected at the player's depth)|
IMPULSE_W = 1.0      # weight of the ball being pushed into the part instead of away from it
MOTION_W = 0.6       # weight of the part moving against the push
SIDE_COST = 2.5      # player on the other side of the net from the ball
NET_BAND = 40        # px around the net line where the ball's side is uncertain
ILLEGAL_PART = {'arm': 0.8, 'hand': 1.2}   # in play only
SAME_PLAYER = 3.0
SAME_TEAM_AFTER_CROSS = 3.0
OTHER_TEAM_NO_CROSS = 2.0
FOURTH_TOUCH = 2.5
NONE_COST = 3.0      # event explained by nobody (sand, net, tracking)
LIFT_NONE = 1.5      # extra "nobody" cost when the ball is sent back up next to a player on its side of the net
LIFT_REACH_M = 0.30  # (a header / shoulder under a falling ball: the ball sits above the head in this camera)
LIFT_VY = -5.0       # px/frame: the ball leaves upward in the image
BALL_M = 0.215
BACK_MAX = 8         # frames a flight start is traced back through hidden frames (serve kicked unseen)
ARM_BIAS_M = 0.10    # an arm next to the torso overlaps it in 2D; touches with the arm are rare, so it loses ties
HEAD_OVER_SHOULDER_M = 0.10  # ball at face height: a shoulder nearer in 2D by less than this is a header off the side
SHIN_BIAS_M = 0.03   # the shin sits right above the foot in 2D; kicks are made with the foot far more often
REST_SPEED = 4.0     # px/frame: a ball slower than this before a kick was resting (serve from the sand)

J = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
     'right_elbow', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_big_toe',
     'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel', 'right_wrist', 'left_wrist',
     'left_acromion', 'right_acromion', 'neck', 'center_of_glabella']
JI = {j: i for i, j in enumerate(J)}
# capsules: (name, side, a, b, radius m). Derived points: FACE, HEADTOP, CHEST, MIDHIP, L_HAND, R_HAND
PARTS = [('head', '', 'FACE', 'HEADTOP', 0.11),
         ('chest', '', 'neck', 'CHEST', 0.15),
         ('shoulder', 'right', 'right_shoulder', None, 0.08), ('shoulder', 'left', 'left_shoulder', None, 0.08),
         ('torso', '', 'CHEST', 'MIDHIP', 0.14),
         ('thigh', 'right', 'right_hip', 'right_knee', 0.09), ('thigh', 'left', 'left_hip', 'left_knee', 0.09),
         ('knee', 'right', 'right_knee', None, 0.07), ('knee', 'left', 'left_knee', None, 0.07),
         ('shin', 'right', 'right_knee', 'right_ankle', 0.05), ('shin', 'left', 'left_knee', 'left_ankle', 0.05),
         ('foot', 'right', 'right_ankle', 'right_big_toe', 0.09), ('foot', 'right', 'right_heel', 'right_small_toe', 0.09),
         ('foot', 'left', 'left_ankle', 'left_big_toe', 0.09), ('foot', 'left', 'left_heel', 'left_small_toe', 0.09),
         ('arm', 'right', 'right_shoulder', 'right_elbow', 0.05), ('arm', 'right', 'right_elbow', 'right_wrist', 0.05),
         ('arm', 'left', 'left_shoulder', 'left_elbow', 0.05), ('arm', 'left', 'left_elbow', 'left_wrist', 0.05),
         ('hand', 'right', 'right_wrist', 'R_HAND', 0.06), ('hand', 'left', 'left_wrist', 'L_HAND', 0.06)]
PLAYERS = ['BRA_A', 'BRA_B', 'ISR_A', 'ISR_B']
TEAM = {p: p[:3] for p in PLAYERS}

H = np.load(os.path.join(ROOT, 'data', 'mappings', 'homography_px_to_m.npy'))
Hi = np.linalg.inv(H)


def px_per_m(cx, cy):
    cx, cy = np.atleast_1d(cx).astype(float), np.atleast_1d(cy).astype(float)
    a = cv2.perspectiveTransform(np.c_[cx - .5, cy][None], Hi)[0]
    b = cv2.perspectiveTransform(np.c_[cx + .5, cy][None], Hi)[0]
    return np.hypot(*(b - a).T)


_NET = cv2.perspectiveTransform(np.array([[[9., 0.], [9., 9.]]]), Hi)[0]


def net_side(x, y):
    """signed px from the net line (camera is close to the net plane, so it holds for a ball in the air too)."""
    nx = _NET[0, 0] + (y - _NET[0, 1]) * (_NET[1, 0] - _NET[0, 0]) / (_NET[1, 1] - _NET[0, 1])
    return x - nx


def side_name(s):
    return 'LEFT' if s < -NET_BAND else 'RIGHT' if s > NET_BAND else 'NET'


# ---------------------------------------------------------------- ball events
def _rally_pieces(args):
    """free-flight pieces of one rally, one list per camera shot."""
    return [segment(t, x, y) for t, x, y in args]


def ball_events(ball, rallies):
    obs = ball.source.isin(OBS).values
    bd = np.where(obs, ((ball.x2 - ball.x1) + (ball.y2 - ball.y1)).values / 2, np.nan)
    jobs = []
    for r in rallies.itertuples():
        seg = ball.iloc[r.start_frame:r.end_frame + 1]
        seg = seg[obs[r.start_frame:r.end_frame + 1]]
        jobs.append([(g.frame.values.astype(float), g.center_x.values, g.center_y.values) for _, g in seg.groupby('shot')])
    with Pool(os.cpu_count()) as pool:
        all_pieces = pool.map(_rally_pieces, jobs)
    rows, flights = [], {}
    shot_of = ball.shot.values
    for r, shots in zip(rallies.itertuples(), all_pieces):
        pieces_r = []
        for p in shots:
            pieces_r += p
            if p:      # the first flight of a shot starts with a touch we cannot see the incoming side of (e.g. serve)
                a = p[0]
                sh = shot_of[a.t_first]
                hidden = 0           # unobserved frames just before the flight, in the same shot
                while hidden < BACK_MAX and a.t_first - hidden - 1 >= 0 and shot_of[a.t_first - hidden - 1] == sh \
                        and not obs[a.t_first - hidden - 1]:
                    hidden += 1
                hyp = [(a.t_first - k, a.pos(a.t_first - k)) for k in range(hidden + 1)]
                hyp = [(t, q) for t, q in hyp if -50 < q[0] < 1970 and -50 < q[1] < 1130]
                rows.append(dict(rally=r.rally, frame=a.t_first, t_exact=float(a.t_first), pos=a.pos(a.t_first),
                                 v_in=np.full(2, np.nan), v_out=a.vel(a.t_first), miss_px=0., gap=hidden + 1, n_in=0,
                                 n_out=a.n, kind='flight_start', hyp=hyp))
            for e in events(p):
                rows.append(dict(e, rally=r.rally, kind='break', hyp=[(e['t_exact'], e['pos'])]))
        flights[r.rally] = pieces_r
    ev = pd.DataFrame(rows)
    ev['dv'] = [np.hypot(*(o - i)) if np.all(np.isfinite(i)) else np.hypot(*o) for i, o in zip(ev.v_in, ev.v_out)]
    ev = ev[(ev.kind == 'flight_start') | (ev.dv >= DV_MIN) | (ev.gap > 3)].copy()
    ev = ev.sort_values(['rally', 't_exact']).reset_index(drop=True)
    # ball size around the event (median of observed boxes, both sides)
    N = len(ball)
    shot = ball.shot.values
    sizes = []
    for e in ev.itertuples():
        w = [g for g in range(e.frame - 5, e.frame + 6) if 0 <= g < N and shot[g] == shot[min(e.frame, N - 1)]]
        v = bd[w]
        sizes.append(float(np.nanmedian(v)) if np.isfinite(v).sum() >= 3 else np.nan)
    ev['ball_d'] = sizes
    ev['ball_x'] = [p[0] for p in ev.pos]
    ev['ball_y'] = [p[1] for p in ev.pos]
    ev['side_px'] = net_side(ev.ball_x.values, ev.ball_y.values)
    ev['shot'] = shot[np.clip(ev.frame.values, 0, N - 1)]
    return ev, flights


# ---------------------------------------------------------------- skeletons
def load_pose(boxes):
    cols = ['frame', 'player'] + [f'{j}_{c}' for j in J for c in 'xys']
    s = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'), usecols=cols)
    nf = int(max(s.frame.max(), boxes.frame.max())) + 1
    K = np.full((nf, len(PLAYERS), len(J), 2), np.nan)
    pi = s.player.map({p: i for i, p in enumerate(PLAYERS)}).values
    for j, name in enumerate(J):
        ok = s[f'{name}_s'].values >= S_MIN
        K[s.frame.values[ok], pi[ok], j, 0] = s[f'{name}_x'].values[ok]
        K[s.frame.values[ok], pi[ok], j, 1] = s[f'{name}_y'].values[ok]
    # depth scale (px per metre) and court side per player and frame, smoothed (a jump lifts the feet)
    b = boxes.sort_values(['player', 'frame']).copy()
    b['scale'] = px_per_m(b.court_x.values, b.court_y.values)
    b['scale'] = b.groupby('player').scale.transform(lambda q: q.rolling(9, center=True, min_periods=1).median())
    S = np.full((nf, len(PLAYERS)), np.nan)
    side = np.full((nf, len(PLAYERS)), '', object)
    bi = b.player.map({p: i for i, p in enumerate(PLAYERS)}).values
    S[b.frame.values, bi] = b.scale.values
    side[b.frame.values, bi] = b.court_side.values
    return K, S, side


def points(k):
    """named points of one skeleton k (J x 2), derived points included; None when missing."""
    def g(n):
        v = k[JI[n]]
        return None if np.isnan(v[0]) else v
    P = {n: g(n) for n in J}
    face = [P[n] for n in ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'center_of_glabella') if P[n] is not None]
    P['FACE'] = np.mean(face, 0) if face else None
    hips = [P[n] for n in ('left_hip', 'right_hip') if P[n] is not None]
    P['MIDHIP'] = np.mean(hips, 0) if hips else None
    neck = P['neck']
    if neck is None:
        sh = [P[n] for n in ('left_shoulder', 'right_shoulder') if P[n] is not None]
        neck = P['neck'] = np.mean(sh, 0) if len(sh) == 2 else None
    P['HEADTOP'] = None if P['FACE'] is None or neck is None else P['FACE'] + 0.55 * (P['FACE'] - neck)
    P['CHEST'] = None if neck is None or P['MIDHIP'] is None else neck + 0.4 * (P['MIDHIP'] - neck)
    for side, H_ in (('right', 'R_HAND'), ('left', 'L_HAND')):
        a, w = P[f'{side}_elbow'], P[f'{side}_wrist']
        P[H_] = None if a is None or w is None else w + 0.35 * (w - a)
    return P


def closest(p, a, b):
    if b is None:
        return a
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1)
    return a + u * ab


def capsules(P):
    out = []
    for name, side, ja, jb, rad in PARTS:
        a = P.get(ja)
        b = P.get(jb) if jb else None
        if a is None or (jb and b is None):
            continue
        out.append((name, side, a, b, rad))
    return out


# ---------------------------------------------------------------- candidates
def candidates(ev, K, S, side):
    nf = K.shape[0]
    rows = []
    for e in ev.itertuples():
        r_px = (e.ball_d if np.isfinite(e.ball_d) else 27.) / 2
        imp = np.asarray(e.v_out) - (np.asarray(e.v_in) if np.all(np.isfinite(e.v_in)) else 0)
        imp_n = imp / max(np.hypot(*imp), 1e-6)
        for pi, pl in enumerate(PLAYERS):
            best = None
            for (th, p), df in [(h, d) for h in e.hyp for d in range(-POSE_DT, POSE_DT + 1)]:
                p = np.asarray(p)
                back_cost = 0.12 * (e.t_exact - th)
                f = int(round(th)) + df
                if not 0 <= f < nf or np.isnan(S[f, pi]):
                    continue
                P = points(K[f, pi])
                sc = S[f, pi]
                # skeleton one frame before/after for the limb's own motion
                Pm = points(K[max(f - 1, 0), pi]); Pp = points(K[min(f + 1, nf - 1), pi])
                parts = []
                for name, sd, a, b, rad in capsules(P):
                    c = closest(p, a, b)
                    d_px = np.hypot(*(p - c))
                    gap = (d_px - r_px) / sc - rad + (ARM_BIAS_M if name in ('arm', 'hand') else SHIN_BIAS_M if name == 'shin' else 0)
                    # impulse: the ball is pushed along the contact normal (part -> ball). Only meaningful when
                    # the ball centre is clearly off the capsule axis
                    imp_cost = 0.
                    if d_px > 0.5 * r_px and np.hypot(*imp) > 3:
                        cosv = float(np.dot((p - c) / d_px, imp_n))
                        imp_cost = (1 - cosv) / 2          # 0 when pushed straight away, 1 when pulled into it
                    # limb motion: the touching part moves with the push (a kick / a header), not against it
                    mot_cost, part_speed = 0., np.nan
                    ja = [x for x in PARTS if x[0] == name and x[1] == sd][0][2]
                    if Pm.get(ja) is not None and Pp.get(ja) is not None:
                        vel = (Pp[ja] - Pm[ja]) / 2
                        part_speed = np.hypot(*vel) / sc * FPS            # m/s
                        if part_speed > 2.0 and np.hypot(*imp) > 3:
                            mot_cost = max(0., -float(np.dot(vel / np.hypot(*vel), imp_n)))
                    parts.append((gap, name, sd, imp_cost, mot_cost, part_speed))
                if not parts:
                    continue
                parts.sort(key=lambda x: x[0])
                # the part: the nearest, unless another part within 4 cm fits the push much better
                g0 = parts[0][0]
                close = [x for x in parts if x[0] < g0 + 0.04]
                pick = min(close, key=lambda x: x[0] / 0.1 + x[3] + x[4])
                g_sel = pick[0]          # the pose frame is chosen on the nearest part, before relabelling
                head = next((x for x in parts if x[1] == 'head'), None)
                if pick[1] == 'shoulder' and head is not None and P['FACE'] is not None and \
                        p[1] < P['FACE'][1] + r_px and head[0] < pick[0] + HEAD_OVER_SHOULDER_M:
                    pick = head
                second = next((x for x in parts if x[1] != pick[1]), None)
                c = dict(event=e.Index, player=pl, team=TEAM[pl], pose_frame=f, court_side=side[f, pi], scale=sc,
                         gap_m=pick[0], body_part=pick[1], limb_side=pick[2], imp_cost=pick[3], mot_cost=pick[4],
                         part_speed=pick[5], part2=second[1] if second else '',
                         gap2_m=second[0] if second else np.nan,
                         dt_cost=0.15 * abs(f - th) + back_cost, touch_t=th)
                key = g_sel / 0.1 + c['dt_cost']
                if best is None or key < best[0]:
                    best = (key, c)
            if best is not None:
                rows.append(best[1])
    return pd.DataFrame(rows)


def score(cand, ev):
    near = cand[cand.gap_m < 0.08]
    solo = near.groupby('event').filter(lambda g: len(g) == 1)
    solo = solo.merge(ev[['ball_d']], left_on='event', right_index=True)
    k = float(np.nanmedian(solo.ball_d / solo.scale)) if len(solo) > 10 else BALL_M
    cand = cand.merge(ev[['ball_d', 'side_px']], left_on='event', right_index=True)
    cand['depth_ratio'] = cand.ball_d / (k * cand.scale)
    cand['prox_cost'] = np.clip(cand.gap_m, 0, None) / 0.1 + cand.dt_cost
    cand['depth_cost'] = DEPTH_W * np.abs(np.log(cand.depth_ratio.fillna(1)))
    bs = cand.side_px.map(side_name)
    cand['side_cost'] = np.where((bs != 'NET') & (bs != cand.court_side), SIDE_COST, 0.)
    cand['physics_cost'] = IMPULSE_W * cand.imp_cost + MOTION_W * cand.mot_cost
    cand['cost'] = cand.prox_cost + cand.depth_cost + cand.side_cost + cand.physics_cost
    cand.loc[cand.gap_m > REACH_M, 'cost'] = np.inf
    return cand.drop(columns=['ball_d', 'side_px']), k


# ---------------------------------------------------------------- rally logic
def crossings(flights, ev, ball):
    """for each event: did the ball cross the net between the previous event and this one?
    Uses the fitted flights, the observed ball centres and the two events' own positions."""
    crossed = np.zeros(len(ev), bool)
    o = ball[ball.source.isin(OBS)]
    for rally, g in ev.groupby('rally'):
        idx = list(g.index)
        for a, b in zip(idx[:-1], idx[1:]):
            t0, t1 = ev.t_exact[a], ev.t_exact[b]
            w = o[(o.frame >= t0) & (o.frame <= t1)]
            s = list(net_side(w.center_x.values, w.center_y.values)) + [ev.side_px[a], ev.side_px[b]]
            for pc in flights[rally]:
                lo, hi = max(t0, pc.t_first), min(t1, pc.t_last)
                for t in np.arange(lo, hi + 1e-6, 1.0):
                    q = pc.pos(t)
                    s.append(net_side(q[0], q[1]))
            s = np.array(s)
            crossed[b] = bool(len(s) and (s < -NET_BAND).any() and (s > NET_BAND).any())
    return crossed


def on_ground(ev, k):
    """is the ball lying on the sand? If its lowest point were on the ground, the homography gives where; the ball's
    pixel size there must match. A ball in the air maps to a farther ground point, where it should look smaller."""
    r = ev.ball_d.fillna(27).values / 2
    g = cv2.perspectiveTransform(np.c_[ev.ball_x.values, ev.ball_y.values + r][None].astype(float), H)[0]
    exp = k * px_per_m(g[:, 0], g[:, 1])
    return ((ev.ball_d.values / exp) < 1.12) & (np.abs(ev.side_px.values) > NET_BAND)


def in_play(ev, best_gap):
    """serve: walking back from the first net crossing, the first event where the ball was resting or first appears
    in the shot. Events before it (bouncing the ball, passing it to the server) are dead. After the ball reaches
    the sand with nobody touching it, the rally is over."""
    live = np.zeros(len(ev), bool)
    for seg, g in ev.groupby('seg'):
        idx = list(g.index)
        first = next((i for i in idx if ev.crossed[i]), None)
        if first is None:            # crossing not seen (ball lost): look back from the end of the rally
            first = idx[-1]
            k = len(idx)
        else:
            k = idx.index(first)
        serve = idx[0]
        for i in reversed(idx[:k]):
            si = ev.speed_in[i]
            if ev.kind[i] == 'flight_start' or (np.isfinite(si) and si < REST_SPEED):
                serve = i
                break
        if ev.shot[serve] != ev.shot[first]:
            serve = next(i for i in idx if ev.shot[i] == ev.shot[first])
        for i in idx[idx.index(serve):]:
            live[i] = True
            if i != serve and ev.ground[i] and best_gap.get(i, 9) > 0.15:
                break
    return live


def held(ev, cand):
    """the ball stops dead in the air on a player's arm/hand: it was caught, so play is over. Events after it in the
    same rally (e.g. a point replayed after a let) get their own serve search and rally rules."""
    near = cand.loc[cand.groupby('event').gap_m.idxmin()].set_index('event')
    out = np.zeros(len(ev), bool)
    for i in near.index:
        v = ev.v_out[i]
        out[i] = np.hypot(*v) < REST_SPEED and not ev.ground[i] and near.gap_m[i] < 0.05 and \
            near.body_part[i] in ('arm', 'hand')
    return out


def viterbi(ev, cand):
    """per rally: state = (who touched now, last real toucher, touches of that team on this side, ball crossed the
    net since that touch by the path test). The team changes when the ball is on the other court side from the last
    toucher; the path test is used only when the ball is at the net (its side is uncertain there)."""
    NONE = '-'
    choice = {}
    for seg, g in ev.groupby('seg'):
        idx = [i for i in g.index if ev.live[i]]
        if not idx:
            continue
        cr = cand[cand.event.isin(g.index) & (cand.court_side != '')]
        court = cr.groupby('player').court_side.agg(lambda q: q.mode().iloc[0]).to_dict()
        layers = []
        for i in idx:
            c = cand[(cand.event == i) & np.isfinite(cand.cost)]
            st = {r.player: r.cost + (ILLEGAL_PART.get(r.body_part, 0)) for r in c.itertuples()}
            # "nobody" must not be a cheap way around a rule when a body part is on the ball
            st[NONE] = NONE_COST + (max(0., 3.0 - min(st.values())) if st else 0.)
            # the ball was sent back up next to a player standing on its side: something hit it, not the sand
            if np.all(np.isfinite(ev.v_in[i])) and ev.v_in[i][1] > 0 and ev.v_out[i][1] < LIFT_VY and \
                    ((c.gap_m < LIFT_REACH_M) & (c.side_cost == 0)).any():
                st[NONE] += LIFT_NONE
            layers.append(st)
        dp = {}
        for who, c in layers[0].items():
            key = (who, who if who != NONE else NONE, 1 if who != NONE else 0, False)
            dp[key] = (c, None)
        hist = [dp]
        for li in range(1, len(idx)):
            i = idx[li]
            bs = side_name(ev.side_px[i])
            new = {}
            for (pw, last, cnt, xs), (acc, _) in hist[-1].items():
                xs2 = xs or bool(ev.crossed[i])
                for who, c in layers[li].items():
                    if who == NONE:
                        key, add = (NONE, last, cnt, xs2), 0.
                    else:
                        add = 0.
                        if last != NONE:
                            crossed = (bs != court[last]) if bs != 'NET' and last in court else xs2
                            same_team = TEAM[who] == TEAM[last]
                            if who == last and pw == last:
                                add += SAME_PLAYER
                            if crossed and same_team:
                                add += SAME_TEAM_AFTER_CROSS
                            if not crossed and not same_team and cnt > 0:
                                add += OTHER_TEAM_NO_CROSS
                            n = cnt + 1 if (same_team and not crossed) else 1
                        else:
                            n = 1
                        if n > 3:
                            add += FOURTH_TOUCH
                        key = (who, who, min(n, 4), False)
                    tot = acc + c + add
                    if key not in new or tot < new[key][0]:
                        new[key] = (tot, (pw, last, cnt, xs))
            hist.append(new)
        key = min(hist[-1], key=lambda k: hist[-1][k][0])
        for li in range(len(idx) - 1, -1, -1):
            choice[idx[li]] = key[0]
            key = hist[li][key][1]
    return choice


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    ball = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_v3video.csv'))
    rallies = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_v3video.csv'))
    boxes = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'player_boxes_v3video.csv'))
    ev, flights = ball_events(ball, rallies)
    K, S, side = load_pose(boxes)
    cand = candidates(ev, K, S, side)
    cand, k = score(cand, ev)
    ev['crossed'] = crossings(flights, ev, ball)
    ev['speed_in'] = [np.hypot(*v) if np.all(np.isfinite(v)) else np.nan for v in ev.v_in]
    ev['ground'] = on_ground(ev, k)
    best_gap = cand.groupby('event').gap_m.min().to_dict()
    ev['seg'] = ev.rally.astype(float)
    ev['live'] = in_play(ev, best_gap)
    catch = held(ev, cand)
    for i in np.flatnonzero(catch & ev.live.values):
        later = ev.index[(ev.seg == ev.seg[i]) & (ev.index > i)]
        ev.loc[later, 'seg'] = ev.seg[i] + 0.01        # play restarts after the catch with a new serve
        ev['live'] = in_play(ev, best_gap)
    ev.loc[catch, 'live'] = False
    choice = viterbi(ev, cand)

    # dead-ball events: nearest player by cost, no rally rules
    out = []
    ov_path = os.path.join(OUT, 'manual_overrides.csv')
    ov = pd.read_csv(ov_path).set_index('frame') if os.path.exists(ov_path) else pd.DataFrame()
    for i, e in ev.iterrows():
        c = cand[(cand.event == i) & np.isfinite(cand.cost)].sort_values('cost')
        if e.live:
            who = choice.get(i, '-')
            if who != '-' and c[c.player == who].body_part.iloc[0] == 'hand':
                ev.loc[i, 'live'] = False      # a hand on the ball = catching it: the ball is dead
                e = ev.loc[i]
        else:
            who = c.player.iloc[0] if len(c) and c.cost.iloc[0] < NONE_COST else '-'
        r = c[c.player == who].iloc[0] if who != '-' else None
        alt = c[c.player != who]
        row = dict(frame=int(e.frame), time_sec=round(e.t_exact / FPS, 2), t_exact=round(e.t_exact, 2),
                   rally=int(e.rally), in_play=bool(e.live), event_kind=e.kind,
                   ball_x=round(e.ball_x, 1), ball_y=round(e.ball_y, 1), ball_d=round(e.ball_d, 1),
                   ball_side=side_name(e.side_px), crossed_net_before=bool(e.crossed),
                   speed_in=round(float(np.hypot(*e.v_in)), 1) if np.all(np.isfinite(e.v_in)) else np.nan,
                   speed_out=round(float(np.hypot(*e.v_out)), 1), velocity_change=round(e.dv, 1),
                   hidden_frames=int(max(e.gap - 1, 0)), ball_on_sand=bool(e.ground))
        if r is None:
            row.update(player='', team='', body_part='', limb_side='', confidence=np.nan)
        else:
            margin = (alt.cost.iloc[0] - r.cost) if len(alt) else 10.
            part_margin = (r.gap2_m - r.gap_m) if np.isfinite(r.gap2_m) else 1.
            conf = 1 / (1 + np.exp(-(margin - 0.7) * 2)) * np.exp(-max(r.gap_m, 0) / 0.2)
            note = []
            if len(alt) and alt.prox_cost.iloc[0] < 1.0:
                a = alt.iloc[0]
                note.append(f'2D overlap with {a.player} {a.body_part} (gap {a.gap_m:.2f} m, depth {a.depth_ratio:.2f} '
                            f'vs {r.depth_ratio:.2f}, push {a.imp_cost:.2f} vs {r.imp_cost:.2f})')
            if r.side_cost > 0:
                note.append('ball on the other side of the net')
            if part_margin < 0.05:
                note.append(f'part close to {r.part2}')
            row['frame'] = int(round(r.touch_t))   # a flight start traced back to an unseen kick
            row.update(player=who, team=r.team, body_part=r.body_part, limb_side=r.limb_side,
                       confidence=round(float(conf), 2), gap_m=round(r.gap_m, 3), depth_ratio=round(r.depth_ratio, 2),
                       push_cost=round(r.imp_cost, 2), part_speed_ms=round(r.part_speed, 1),
                       second_player=alt.player.iloc[0] if len(alt) else '',
                       second_part=alt.body_part.iloc[0] if len(alt) else '',
                       cost_margin=round(float(margin), 2), note='; '.join(note))
        row['source'] = 'auto'
        if len(ov) and row['frame'] in ov.index:
            o = ov.loc[row['frame']]
            for col in ('player', 'body_part', 'limb_side', 'in_play'):
                if col in o and pd.notna(o[col]):
                    row[col] = o[col] if col != 'in_play' else bool(o[col])
            if row['player'] == '-':           # watched: nobody touched it
                row.update(player='', body_part='', limb_side='', gap_m=np.nan, second_player='', second_part='')
            row['team'] = TEAM.get(row['player'], '')
            row['source'] = 'video_review'
            row['confidence'] = 1.0
            row['note'] = str(o.get('note', '')) if pd.notna(o.get('note', '')) else ''
        out.append(row)
    res = pd.DataFrame(out)
    res['event_type'] = np.where(res.player != '', 'touch', 'no_player (sand/net/tracking)')
    res.to_csv(os.path.join(OUT, 'ball_contacts_v3.csv'), index=False)
    cand.round(3).to_csv(os.path.join(OUT, 'contact_candidates_v3.csv'), index=False)

    # per frame
    fr = pd.DataFrame({'frame': ball.frame, 'v2_frame': ball.v2_frame, 'time_sec': ball.time_sec})
    for c in ('player', 'team', 'body_part', 'limb_side'):
        fr[c] = ''
    fr['in_play'] = False
    fr['confidence'] = np.nan
    t = res[res.player != ''].drop_duplicates('frame', keep='first').set_index('frame')
    for c in ('player', 'team', 'body_part', 'limb_side', 'in_play', 'confidence'):
        fr.loc[fr.frame.isin(t.index), c] = t.loc[fr.frame[fr.frame.isin(t.index)], c].values
    fr.to_csv(os.path.join(OUT, 'contact_frames_v3.csv'), index=False)

    tch = res[res.player != '']
    print(f'ball size factor k={k:.3f}; {len(res)} events, {len(tch)} touches '
          f'({tch.in_play.sum()} in play), {len(res) - len(tch)} with no player')
    print(tch[tch.in_play].body_part.value_counts().to_dict())
    print(tch[tch.in_play].player.value_counts().to_dict())


if __name__ == '__main__':
    main()
