#!/usr/bin/env python3
"""v2 hypotheses: body orientation, stance, attack->defence transition, pose-based contact region.

Run after build_tables.py and analyze.py, from repo root:
    python3 footvolley-analysis-v1/scripts/analyze_v2.py
Pose = 2D ViTPose BODY_25 in image pixels (pose/vitpose_body25.csv). All angles are 2D image
proxies, never 3D joint angles. Gate: keypoint score >= 0.5. Writes derived/analysis_v2.json.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'footvolley-analysis-v1' / 'derived'
FPS = 25.0
GATE = 0.5
RNG = np.random.default_rng(7)
warnings.filterwarnings('ignore', category=RuntimeWarning)

pose = pd.read_csv(ROOT / 'pose/vitpose_body25.csv')
pl = pd.read_csv(D / 'players_norm.csv')
ball = pd.read_csv(ROOT / 'correct_bbox/ball_boxes_corrected3.csv')
ball = ball[ball.source.isin(['detected', 'rescued', 'manual'])].set_index('frame')
inc = pd.read_csv(D / 'incoming_possessions.csv')
snap = pd.read_csv(D / 'snapshots_incoming.csv')
poss = pd.read_csv(D / 'possessions.csv')
cands = pd.read_csv(D / 'contact_candidates.csv')
anchor = snap.groupby('event_id').anchor_frame.first()
inc['anchor_frame'] = inc.event_id.map(anchor)
res = {}

P = pose.merge(pl[['frame', 'player', 'x1', 'y1', 'x2', 'y2', 'court_x', 'depth_m', 'lateral_m', 'observed']],
               on=['frame', 'player'], how='inner')
P['box_h'] = P.y2 - P.y1
res['pose_join'] = {'pose_rows': int(len(pose)), 'joined_to_v9': int(len(P))}


def ok(df, *joints):
    m = pd.Series(True, index=df.index)
    for j in joints:
        m &= df[f'{j}_s'] >= GATE
    return m


# --------------------------------------------------------------- head yaw proxy toward ball
# nose x relative to mean of confidently-visible ears; sign compared with ball direction in image
ex = np.where(ok(P, 'l_ear') & ok(P, 'r_ear'), (P.l_ear_x + P.r_ear_x) / 2,
              np.where(ok(P, 'l_ear'), P.l_ear_x, np.where(ok(P, 'r_ear'), P.r_ear_x, np.nan)))
P['yaw_px'] = np.where(ok(P, 'nose'), P.nose_x - ex, np.nan)
P['yaw_norm'] = P.yaw_px / P.box_h
# 2D knee angle (hip-knee-ankle), mean of valid legs; hip height fraction in box (crouch proxy)


def ang(ax, ay, bx, by, cx, cy):
    v1 = np.c_[ax - bx, ay - by]; v2 = np.c_[cx - bx, cy - by]
    c = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


kl = np.where(ok(P, 'l_hip', 'l_knee', 'l_ankle'), ang(P.l_hip_x, P.l_hip_y, P.l_knee_x, P.l_knee_y, P.l_ankle_x, P.l_ankle_y), np.nan)
kr = np.where(ok(P, 'r_hip', 'r_knee', 'r_ankle'), ang(P.r_hip_x, P.r_hip_y, P.r_knee_x, P.r_knee_y, P.r_ankle_x, P.r_ankle_y), np.nan)
P['knee2d_deg'] = np.nanmean(np.c_[kl, kr], axis=1)
P['hip_rel_h'] = np.where(ok(P, 'mid_hip'), (P.y2 - P.mid_hip_y) / P.box_h, np.nan)  # fraction of box height above feet


def at(frame, team, offset=0):
    f = frame - offset
    g = P[(P.frame == f) & (P.team == team)].copy()
    if f in ball.index:
        bx = ball.loc[f, 'center_x']
        g['toward_ball'] = (np.sign(g.yaw_px) == np.sign(bx - g.nose_x)).astype(float)
        g.loc[g.yaw_px.abs() < 0.02 * g.box_h, 'toward_ball'] = np.nan  # frontal/ambiguous: unknown
        g.loc[g.yaw_px.isna(), 'toward_ball'] = np.nan
    else:
        g['toward_ball'] = np.nan
    return g


lab = inc[inc.fail.notna()].copy()
rows = []
for r in lab.itertuples():
    for off in (0, 5, 10):
        g = at(int(r.anchor_frame), r.team, off)
        for q in g.itertuples():
            rows.append({'event_id': r.event_id, 'rally_id': r.rally_id, 'team': r.team, 'player': q.player, 'offset': off,
                         'fail': r.fail, 'toward_ball': q.toward_ball, 'knee2d': q.knee2d_deg, 'hip_rel_h': q.hip_rel_h,
                         'depth_m': q.depth_m})
E = pd.DataFrame(rows)
E.to_csv(D / 'orientation_events.csv', index=False)


def pair_stat(df, col, fn):
    return df.groupby('event_id').agg(v=(col, fn), fail=('fail', 'first'), team=('team', 'first'), rally=('rally_id', 'first'))


h6 = {}
for off in (0, 5, 10):
    e = E[E.offset == off]
    known = e.dropna(subset=['toward_ball'])
    both = known.groupby('event_id').filter(lambda g: len(g) == 2)
    pe = both.groupby('event_id').agg(both_toward=('toward_ball', lambda x: bool(x.astype(bool).all())),
                                      fail=('fail', 'first'), team=('team', 'first'), rally=('rally_id', 'first'))
    out = {'player_rows_known': int(len(known)), 'pct_toward_ball_by_team': known.groupby('team').toward_ball.apply(lambda x: round(float(x.astype(float).mean()), 3)).to_dict()}
    for team in ('ISR', 'BRA', 'ALL'):
        t = pe if team == 'ALL' else pe[pe.team == team]
        a, b = t[t.both_toward], t[~t.both_toward]
        out[team] = {'both_toward_fail': f'{int(a.fail.sum())}/{len(a)}', 'not_both_fail': f'{int(b.fail.sum())}/{len(b)}',
                     'rallies': int(t.rally.nunique())}
    h6[f'offset_{off}f'] = out
res['H6_head_orientation_vs_fail'] = h6

h7 = {}
for off in (0, 5, 10):
    e = E[(E.offset == off)]
    pe = e.groupby('event_id').agg(knee=('knee2d', 'min'), hip=('hip_rel_h', 'min'), fail=('fail', 'first'), team=('team', 'first'))
    h7[f'offset_{off}f'] = {t: {'returned_median_min_knee2d': round(float(g[g.fail == 0].knee.median()), 1),
                                'failed_median_min_knee2d': round(float(g[g.fail == 1].knee.median()), 1),
                                'returned_median_min_hip_rel_h': round(float(g[g.fail == 0].hip.median()), 3),
                                'failed_median_min_hip_rel_h': round(float(g[g.fail == 1].hip.median()), 3),
                                'n_ret': int((g.fail == 0).sum()), 'n_fail': int((g.fail == 1).sum())}
                            for t, g in pe.groupby('team')}
res['H7_stance_vs_fail'] = h7

# --------------------------------------------------------------- H8 transition after own ball crosses out
# For each clean crossing OUT of team T's half followed by a clean incoming crossing back:
#  net player = partner with smaller depth at the out-crossing frame; measure his depth at both moments.
trans = []
for p in poss.itertuples():
    if p.is_last or not p.next_cross_clean:
        continue
    opp = poss[(poss.rally_id == p.rally_id) & (poss.poss_idx == p.poss_idx + 1)].iloc[0]
    if opp.is_last or not opp.next_cross_clean:
        back_fail = None
        outcome = 'opp_no_clean_return'
    else:
        ev = f'{p.rally_id}_in{p.poss_idx + 2}'
        m = inc[inc.event_id == ev]
        back_fail = None if m.empty or pd.isna(m.fail.iloc[0]) else float(m.fail.iloc[0])
        outcome = 'labelled' if back_fail is not None else 'unlabelled'
    f0, f1 = int(p.last_frame), int(opp.last_frame)
    a = pl[(pl.frame == f0) & (pl.team == p.team) & pl.observed]
    b = pl[(pl.frame == f1) & (pl.team == p.team) & pl.observed]
    if a.player.nunique() != 2 or b.player.nunique() != 2:
        continue
    netp = a.loc[a.depth_m.idxmin()]
    other = a.loc[a.depth_m.idxmax()]
    nb = b[b.player == netp.player].iloc[0]; ob = b[b.player == other.player].iloc[0]
    sep1 = float(np.hypot(*(b[['lateral_m', 'depth_m']].iloc[0] - b[['lateral_m', 'depth_m']].iloc[1])))
    trans.append({'rally_id': p.rally_id, 'team': p.team, 'net_player': netp.player, 'out_frame': f0, 'in_frame': f1,
                  'opp_time_s': (f1 - f0) / FPS, 'net_depth_out': netp.depth_m, 'net_depth_in': nb.depth_m,
                  'partner_depth_out': other.depth_m, 'partner_depth_in': ob.depth_m,
                  'net_recovery_m': nb.depth_m - netp.depth_m, 'sep_in': sep1, 'fail_next': back_fail, 'label': outcome})
T = pd.DataFrame(trans)
T.to_csv(D / 'transitions.csv', index=False)
h8 = {'transitions': int(len(T)), 'labelled': int(T.fail_next.notna().sum())}
for team, g in T.groupby('team'):
    h8[team] = {'n': int(len(g)), 'median_net_depth_at_attack_m': round(float(g.net_depth_out.median()), 2),
                'median_net_depth_at_next_incoming_m': round(float(g.net_depth_in.median()), 2),
                'median_recovery_m': round(float(g.net_recovery_m.median()), 2),
                'median_opp_time_s': round(float(g.opp_time_s.median()), 2),
                'net_player_counts': g.net_player.value_counts().to_dict()}
    l = g[g.fail_next.notna()]
    for name, m in [('net_depth_in<3m', l.net_depth_in < 3.0), ('net_depth_in>=3m', l.net_depth_in >= 3.0)]:
        h8[team][name] = f'{int(l[m].fail_next.sum())}/{int(m.sum())}'
    for name, m in [('net_player_recovered>=1.5m', l.net_recovery_m >= 1.5), ('recovered<1.5m', l.net_recovery_m < 1.5)]:
        h8[team][name] = f'{int(l[m].fail_next.sum())}/{int(m.sum())}'
# within-player: same net player, stayed-near-net vs recovered
res['H8_transition'] = h8

# --------------------------------------------------------------- H9 contact region via nearest pose keypoint
groups = {'head': ['nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear'], 'chest': ['neck', 'r_shoulder', 'l_shoulder'],
          'thigh': ['r_hip', 'l_hip', 'r_knee', 'l_knee'], 'foot': ['r_ankle', 'l_ankle', 'r_big_toe', 'l_big_toe', 'r_heel', 'l_heel']}
cc = cands.merge(pose.drop(columns=['team', 'box_source']), on=['frame', 'player'], how='left')
reg, dist = [], []
for r in cc.itertuples():
    best, bd = None, 1e9
    for gname, js in groups.items():
        for j in js:
            s = getattr(r, f'{j}_s', np.nan)
            if pd.notna(s) and s >= GATE:
                d = np.hypot(getattr(r, f'{j}_x') - r.ball_x, getattr(r, f'{j}_y') - r.ball_y)
                if d < bd:
                    best, bd = gname, d
    reg.append(best); dist.append(bd if best else np.nan)
cc['pose_region'] = reg; cc['pose_dist_px'] = dist
bh = cc.merge(pl[['frame', 'player', 'y1', 'y2']], on=['frame', 'player'])
bh['dist_rel'] = bh.pose_dist_px / (bh.y2 - bh.y1)
close = bh[bh.dist_rel <= 0.15]
res['H9_pose_contact_region'] = {
    'candidates': int(len(cc)), 'with_pose_region': int(cc.pose_region.notna().sum()),
    'close_to_keypoint(<=15% box h)': int(len(close)),
    'agreement_with_box_band': round(float((close.pose_region == close.coarse_region.map({'head': 'head', 'chest_shoulder': 'chest', 'thigh_hip': 'thigh', 'lower_leg_foot': 'foot'})).mean()), 3),
    'region_by_player_close': close.groupby('player').pose_region.value_counts().unstack(fill_value=0).to_dict('index')}
close.to_csv(D / 'contact_candidates_pose.csv', index=False)

(D / 'analysis_v2.json').write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
