#!/usr/bin/env python3
"""Build canonical, provenance-tagged analysis tables for the single match in this repo.

Read-only on source data. Writes only into footvolley-analysis-v1/derived/.
Run from the repository root:  python3 footvolley-analysis-v1/scripts/build_tables.py

Conventions (verified in data_audit.md):
  * Timebase = `frame` of the cleaned video (25 fps, 0.04 s nominal spacing). All source
    tables used here share it (player_boxes_v9.frame, ball_boxes_corrected3.frame,
    point_events.frame, team_side_ranges.*_frame).
  * Court metres from homography_px_to_m.npy: x along length 0..18 (net at x=9),
    y across 0..9 (y=0 far sideline / top of image, y=9 near sideline / camera side).
  * Team-relative frame: depth = |x-9| (0 at net, 9 at baseline);
    lateral = left-to-right looking toward the net
      LEFT half (facing +x):  lateral = y
      RIGHT half (facing -x): lateral = 9 - y
  * Ball positions are image pixels only. Ball side-of-net is decided by the pixel distance
    to the projected net line with a +-40 px neutral band (ball is airborne; homography is
    NOT applied to the ball).
  * Contact candidates are INFERRED from the 2D ball path (local lowest image point / sharp
    vertical reversal) close to a player box. They are not verified touches.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'footvolley-analysis-v1' / 'derived'
OUT.mkdir(parents=True, exist_ok=True)
FPS = 25.0
NET_BAND_PX = 40
MAX_GAP = 6  # frames allowed between ball observations when reading sides / contacts
OBSERVED_PLAYER_SOURCES_EXCLUDE = {'interpolated'}

H = np.load(ROOT / 'homography_px_to_m.npy', allow_pickle=False)
Hi = np.linalg.inv(H)


def to_px(x, y):
    v = Hi @ np.array([x, y, 1.0])
    return v[:2] / v[2]


NET_FAR, NET_NEAR = to_px(9, 0), to_px(9, 9)


def net_x(py):
    return NET_FAR[0] + (py - NET_FAR[1]) * (NET_NEAR[0] - NET_FAR[0]) / (NET_NEAR[1] - NET_FAR[1])


# ---------------------------------------------------------------- source tables
players = pd.read_csv(ROOT / 'correct_bbox/player_boxes_v9.csv')
ball = pd.read_csv(ROOT / 'correct_bbox/ball_boxes_corrected3.csv')
points = pd.read_csv(ROOT / 'point_events_FINAL_VERIFIED.csv')
sides = pd.read_csv(ROOT / 'team_side_ranges_full_video.csv')

TEAM_COLOR = {'ISR': 'BLUE', 'BRA': 'YELLOW'}  # verified: point_events.country/team, and halves agree


def team_half(frame, team):
    r = sides[(sides.start_frame <= frame) & (sides.end_frame >= frame)].iloc[0]
    return r.blue_side if team == 'ISR' else r.yellow_side


# ---------------------------------------------------------------- players (normalised)
players['observed'] = ~players.source.isin(OBSERVED_PLAYER_SOURCES_EXCLUDE)
players['depth_m'] = (players.court_x - 9).abs()
players['lateral_m'] = np.where(players.court_x < 9, players.court_y, 9 - players.court_y)
players['valid_ground'] = players.court_x.between(-3, 21) & players.court_y.between(-3, 12)
players.to_csv(OUT / 'players_norm.csv', index=False)

# ---------------------------------------------------------------- rallies
rows = []
set_id = 1
prev_frame = -1
prev_winner = None
i = 0
pts = points.sort_values(['frame', 'point_number']).reset_index(drop=True)
while i < len(pts):
    p = pts.iloc[i]
    batch = pts[pts.frame == p.frame]
    n = len(batch)
    last = batch.iloc[-1]
    winner = p.country if batch.country.nunique() == 1 else None
    rows.append({
        'set_id': set_id,
        'rally_id': f's{set_id}_p{int(p.point_number)}' + (f'-{int(last.point_number)}' if n > 1 else ''),
        'point_numbers': ','.join(str(int(x)) for x in batch.point_number),
        'start_frame': prev_frame + 1, 'end_frame': int(p.frame),
        'winner': winner,
        'server_team': prev_winner if (prev_winner and n == 1) else None,
        'isr_half_at_start': team_half(prev_frame + 1, 'ISR'), 'isr_half_at_end': team_half(int(p.frame), 'ISR'),
        'eligible': n == 1,
        'exclude_reason': '' if n == 1 else 'batch scoreboard update: 2 rallies inseparable',
        'score_after': f'ISR {int(last.ISR_score_after)}-BRA {int(last.BRA_score_after)}',
        'label_source': ';'.join(batch.source.unique()),
    })
    prev_frame = int(p.frame)
    prev_winner = winner
    if (batch.event_type == 'SET_WIN').any():
        set_id += 1
        prev_winner = None  # first serve of next set not verified
    i += n
rallies = pd.DataFrame(rows)

# ---------------------------------------------------------------- ball sides & crossings
ball['observed'] = ball.source.isin(['detected', 'rescued', 'manual'])
obs = ball[ball.observed].copy()
obs['d_net_px'] = obs.center_x - net_x(obs.center_y)
obs['side'] = np.where(obs.d_net_px < -NET_BAND_PX, 'LEFT', np.where(obs.d_net_px > NET_BAND_PX, 'RIGHT', 'NET'))

crossings, possessions = [], []
for r in rallies.itertuples():
    w = obs[(obs.frame >= r.start_frame) & (obs.frame <= r.end_frame) & (obs.side != 'NET')]
    seq = []  # runs of (side, first_frame, last_frame, shot)
    for f, s, sh in zip(w.frame, w.side, w.shot):
        if seq and seq[-1][0] == s and f - seq[-1][2] <= 3 * FPS and seq[-1][3] == sh:
            seq[-1][2] = f
        else:
            seq.append([s, f, f, sh])
    # drop 1-2 frame blips (detector noise near the band)
    seq = [s for s in seq if s[2] - s[1] >= 2]
    merged = []
    for s in seq:
        if merged and merged[-1][0] == s[0] and merged[-1][3] == s[3]:
            merged[-1][2] = s[2]
        else:
            merged.append(list(s))
    for k, s in enumerate(merged):
        # half->team mapping taken at the run's own frame (side switches can fall inside a window)
        team = 'ISR' if s[0] == team_half(int(s[1]), 'ISR') else 'BRA'
        nxt = merged[k + 1] if k + 1 < len(merged) else None
        clean_cross = bool(nxt and nxt[3] == s[3] and nxt[1] - s[2] <= MAX_GAP + 4)
        possessions.append({
            'rally_id': r.rally_id, 'set_id': r.set_id, 'poss_idx': k, 'team': team, 'half': s[0],
            'first_frame': s[1], 'last_frame': s[2], 'shot': s[3],
            'next_cross_clean': clean_cross, 'is_last': nxt is None,
        })
poss = pd.DataFrame(possessions)

# rally-level summary of observed play
summ = poss.groupby('rally_id').agg(n_runs=('poss_idx', 'size'), first_team=('team', 'first'),
                                    last_team=('team', 'last'),
                                    clean_crossings=('next_cross_clean', 'sum')).reset_index()
rallies = rallies.merge(summ, on='rally_id', how='left')
rallies.to_csv(OUT / 'rallies.csv', index=False)

# possession outcome (only for possessions inside eligible rallies)
poss = poss.merge(rallies[['rally_id', 'winner', 'eligible', 'server_team']], on='rally_id')



poss['outcome'] = np.where(~poss.is_last, np.where(poss.next_cross_clean, 'returned', 'returned_unclean'),
                           np.where(poss.winner.isna(), 'unknown',
                                    np.where(poss.winner == poss.team, 'terminal_won', 'terminal_lost')))

# ---------------------------------------------------------------- contact candidates
pl = players.set_index('frame')
cands = []
for r in rallies.itertuples():
    w = obs[(obs.frame >= r.start_frame) & (obs.frame <= r.end_frame)].sort_values('frame')
    f = w.frame.to_numpy(); y = w.center_y.to_numpy(); x = w.center_x.to_numpy(); sh = w.shot.to_numpy()
    for j in range(2, len(f) - 2):
        if not (f[j + 2] - f[j - 2] <= MAX_GAP and sh[j - 2] == sh[j + 2]):
            continue
        vy_in = (y[j] - y[j - 2]) / (f[j] - f[j - 2])
        vy_out = (y[j + 2] - y[j]) / (f[j + 2] - f[j])
        # image y grows downward: falling ball has vy>0; a touch sends it up (vy<0)
        if not (vy_in > 2 and vy_out < -2 and y[j] >= max(y[j - 1], y[j + 1])):
            continue
        if f[j] not in pl.index:
            continue
        fr = pl.loc[[f[j]]]
        mx = (fr.x2 - fr.x1) * 0.35
        my = (fr.y2 - fr.y1) * 0.15
        near = fr[(x[j] >= fr.x1 - mx) & (x[j] <= fr.x2 + mx) & (y[j] >= fr.y1 - my) & (y[j] <= fr.y2 + my)]
        if near.empty:
            continue
        cx = (near.x1 + near.x2) / 2
        near = near.assign(dist=np.hypot(x[j] - cx, 0))
        best = near.sort_values('dist').iloc[0]
        ambiguous = len(near) > 1 and (near.sort_values('dist').iloc[1].dist - best.dist) < 15
        rel_h = (y[j] - best.y1) / (best.y2 - best.y1)
        region = ('head' if rel_h < 0.18 else 'chest_shoulder' if rel_h < 0.45 else
                  'thigh_hip' if rel_h < 0.70 else 'lower_leg_foot')
        cands.append({'rally_id': r.rally_id, 'frame': int(f[j]), 'time_s': f[j] / FPS, 'shot': int(sh[j]),
                      'player': best.player, 'team': best.team, 'ball_x': x[j], 'ball_y': y[j],
                      'rel_height_in_box': round(rel_h, 3), 'coarse_region': region,
                      'player_box_source': best.source, 'ambiguous_player': bool(ambiguous),
                      'vy_in': round(vy_in, 2), 'vy_out': round(vy_out, 2),
                      'evidence': 'inferred_2d_ball_reversal_near_box'})
cands = pd.DataFrame(cands)
# dedupe: keep one candidate per 4-frame cluster
cands = cands.sort_values('frame')
cands = cands[cands.frame.diff().fillna(99) > 4]


# attach candidate to possession and order within possession
def poss_of(row):
    m = poss[(poss.rally_id == row.rally_id) & (poss.first_frame - 6 <= row.frame) & (poss.last_frame + 6 >= row.frame)]
    m = m[m.team == row.team]
    return int(m.iloc[0].poss_idx) if len(m) else -1


cands['poss_idx'] = cands.apply(poss_of, axis=1)
cands['order_in_poss'] = cands.groupby(['rally_id', 'poss_idx']).cumcount() + 1
cands.loc[cands.poss_idx < 0, 'order_in_poss'] = 0
cands.to_csv(OUT / 'contact_candidates.csv', index=False)
poss['n_candidate_contacts'] = poss.apply(
    lambda p: int(((cands.rally_id == p.rally_id) & (cands.poss_idx == p.poss_idx)).sum()), axis=1)
poss.to_csv(OUT / 'possessions.csv', index=False)

# ---------------------------------------------------------------- team-shape snapshots
# Anchor A ("incoming"): last observed ball frame on the opponent half before a clean crossing
#   into the team's half -> shape of the receiving/defending pair at the opponent's last action.
snaps = []
for p in poss.itertuples():
    if p.is_last or not p.next_cross_clean:
        continue
    nxt = poss[(poss.rally_id == p.rally_id) & (poss.poss_idx == p.poss_idx + 1)].iloc[0]
    anchor = int(p.last_frame)
    fr = players[(players.frame == anchor) & (players.team == nxt.team)]
    # windows often start mid-rally (cleaned video), so the first incoming ball is NOT a verified serve
    phase = 'first_incoming_in_window' if p.poss_idx == 0 else 'later_incoming'
    for q in fr.itertuples():
        snaps.append({'match_id': 'ISR-BRA', 'set_id': p.set_id, 'rally_id': p.rally_id,
                      'event_id': f'{p.rally_id}_in{p.poss_idx + 1}', 'phase': phase, 'team_id': nxt.team,
                      'player_id': q.player, 'time_s': f'{anchor / FPS:.2f}', 'x_m': f'{q.lateral_m:.3f}',
                      'depth_m': f'{q.depth_m:.3f}', 'valid_ground': str(bool(q.valid_ground)).lower(),
                      'observed': str(bool(q.observed)).lower(), 'anchor_frame': anchor,
                      'incoming_poss_idx': p.poss_idx + 1})
snaps = pd.DataFrame(snaps)
snaps.to_csv(OUT / 'snapshots_incoming.csv', index=False)
snaps.drop(columns=['anchor_frame', 'incoming_poss_idx']).to_csv(OUT / 'coverage_input.csv', index=False)

meta = {
    'fps': FPS, 'net_px_far': NET_FAR.round(1).tolist(), 'net_px_near': NET_NEAR.round(1).tolist(),
    'net_band_px': NET_BAND_PX, 'rallies': len(rallies), 'eligible_rallies': int(rallies.eligible.sum()),
    'possession_runs': len(poss), 'contact_candidates': len(cands), 'snapshot_rows': len(snaps),
}
(OUT / 'build_meta.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
