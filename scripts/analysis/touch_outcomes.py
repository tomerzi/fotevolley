"""Outcome of every touch, the touch that won / lost each point, set quality (v3 video).

Inputs (all hand-checked except the action, which is inferred from the touch order):
  data/contacts/touches_master.csv     touch, player, body part (hand labels), action, possession numbering
  data/points/point_end_verified.csv   how each point ended, last touch, on-court winner (owner review, 56 of 60)
  data/analysis/ball_height_v2.csv     ball height per v2 frame (estimate, height_sigma_m per frame)
  data/analysis/player_positions_v3.csv

Success of a touch (definition by the coach, 2026-11-22): a touch is successful if the ball was not lost by it, i.e.
it advanced the play: a teammate touched next, or the ball went to the opponent's side and the point did not end on
it. A touch fails when it is the last touch of the point and the toucher's team lost the point (own error: out, net,
fault, ball dropped on the own side, a defence that could not keep the ball). The 4 points the video does not show
are left out (outcome unknown for their last touch).
Quality levels, for the coach's finer view:
  reception: 'to_attack' = the possession reached an attack, 'kept' = kept but no attack (a free ball over),
             'lost'
  set:       'to_attack' = the next touch is the partner's attack, 'kept', 'lost'
  attack:    'kill' = point won on it, 'returned' = the opponent kept the ball, 'lost' = own error
Set quality: air time to the next touch (s), the ball's highest point in between (m, estimate) and how far the
partner had to move (m) from the set to his touch.

Outputs data/analysis/touch_outcomes.csv (one row per in-play touch), touch_success_summary.csv (player x action x
body part: n, success rate), point_touches.csv (per point: the winning / losing touch, player, body part, touch in
possession, how it ended), set_quality.csv.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS = 25
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[(t.kind == 'player') & t.in_play & t.player.notna()].sort_values(['rally', 'frame']).reset_index(drop=True)
e = pd.read_csv('data/points/point_end_verified.csv').rename(columns={'point': 'rally'}).set_index('rally')
bh = pd.read_csv('data/analysis/ball_height_v2.csv', usecols=['frame', 'height_m', 'height_sigma_m']).set_index('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'player', 'x', 'y']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])

t['next_player'] = t.groupby('rally').player.shift(-1)
t['next_team'] = t.groupby('rally').team.shift(-1)
t['next_frame'] = t.groupby('rally').frame.shift(-1)
t['next_v2'] = t.groupby('rally').v2_frame.shift(-1)
t['next_action'] = t.groupby('rally').action.shift(-1)
t['last'] = t.next_player.isna()
t['shown'] = t.rally.map(e.end_type) != 'excluded'
t['point_winner'] = t.rally.map(e.winner)
t['end_type'] = t.rally.map(e.end_type)
t['lost'] = t['last'] & t.shown & (t.point_winner != t.team)
t['won_point'] = t['last'] & t.shown & (t.point_winner == t.team)
t['success'] = np.where(t['last'] & ~t.shown, np.nan, (~t.lost).astype(float))
# possession reached an attack?
t['poss_key'] = t.rally.astype(str) + '_' + t.possession.astype(str)
att = t[t.action == 'attack'].poss_key.unique()
def quality(r):
    if r['last'] and not r.shown:
        return 'unknown'
    if r.lost:
        return 'lost'
    if r.action == 'attack':
        return 'kill' if r.won_point else 'returned'
    if r.action == 'reception':
        return 'to_attack' if r.poss_key in att else 'kept'
    if r.action == 'set':
        return 'to_attack' if (r.next_team == r.team and r.next_action == 'attack') else 'kept'
    return 'kept'
t['quality'] = t.apply(quality, axis=1)

# ---------- set quality: every touch passed to the partner ----------
sq = []
for r in t[(t.next_team == t.team) & (t.next_player != t.player)].itertuples():
    f0, f1 = int(r.v2_frame), int(r.next_v2)
    h = bh.loc[f0:f1]
    p0 = pos.loc[(r.frame, r.next_player)] if (r.frame, r.next_player) in pos.index else None
    p1 = pos.loc[(r.next_frame, r.next_player)] if (r.next_frame, r.next_player) in pos.index else None
    sq.append(dict(touch=r.touch, rally=r.rally, set=r.set, player=r.player, team=r.team, action=r.action,
                   body_part=r.body_part, to=r.next_player, next_action=r.next_action, quality=r.quality,
                   air_time_s=round((r.next_frame - r.frame) / FPS, 2),
                   apex_m=round(float(h.height_m.max()), 2) if len(h) else np.nan,
                   apex_sigma_m=round(float(h.height_sigma_m.loc[h.height_m.idxmax()]), 2) if len(h) and h.height_m.notna().any() else np.nan,
                   partner_move_m=round(float(np.hypot(p1.x - p0.x, p1.y - p0.y)), 2) if p0 is not None and p1 is not None else np.nan))
sq = pd.DataFrame(sq)

# ---------- point touches ----------
pt = []
for rally, g in t.groupby('rally'):
    l = g.iloc[-1]
    if not l.shown:
        continue
    pt.append(dict(rally=rally, set=l.set, winner=l.point_winner, end_type=l.end_type,
                   owner_category=e.loc[rally, 'owner_category'],
                   last_player=l.player, last_team=l.team, body_part=l.body_part, side=l.limb_side, action=l.action,
                   touch_in_possession=l.touch_in_possession, possession_len=l.possession_len,
                   kind='winning_touch' if l.won_point else 'losing_touch',
                   matches_owner_last_touch=l.player == e.loc[rally, 'last_touch'],
                   x=e.loc[rally, 'x'], y=e.loc[rally, 'y'], square=e.loc[rally, 'square']))
pt = pd.DataFrame(pt)

summ = (t[t.success.notna()].groupby(['player', 'action', 'body_part'])
        .agg(n=('success', 'size'), success_rate=('success', 'mean'), lost=('lost', 'sum')).reset_index())
summ['success_rate'] = summ.success_rate.round(3)
summ_q = t.groupby(['player', 'action', 'body_part', 'quality']).size().unstack(fill_value=0).reset_index()
summ = summ.merge(summ_q, on=['player', 'action', 'body_part'], how='left')

cols = ['touch', 'frame', 'v2_frame', 'time_sec', 'rally', 'set', 'player', 'team', 'body_part', 'limb_side', 'action',
        'action_subtype', 'possession', 'touch_in_possession', 'possession_len', 'next_player', 'last', 'shown',
        'lost', 'won_point', 'success', 'quality', 'end_type']
t[cols].to_csv('data/analysis/touch_outcomes.csv', index=False)
summ.to_csv('data/analysis/touch_success_summary.csv', index=False)
pt.to_csv('data/analysis/point_touches.csv', index=False)
sq.to_csv('data/analysis/set_quality.csv', index=False)

print('touches', len(t), 'unknown', int(t.success.isna().sum()), 'lost', int(t.lost.sum()))
print('last toucher agrees with the owner review:', int(pt.matches_owner_last_touch.sum()), '/', len(pt))
print(pt[~pt.matches_owner_last_touch][['rally', 'last_player', 'end_type', 'owner_category']].to_string())
print(t[t.success.notna()].groupby(['player', 'action']).success.agg(['size', 'mean']).round(2))
r = t[t.action == 'reception']
print(pd.crosstab([r.player, r.body_part], r.quality))
print(pt.groupby(['kind', 'last_team', 'body_part']).size())
print(pd.crosstab([pt[pt.kind == 'losing_touch'].last_player], [pt[pt.kind == 'losing_touch'].touch_in_possession]))
print(sq.groupby(['player', 'body_part']).agg(n=('air_time_s', 'size'), air=('air_time_s', 'median'), apex=('apex_m', 'median'), move=('partner_move_m', 'median')).round(2))
