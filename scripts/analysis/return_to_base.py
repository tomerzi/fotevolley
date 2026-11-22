"""Return to base after an attack or a set (v3 video).

Base (definition by the coach, 2026-11-22) = the player's mean position in his own view over all measured rally
frames of the match (own view as in phase_heatmaps.py: x' = metres from the own baseline, net at 8; y' across, 0 = the
player's left). Also given per set, since a pair may change its base between sets.

For every in-play attack or set touch by the player: distance from base at the touch, and the time until he is back
within R_BASE m of the base (smoothed positions, player_positions_v3.csv). 'censored' = the rally ended or the
player touched the ball again before he got back (time is then a lower bound: at least that long).

Output: data/analysis/return_to_base.csv (one row per touch), return_to_base_summary.csv (player x action), and the
bases in return_to_base_bases.csv.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS = 25
R_BASE = 1.0
GOOD_SRC = {'ankles', 'ankle', 'planted', 'hips'}
ral = pd.read_csv('data/points/rallies_v3video.csv').set_index('rally')
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[(t.kind == 'player') & t.in_play & t.player.notna()].sort_values('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'player', 'team', 'src', 'occluded', 'edge', 'raw_x', 'x', 'y'])
pos = pos.drop_duplicates(['frame', 'player'])
rid = np.zeros(int(ral.end_frame.max()) + 2, int)
for r, x in ral.iterrows():
    rid[int(x.start_frame):int(x.end_frame) + 1] = r
pos = pos[pos.frame < len(rid)]
pos['rally'] = rid[pos.frame.to_numpy()]
pos = pos[pos.rally > 0]
left = pos.groupby(['rally', 'team']).x.median().lt(8)
L = left.reindex(pd.MultiIndex.from_arrays([pos.rally, pos.team])).to_numpy()
pos['xo'] = np.where(L, pos.x, 16 - pos.x); pos['yo'] = np.where(L, pos.y, 8 - pos.y)
pos['set'] = pos.rally.map(ral.set)
strict = pos.src.isin(GOOD_SRC) & ~pos.occluded & ~pos.edge & pos.raw_x.notna()
base = pos[strict].groupby('player')[['xo', 'yo']].mean()
base_set = pos[strict].groupby(['player', 'set'])[['xo', 'yo']].mean()
bases = pd.concat([base.assign(set='all').reset_index(), base_set.reset_index()])
bases['from_net_m'] = (8 - bases.xo).round(2); bases = bases.round(2)
bases.to_csv('data/analysis/return_to_base_bases.csv', index=False)
P = pos.set_index(['player', 'frame']).sort_index()

rows = []
for r in t[t.action.isin(['attack', 'set'])].itertuples():
    p = r.player; b = base.loc[p]
    end = int(ral.loc[r.rally, 'end_frame'])
    nxt = t[(t.player == p) & (t.frame > r.frame) & (t.rally == r.rally)]
    stop = min(end, int(nxt.frame.iloc[0]) if len(nxt) else end)
    if p not in P.index.get_level_values(0):
        continue
    g = P.loc[p].loc[r.frame:stop]
    if g.empty:
        continue
    d = np.hypot(g.xo - b.xo, g.yo - b.yo).to_numpy()
    back = np.flatnonzero(d <= R_BASE)
    d0 = d[0]
    if d0 <= R_BASE:
        tb, cens = 0.0, False
    elif len(back):
        tb, cens = (g.index[back[0]] - r.frame) / FPS, False
    else:
        tb, cens = (stop - r.frame) / FPS, True
    rows.append(dict(touch=r.touch, frame=r.frame, rally=r.rally, set=r.set, player=p, team=r.team, action=r.action,
                     body_part=r.body_part, dist_from_base_m=round(d0, 2), min_dist_m=round(d.min(), 2),
                     return_s=round(tb, 2), censored=cens, ended_by='rally_end' if stop == end else 'own_touch'))
out = pd.DataFrame(rows)
out.to_csv('data/analysis/return_to_base.csv', index=False)
summ = out.groupby(['player', 'action']).agg(n=('return_s', 'size'), dist_from_base=('dist_from_base_m', 'median'),
                                             return_s_median=('return_s', lambda s: s[~out.loc[s.index, 'censored']].median()),
                                             returned_share=('censored', lambda c: 1 - c.mean())).round(2).reset_index()
summ.to_csv('data/analysis/return_to_base_summary.csv', index=False)
print(bases[bases.set == 'all'].to_string(index=False)); print(summ.to_string(index=False))
