"""How each pair stands during rallies and at the end of each point (16 x 8 m court).

Per v2 frame inside a rally (data/points/rallies_v2.csv start..end) and team, from the smoothed positions
(data/analysis/player_positions_v3.csv): distance between the partners, depth gap (difference in distance from the
net), lateral gap (difference across the court), who is closer to the net, and a formation:
  'both_net'   both within NET_ZONE of the net
  'front_back' depth gap larger than the lateral gap (one up, one back)
  'side_side'  lateral gap larger (next to each other)
Point moments come from data/analysis/conceded_partner_distance.csv (point_frame), as in point_positions.py.

outputs: data/analysis/formation_frames.csv  (rally, set, v2_frame, team, ...)
         data/analysis/formation_points.csv  (one row per point and team, won / lost)
usage  : python scripts/analysis/formation.py"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIR = {'BRA': ('Franklin', 'Hiltinho'), 'ISR': ('Maor_Haas', 'Ron_Ben_Ishay')}
NET_ZONE = 8 / 3


def frames(pos, team, sel):
    a, b = PAIR[team]
    w = pos[pos.player.isin(PAIR[team]) & sel].pivot_table(index='v2_frame', columns='player', values=['x', 'y'])
    w = w.dropna()
    na, nb = (w['x'][a] - 8).abs(), (w['x'][b] - 8).abs()
    dgap, lgap = (na - nb).abs(), (w['y'][a] - w['y'][b]).abs()
    f = np.where((na < NET_ZONE) & (nb < NET_ZONE), 'both_net', np.where(dgap > lgap, 'front_back', 'side_side'))
    return pd.DataFrame(dict(v2_frame=w.index, team=team, distance=np.hypot(w['x'][a] - w['x'][b], w['y'][a] - w['y'][b]).round(2).values,
                             depth_gap=dgap.round(2).values, lateral_gap=lgap.round(2).values,
                             front=np.where(na < nb, a, b), formation=f))


def main():
    os.chdir(ROOT)
    pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).dropna()
    pos = pos.drop_duplicates(['v2_frame', 'player'])
    r = pd.read_csv('data/points/rallies_v2.csv')
    rid = pd.Series(np.nan, index=np.arange(pos.v2_frame.max() + 1)); rset = rid.copy()
    for q in r.itertuples():
        rid[q.start_frame:q.end_frame] = q.rally; rset[q.start_frame:q.end_frame] = q.set
    pos['rally'] = rid.reindex(pos.v2_frame).values
    inr = pos.rally.notna()
    fr = pd.concat([frames(pos, t, inr) for t in PAIR])
    fr['rally'] = rid.reindex(fr.v2_frame).values.astype(int); fr['set'] = rset.reindex(fr.v2_frame).values.astype(int)
    fr.to_csv('data/analysis/formation_frames.csv', index=False)

    pts = pd.read_csv('data/analysis/conceded_partner_distance.csv')
    allf = pd.concat([frames(pos, t, pos.v2_frame >= 0) for t in PAIR]).set_index(['team', 'v2_frame'])
    rows = []
    for p in pts.itertuples():
        for t in PAIR:
            g = allf.loc[t]
            g = g[(g.index >= p.point_frame - 5) & (g.index <= p.point_frame + 5)]
            if not len(g):
                continue
            k = g.iloc[np.abs(g.index - p.point_frame).argmin()]
            rows.append(dict(rally=p.rally, set=p.set, team=t, result='lost' if t == p.loser else 'won', **k.to_dict()))
    pd.DataFrame(rows).to_csv('data/analysis/formation_points.csv', index=False)


if __name__ == '__main__':
    main()
