"""Player movement during rallies: set 1 vs set 2 and moves between the 3x3 squares (16 x 8 m court).

Rally frames = data/points/rallies_v2.csv start..end; positions = data/analysis/player_positions_v3.csv; squares as the
player sees them facing the net (scripts/analysis/point_positions.py: depth net / mid / base x lane left / mid / right).
A player is "in" a square after MIN_RUN consecutive frames there (filters jitter on the lines).

outputs: data/analysis/movement_by_set.csv     per player and set: rally time, distance, m per rally-minute, top speed,
                                               sprint share (speed > 3.56 m/s), share of time per depth and lane,
                                               square changes per rally
         data/analysis/square_transitions.csv  per player: moves from one square to another, counted over all rallies
usage  : python scripts/analysis/movement_sets.py"""
import os
import numpy as np, pandas as pd
from point_positions import cell

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_RUN = 5
SPRINT = 3.56
FPS = 25


def main():
    os.chdir(ROOT)
    p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y', 'speed']).dropna()
    p = p.drop_duplicates(['v2_frame', 'player'])
    r = pd.read_csv('data/points/rallies_v2.csv')
    rid = pd.Series(np.nan, index=np.arange(p.v2_frame.max() + 1)); rs = rid.copy()
    for q in r.itertuples():
        rid[q.start_frame:q.end_frame] = q.rally; rs[q.start_frame:q.end_frame] = q.set
    p['rally'] = rid.reindex(p.v2_frame).values; p['set'] = rs.reindex(p.v2_frame).values
    p = p.dropna(subset=['rally']).sort_values(['player', 'v2_frame'])
    c = [cell(x, y) for x, y in zip(p.x, p.y)]
    p['lane'] = [k[1] for k in c]; p['depth'] = [k[2] for k in c]
    p['cellid'] = [f'{k[2]}-{k[1]}' if k[1] else 'out' for k in c]
    step = np.hypot(p.groupby(['player', 'rally']).x.diff(), p.groupby(['player', 'rally']).y.diff()).fillna(0)
    p['step'] = np.where(step < 0.5, step, 0)

    trans, changes = [], []
    for (pl, ra), g in p.groupby(['player', 'rally']):
        runs, cur, n = [], None, 0
        for v in g.cellid.values:
            if v == cur:
                n += 1
            else:
                if cur is not None:
                    runs.append((cur, n))
                cur, n = v, 1
        runs.append((cur, n))
        st = [k for k, n in runs if n >= MIN_RUN]
        st = [k for i, k in enumerate(st) if i == 0 or k != st[i - 1]]
        trans += [dict(player=pl, set=int(g.set.iloc[0]), move_from=a, move_to=b) for a, b in zip(st, st[1:])]
        changes.append(dict(player=pl, set=int(g.set.iloc[0]), changes=max(len(st) - 1, 0)))
    t = pd.DataFrame(trans).groupby(['player', 'move_from', 'move_to']).size().rename('count').reset_index()
    t.sort_values(['player', 'count'], ascending=[True, False]).to_csv('data/analysis/square_transitions.csv', index=False)

    ch = pd.DataFrame(changes).groupby(['player', 'set']).changes.mean()
    rows = []
    for (pl, st), g in p.groupby(['player', 'set']):
        sec = len(g) / FPS
        row = dict(player=pl, set=int(st), rally_sec=round(sec, 1), distance_m=round(g.step.sum(), 1),
                   m_per_rally_min=round(g.step.sum() / sec * 60, 1), top_speed_kmh=round(g.speed.max() * 3.6, 1),
                   sprint_pct=round((g.speed > SPRINT).mean() * 100, 1),
                   square_changes_per_rally=round(ch[(pl, st)], 2))
        for k in ('net', 'mid', 'base'):
            row[f'depth_{k}_pct'] = round((g.depth == k).mean() * 100, 1)
        for k in ('left', 'mid', 'right'):
            row[f'lane_{k}_pct'] = round((g.lane == k).mean() * 100, 1)
        rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv('data/analysis/movement_by_set.csv', index=False)
    pd.set_option('display.width', 250)
    print(m.to_string(index=False))
    print(t.groupby('player').head(4).to_string(index=False))


if __name__ == '__main__':
    main()
