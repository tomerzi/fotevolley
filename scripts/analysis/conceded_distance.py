"""Distance between the two players of the team that lost each point, at the moment the point was lost:
the first time the ball reaches the sand after the last player touch of the rally (hand-marked landing
when available, else the ball-height track), not later when a player has already run to the ball."""
import glob, json
import numpy as np, pandas as pd

h = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
r = pd.read_csv('data/points/rallies_v2.csv')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv'); tl = tl[tl.kind == 'player'].copy()
tl['v2'] = tl.frame.map(fm); tl = tl.dropna(subset=['v2']).sort_values('v2'); tl['team'] = tl.player.str[:3]
marks = sorted(json.load(open(f))['frame'] for f in glob.glob('data/experiments/sand_marks/marks/landings/*.json')
               if json.load(open(f)).get('landing') is True)
p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y', 'occluded', 'airborne'])
p['X'] = p.x * 16 / 18; p['Y'] = p.y * 8 / 9
P = p.drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])


def landing(t0, t1):
    m = [f for f in marks if t0 < f <= t1]
    if m:
        return m[0], 'hand mark'
    seg = h.loc[t0 + 1:t1]
    seg = seg[seg.method == 'flight_fit']
    low = seg[seg.height_m <= 0.3]
    if len(low):
        return int(low.index[0]), 'ball track'
    if len(seg):
        return int(seg.index[-1]), 'end of ball track'
    return None, 'none'


rows = []
for q in r.itertuples():
    lose = 'ISR' if q.winner == 'BRA' else 'BRA'
    g = tl[(tl.v2 >= q.start_frame) & (tl.v2 <= q.end_frame)]
    t_last = int(g.v2.iloc[-1]) if len(g) else q.start_frame
    last_team = g.team.iloc[-1] if len(g) else ''
    f, src = landing(t_last, q.end_frame + 15)
    if f is None:
        f, src = q.end_frame, 'rally end'
    d, flag = np.nan, True
    for off in (0, -1, 1, -2, 2, -3, 3):  # nearest frame with both players tracked
        try:
            a, b = P.loc[(f + off, lose + '_A')], P.loc[(f + off, lose + '_B')]
        except KeyError:
            continue
        d = float(np.hypot(a.X - b.X, a.Y - b.Y)); flag = bool(a.occluded or b.occluded or a.airborne or b.airborne or off != 0)
        break
    rows.append(dict(rally=q.rally, set=q.set, loser=lose, score=f'{q.ISR_score_after}-{q.BRA_score_after}',
                     last_touch=t_last, last_touch_team=last_team, point_frame=f, time_sec=round(f / 25, 2),
                     source=src, distance_m=round(d, 2), uncertain=flag))
d = pd.DataFrame(rows)
d.to_csv('data/analysis/conceded_partner_distance.csv', index=False)
pd.set_option('display.width', 200)
print(d.to_string(index=False))
print(d.groupby('loser').distance_m.agg(['median', 'mean', 'min', 'max', 'count']).round(2))
print(d.source.value_counts())
