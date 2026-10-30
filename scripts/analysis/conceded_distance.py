"""Distance between the two players of the team that lost each point, at the moment the point was lost.

Moment of the point:
  - the losing team touched the ball last (own error: out, net, failed dig) -> that last touch;
  - the winning team touched it last -> the first time the ball reaches the sand after that touch:
    a hand-marked landing, else the ball track (not trusted within 1.5 m of the net, where the
    computed height is least reliable);
  - a hand-marked moment from the point-marking page (data/experiments/point_marks/marks/) overrides all.
Points where the moment or the players' positions are doubtful get `uncertain=True` and a reason.
"""
import glob, json, os
import numpy as np, pandas as pd

h = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
r = pd.read_csv('data/points/rallies_v2.csv')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv'); tl = tl[tl.kind == 'player'].copy()
tl['v2'] = tl.frame.map(fm); tl = tl.dropna(subset=['v2']).sort_values('v2'); tl['team'] = tl.player.str[:3]
marks = sorted(m['frame'] for m in (json.load(open(f)) for f in glob.glob('data/experiments/sand_marks/marks/landings/*.json'))
               if m.get('landing') is True)
hand = {}
for f in glob.glob('data/experiments/point_marks/marks/moments/*.json'):
    m = json.load(open(f))
    hand[int(m['rally'])] = m
p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y', 'occluded', 'airborne'])
p['X'] = p.x * 16 / 18; p['Y'] = p.y * 8 / 9
P = p.drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])


def landing(t0, t1):
    m = [f for f in marks if t0 < f <= t1]
    if m and m[0] - t0 <= 75:
        return m[0], 'hand-marked landing'
    seg = h.loc[t0 + 1:t1]
    seg = seg[seg.method == 'flight_fit']
    low = seg[seg.height_m <= 0.3]
    if len(low) and low.index[0] - t0 <= 75 and abs(low.court_x_m.iloc[0] - 8) > 1.5:
        return int(low.index[0]), 'ball track landing'
    return None, None


def partner_distance(f, team):
    for off in (0, -1, 1, -2, 2):  # nearest frame with both players tracked
        try:
            a, b = P.loc[(f + off, team + '_A')], P.loc[(f + off, team + '_B')]
        except KeyError:
            continue
        hidden = bool(a.occluded or b.occluded)  # airborne is normal at a touch; its position is interpolated
        return float(np.hypot(a.X - b.X, a.Y - b.Y)), hidden
    return np.nan, True


rows = []
for q in r.itertuples():
    lose = 'ISR' if q.winner == 'BRA' else 'BRA'
    g = tl[(tl.v2 >= q.start_frame) & (tl.v2 <= q.end_frame)]
    why = []
    f, src = None, None
    if not len(g):
        why.append('no labelled touches')
    else:
        t = int(g.v2.iloc[-1]); team = g.team.iloc[-1]
        gaps = np.diff(g.v2.values)
        if len(gaps) and gaps[-1] > 75:
            why.append(f'last touch {gaps[-1] / 25:.1f} s after the previous one')
        ht = h.height_m.get(t, np.nan)
        if ht == ht and ht < 0.15:
            why.append('last touch at sand level (maybe after the point)')
        if team == lose:
            f, src = t, 'own error (last touch)'
        else:
            f, src = landing(t, q.end_frame + 15)
            if f is None:
                why.append('no clear landing (near the net or hidden)')
    if q.rally in hand and hand[q.rally].get('kind') == 'moment':
        f, src, why = int(hand[q.rally]['frame']), 'hand-marked moment', []
    elif q.rally in hand:
        why.append('marked unclear by hand')
    if f is None:
        f, src = q.end_frame, 'rally end (fallback)'
    d, hidden = partner_distance(f, lose)
    if src != 'hand-marked moment':
        if hidden:
            why.append('a player hidden from the camera')
        if not (1.2 <= d <= 5.5):
            why.append(f'unusual distance {d:.2f} m')
    rows.append(dict(rally=q.rally, set=q.set, loser=lose, score=f'{q.ISR_score_after}-{q.BRA_score_after}',
                     point_frame=f, time_sec=round(f / 25, 2), source=src, distance_m=round(d, 2),
                     uncertain=bool(why), reason='; '.join(why)))
d = pd.DataFrame(rows)
d.to_csv('data/analysis/conceded_partner_distance.csv', index=False)
pd.set_option('display.width', 220)
print(d.to_string(index=False))
print('uncertain:', d.uncertain.sum(), list(d.rally[d.uncertain]))
