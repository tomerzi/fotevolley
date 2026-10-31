"""Where the ball ended each of the 60 points (16 x 8 m court).

End of the point = the first time the ball reaches the sand after the last player touch of the rally: a hand-marked
landing (data/experiments/sand_marks/marks/landings/) within the rally, else the first fitted ball frame at most
LOW_M above the sand (data/analysis/ball_height_v2.csv), else the rally end. Ball position = court_x_m / court_y_m
of ball_height_v2.csv at that frame. The half it ended in belongs to the team whose players stand there; the square
is given as that team sees it facing the net (lane left / mid / right x depth net / mid / base), as in
scripts/analysis/point_positions.py.

kind: 'in_court' (inside the lines), 'out' (outside the lines), 'net' (within NET_M of the net line).
landed_on: 'loser' (the ball came down in the losing team's half) or 'winner' (it never got over: net / own side).

output : data/analysis/point_landings.csv
usage  : python scripts/analysis/point_landings.py"""
import glob, json, os
import numpy as np, pandas as pd
from point_positions import cell, TEAM_OF

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOW_M = 0.3
NET_M = 0.4


def main():
    os.chdir(ROOT)
    h = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
    r = pd.read_csv('data/points/rallies_v2.csv')
    fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
    tl = pd.read_csv('data/contacts/touch_labels_final.csv'); tl = tl[tl.kind == 'player'].copy()
    tl['v2'] = tl.frame.map(fm)
    marks = sorted(m['v2_frame'] for m in (json.load(open(f)) for f in glob.glob('data/experiments/sand_marks/marks/landings/*.json'))
                   if m.get('landing') is True)
    pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x'])
    rows = []
    for q in r.itertuples():
        loser = 'ISR' if q.winner == 'BRA' else 'BRA'
        t = tl[(tl.v2 >= q.start_frame) & (tl.v2 <= q.end_frame + 5)]
        t0 = int(t.v2.max()) if len(t) else q.start_frame
        last = t.loc[t.v2.idxmax()] if len(t) else None
        m = [f for f in marks if t0 < f <= q.end_frame + 10]
        if m:
            f, src = m[0], 'hand-marked landing'
        else:
            seg = h.loc[t0 + 1:q.end_frame + 10]
            low = seg[(seg.method == 'flight_fit') & (seg.height_m <= LOW_M)]
            f, src = (int(low.index[0]), 'ball track') if len(low) else (q.end_frame, 'rally end')
        b = h.loc[f] if f in h.index else h.loc[h.index[np.abs(h.index - f).argmin()]]
        x, y = float(b.court_x_m), float(b.court_y_m)
        # which team stands on the half where the ball is
        side_x = pos[pos.v2_frame.between(f - 12, f + 12)].assign(team=lambda d: d.player.map(TEAM_OF)).groupby('team').x.mean()
        half_team = None
        if len(side_x) == 2:
            left_team = side_x.idxmin()
            half_team = left_team if x < 8 else [k for k in side_x.index if k != left_team][0]
        sq, lane, depth = cell(x, y)
        kind = 'net' if abs(x - 8) < NET_M else ('in_court' if sq != 'out' else 'out')
        rows.append(dict(rally=q.rally, set=q.set, winner=q.winner, loser=loser, frame=f, source=src,
                         last_touch_player=None if last is None else last.player,
                         last_touch_part=None if last is None else last.body_part,
                         x=round(x, 2), y=round(y, 2), height_m=round(float(b.height_m), 2), kind=kind,
                         half_team=half_team, landed_on=('loser' if half_team == loser else 'winner') if half_team else None,
                         square=sq, lane=lane, depth=depth))
    d = pd.DataFrame(rows)
    d.to_csv('data/analysis/point_landings.csv', index=False)
    print(d.source.value_counts(), d.kind.value_counts(), pd.crosstab(d.loser, [d.landed_on, d.kind]), sep='\n')


if __name__ == '__main__':
    main()
