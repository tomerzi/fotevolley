"""Where every player stood when each of the 60 points ended, on the 3x3 squares of each half (16 x 8 m court).

Points: only reliable ones, with the on-court winner (verified_point_moments).
Point moment = point_frame of data/analysis/conceded_partner_distance.csv (v2 frames: hand-marked moment, ball on
the sand, or the rally end as a fallback). Position = smoothed position from data/analysis/player_positions_v3.csv
at that v2 frame (nearest frame within NEAR frames when the exact one has none). Squares: the court labels L1-L9 /
R1-R9 (scripts/court/court_squares.py) and the same square as the player sees it facing the net:
lane left / mid / right x depth net / mid / base.

outputs: data/analysis/point_positions.csv  one row per (point, player): won / lost, x, y, square, lane, depth
         data/analysis/point_positions_summary.csv  per player: share of time and counts at won / lost points per cell
usage: python scripts/analysis/point_positions.py"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
TEAM_OF = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
NEAR = 5
C = 8 / 3
LANES, DEPTHS = ('left', 'mid', 'right'), ('net', 'mid', 'base')


def cell(x, y):
    """(square, lane, depth) on the 16 x 8 court, lane / depth as the player sees it facing the net."""
    if not (0 <= x <= 16 and 0 <= y <= 8):
        return 'out', None, None
    left = x < 8
    col = min(int((x if left else x - 8) // C), 2)
    row = min(int(y // C), 2)
    sq = f"{'L' if left else 'R'}{row * 3 + col + 1}"
    # left half: y = 0 (far sideline) is the team's left, column 2 is at the net; right half mirrored
    lane = LANES[row] if left else LANES[2 - row]
    depth = DEPTHS[2 - col] if left else DEPTHS[col]
    return sq, lane, depth


def verified_point_moments(pts):
    """keep only reliable point moments and use the on-court winner (data/points/rallies_on_court.csv): drop points
    the video does not show or the owner could not judge (data/points/point_end_verified.csv, end_type 'excluded')
    and points whose end moment is marked uncertain."""
    oc = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_on_court.csv')).set_index('rally').on_court_winner
    ver = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'point_end_verified.csv')).set_index('point').end_type
    pts = pts[~pts.uncertain.astype(bool) & (pts.rally.map(ver) != 'excluded')].copy()
    pts['loser'] = pts.rally.map(oc).map({'BRA': 'ISR', 'ISR': 'BRA'})
    return pts


def main():
    pts = verified_point_moments(pd.read_csv(os.path.join(OUT, 'conceded_partner_distance.csv')))
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'), usecols=['v2_frame', 'player', 'x', 'y'])
    pos = pos.dropna(subset=['x', 'y'])
    rows = []
    for p in pts.itertuples():
        for pl, team in TEAM_OF.items():
            g = pos[(pos.player == pl) & (pos.v2_frame.between(p.point_frame - NEAR, p.point_frame + NEAR))]
            if len(g):
                r = g.iloc[(g.v2_frame - p.point_frame).abs().argmin()]
                x, y = float(r.x), float(r.y)
                sq, lane, depth = cell(x, y)
            else:
                x = y = np.nan; sq, lane, depth = 'unseen', None, None
            rows.append(dict(rally=p.rally, set=p.set, score=p.score, point_frame=p.point_frame, time_sec=p.time_sec,
                             moment=p.source, player=pl, team=team, result='lost' if team == p.loser else 'won',
                             x=round(x, 2), y=round(y, 2), square=sq, lane=lane, depth=depth))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'point_positions.csv'), index=False)

    allpos = pos.copy()
    c = [cell(x, y) for x, y in zip(allpos.x, allpos.y)]
    allpos['lane'] = [k[1] for k in c]; allpos['depth'] = [k[2] for k in c]
    summ = []
    for pl in TEAM_OF:
        a = allpos[allpos.player == pl]
        for depth in DEPTHS:
            for lane in LANES:
                m = d[(d.player == pl) & (d.lane == lane) & (d.depth == depth)]
                summ.append(dict(player=pl, depth=depth, lane=lane,
                                 time_share=round(((a.lane == lane) & (a.depth == depth)).mean(), 4),
                                 points_won=int((m.result == 'won').sum()), points_lost=int((m.result == 'lost').sum())))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(OUT, 'point_positions_summary.csv'), index=False)
    print(d.square.value_counts().head(), '\n', s.groupby('player')[['points_won', 'points_lost']].sum())


if __name__ == '__main__':
    main()
