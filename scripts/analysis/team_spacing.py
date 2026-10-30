"""Distance between the two defenders at every opponent attack (v3 video): when a point was scored against a team,
and as the baseline, when the team defended the attack.

Attacks come from data/analysis/touch_actions_v3.csv (run scripts/analysis/touch_actions.py first). For each attack
the defending team's two players (smoothed positions, nearest frame within NEAR frames): distance, its part along the
net (|dy|) and towards the net (|dx|), each player's 3x3 square. outcome = the attack's subtype:
  kill / failed_defence  the point was scored against the defenders (conceded = True)
  returned               the defenders kept the ball in play
  error                  the attack itself failed
For conceded points a second row is added at the landing: the first in-play "no player" ball event after the rally's
last touch (ball_contacts_v3.csv), else the rally's end frame.

Lengths are court units (court lines = 18 x 9); x 8/9 for metres if the court is 16 x 8 (docs/ANALYSIS_PLAN.md).

inputs : data/analysis/player_positions_v3.csv, data/analysis/touch_actions_v3.csv, data/points/rallies_v3video.csv,
         data/contacts/ball_contacts_v3.csv
output : data/analysis/team_spacing_v3.csv  one row per attack (moment = attack) + one per conceded point (landing)
usage: python scripts/analysis/team_spacing.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
NEAR = 5             # frames: use the nearest position within this many frames when the exact frame has none
TEAMS = {'BRA': ('BRA_A', 'BRA_B'), 'ISR': ('ISR_A', 'ISR_B')}
OTHER = {'BRA': 'ISR', 'ISR': 'BRA'}


def main():
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'), usecols=['frame', 'player', 'x', 'y', 'square'])
    by = {pl: g.set_index('frame') for pl, g in pos.groupby('player')}
    act = pd.read_csv(os.path.join(OUT, 'touch_actions_v3.csv'))
    rallies = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_v3video.csv')).set_index('rally')
    ev = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'ball_contacts_v3.csv'))
    ev = ev[ev.in_play & (ev.event_type != 'touch')]

    def at(pl, f):
        g = by[pl]
        for d in sorted(range(-NEAR, NEAR + 1), key=abs):
            if f + d in g.index:
                return g.loc[f + d]
        return None

    def spacing(team, f):
        a, b = TEAMS[team]
        pa, pb = at(a, int(f)), at(b, int(f))
        if pa is None or pb is None:
            return {}
        dx, dy = pa.x - pb.x, pa.y - pb.y
        return dict(distance=np.hypot(dx, dy), towards_net=abs(dx), along_net=abs(dy), p1_square=pa.square,
                    p2_square=pb.square, p1_x=pa.x, p1_y=pa.y, p2_x=pb.x, p2_y=pb.y)

    rows = []
    for a in act[act.action == 'attack'].itertuples():
        d = OTHER[a.team]
        conceded = a.subtype in ('kill', 'failed_defence')
        base = dict(rally=a.rally, set=a.set, attack_touch=a.touch, attacker=a.player, attacker_part=a.body_part,
                    defending_team=d, outcome=a.subtype, conceded=conceded)
        rows.append(dict(base, moment='attack', frame=a.frame, **spacing(d, a.frame)))
        if conceded:
            last = act[(act.rally == a.rally) & act.in_play & act.touch.notna()].frame.max()
            after = ev[(ev.rally == a.rally) & (ev.frame > last)]
            f, src = (after.frame.iloc[0], 'ball_event') if len(after) else (rallies.loc[a.rally, 'end_frame'],
                                                                             'rally_end')
            rows.append(dict(base, moment='landing', frame=f, landing_source=src, **spacing(d, f)))
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(OUT, 'team_spacing_v3.csv'), index=False)
    s = df[df.moment == 'attack'].dropna(subset=['distance']).groupby(['defending_team', 'outcome']).distance
    print(s.agg(['count', 'median', 'mean']).round(2).to_string())


if __name__ == '__main__':
    main()
