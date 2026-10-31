"""How each of the 60 points ended, from the project owner's review only (no automatic guesses).

Input: data/points/landing_marks_owner.json (the owner's marks from the review page), data/analysis/point_landings.csv
(the landing frame / spot shown in the review clips, used only where the owner marked it correct), the rally winner
(data/points/rallies_v2.csv) and player positions (which team stands on which half).

end_type:
  landed_in   ball came down inside the lines or on a line (within LINE_M) (spot marked by the owner, or the shown spot confirmed by the owner)
  landed_out  ball came down outside the lines (spot marked by the owner)
  out         ball left the court; only the line it crossed is known
  net         ball hit the net and fell back
  fault       net touch by a player / ball passed under the net
  excluded    video does not show the point, unclear, or not reviewed
verified = the owner confirmed it and no check below failed. Checks: a landing inside the lines must be on the losing
team's half; a ball that went out or into the net must have been touched last by the losing team; a confirmed shown spot must be within GAP_MAX px of the detected ball; the winner is the owner's when given
(winner_owner), else the recorded one; a disagreement is noted in check but does not by itself fail the point.

output: data/points/point_end_verified.csv
usage : python scripts/analysis/point_end_verified.py"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_positions import cell, TEAM_OF

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAP_MAX = 80
LINE_M = 0.15   # a marked spot this close outside a line is a line ball (ball on the line is in); click/homography error ~0.1 m


def main():
    os.chdir(ROOT)
    marks = {m['point']: m for m in json.load(open('data/points/landing_marks_owner.json'))['marks']}
    shown = pd.read_csv('data/analysis/point_landings.csv').set_index('rally')
    gaps = pd.read_csv('coach_analysis/review/clips_index.csv').set_index('rally').gap_px
    r = pd.read_csv('data/points/rallies_v2.csv').set_index('rally')
    pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x'])
    rows = []
    for n in range(1, 61):
        m, q = marks[n], r.loc[n]
        winner = m.get('winner_owner') or q.winner; loser = 'ISR' if winner == 'BRA' else 'BRA'
        row = dict(point=n, set=int(q.set), winner=winner, recorded_winner=q.winner, loser=loser, owner_category=m['category'],
                   last_touch=m['last_touch'], out_edge=m['out_edge'], frame=None, x=None, y=None, square=None,
                   lane=None, depth=None, half_team=None, end_type='excluded', verified=False, check='', check_info='', note=m.get('note', ''))
        c = m['category']
        if c in ('not_shown', 'unclear', 'unreviewed'):
            row['check'] = {'not_shown': 'video does not show the point', 'unclear': 'owner: unclear',
                            'unreviewed': 'not reviewed by the owner'}[c]
        elif c in ('net_fault', 'under_net'):
            row.update(end_type='fault', verified=True)
        elif c == 'net_fall':
            row.update(end_type='net', verified=True)
        elif c == 'out' and not (m['spot'] and m['out_edge'] is None):
            row.update(end_type='out', verified=True)
            if m['spot']:
                row['check'] = 'owner chose the line crossed; the clicked spot (inside the lines, image edge) is ignored'
        else:   # ok / relabel: a landing spot
            if m['spot']:
                s = m['spot']; f, x, y = s['frame'], s['court_x'], s['court_y']
            else:
                f, x, y = int(shown.at[n, 'frame']), float(shown.at[n, 'x']), float(shown.at[n, 'y'])
                if gaps.get(n, np.nan) > GAP_MAX:
                    row['check'] = f'confirmed shown spot is {int(gaps[n])} px from the detected ball'
            on_line = sq_out = not (0 <= x <= 16 and 0 <= y <= 8)
            if on_line:
                on_line = (-LINE_M <= x <= 16 + LINE_M) and (-LINE_M <= y <= 8 + LINE_M)
            sq, lane, depth = cell(min(max(x, 0), 16), min(max(y, 0), 8)) if on_line else cell(x, y)
            side = pos[pos.v2_frame.between(f - 12, f + 12)].assign(team=lambda d: d.player.map(TEAM_OF)).groupby('team').x.mean()
            half = (side.idxmin() if x < 8 else side.idxmax()) if len(side) == 2 else None
            row.update(frame=f, x=x, y=y, square=sq, lane=lane, depth=depth, half_team=half,
                       end_type='landed_out' if sq == 'out' else 'landed_in')
            if on_line:
                row['check_info'] = f'line ball: marked {max(-x, -y, x - 16, y - 8):.2f} m outside the line, within LINE_M, counted as in'
            if row['end_type'] == 'landed_in' and half != loser:
                row['check'] = (row['check'] + '; ' if row['check'] else '') + f'landed on the {half} half but {loser} lost the point'
            row['verified'] = not row['check']
        if row['end_type'] in ('out', 'landed_out', 'net') and m['last_touch'] and TEAM_OF[m['last_touch']] != loser:
            row['verified'] = False
            row['check'] = (row['check'] + '; ' if row['check'] else '') + \
                f"ball went out / into the net after {m['last_touch']}'s touch, but {TEAM_OF[m['last_touch']]} is recorded as winning"
        if winner != q.winner:
            row['check'] = (row['check'] + '; ' if row['check'] else '') + f'owner says {winner} won; the score files say {q.winner}'
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv('data/points/point_end_verified.csv', index=False)
    print(d.groupby(['end_type', 'verified']).size())
    print(d[(d.check != '') | (d.check_info != '')][['point', 'owner_category', 'end_type', 'verified', 'check', 'check_info']].to_string(index=False))


if __name__ == '__main__':
    main()
