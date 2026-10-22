"""Serve / reception / set / attack for every touch (v3 video), from the touch labels.

A possession is a run of in-play touches by one team (touch_labels_final.csv, kind == player). In a possession of
n touches: n = 3 -> reception, set, attack; n = 2 -> reception, attack; n = 1 -> attack (one-touch return); longer
runs (label problems) -> reception, set..., attack, flagged.
The rally's last possession, if it belongs to the team that lost and has 1-2 touches, is a failed defence
(reception/set, subtype failed) unless the ball then landed at the net or on the other side (ball_contacts_v3.csv,
first in-play no-player event after the last touch; the team's side from player_positions_v3.csv) - then it was
a one/two-touch attack that failed.
reception subtype: after_serve (first possession of the rally), after_attack (defence) or failed.
attack subtype: kill (no touch by the other team after it), failed_defence (the other team touched it 1-2 times and
lost the rally), error (this team lost the rally on this attack), returned (play went on).

The serve is almost never in this video (the main camera picks the rally up at the reception). A first touch by
the serving team (winner of the previous rally) that is a lone foot touch is the serve; otherwise an inferred serve
row is added at the rally start (team = previous winner; in rally 1 the team that did not receive), player unknown.
Out-of-play touches (pre-serve juggling, after the ball died) get action 'dead'.

inputs : data/contacts/touch_labels_final.csv, data/points/rallies_v3video.csv
output : data/analysis/touch_actions_v3.csv
usage: python scripts/analysis/touch_actions.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
FPS = 25
OTHER = {'BRA': 'ISR', 'ISR': 'BRA'}


def possession_actions(n):
    if n == 1:
        return ['attack']
    if n == 2:
        return ['reception', 'attack']
    return ['reception'] + ['set'] * (n - 2) + ['attack']


def main():
    t = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'touch_labels_final.csv'))
    t = t[t.kind == 'player'].copy()
    t['team'] = t.player.str[:3]
    rallies = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_v3video.csv'))
    prev_winner = dict(zip(rallies.rally, rallies.winner.shift(1)))
    ev = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'ball_contacts_v3.csv'))
    ev = ev[ev.in_play & (ev.event_type != 'touch')]
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'), usecols=['frame', 'team', 'x'])
    rows = []
    for r in rallies.itertuples():
        tr = t[t.rally == r.rally].sort_values('frame')
        play = tr[tr.in_play]
        for d in tr[~tr.in_play].itertuples():
            rows.append(dict(touch=d.touch, frame=d.frame, rally=r.rally, set=r.set, player=d.player, team=d.team,
                             body_part=d.body_part, in_play=False, action='dead'))
        server = prev_winner[r.rally] if isinstance(prev_winner[r.rally], str) else \
            (OTHER[play.team.iloc[0]] if len(play) else None)
        runs = (play.team != play.team.shift()).cumsum().values
        poss = [g for _, g in play.groupby(runs, sort=False)]
        serve_seen = bool(poss) and poss[0].team.iloc[0] == server and len(poss[0]) == 1 and \
            poss[0].body_part.iloc[0] == 'foot'
        if not serve_seen and server is not None:
            rows.append(dict(touch=np.nan, frame=r.start_frame, rally=r.rally, set=r.set, player=None, team=server,
                             body_part=None, in_play=True, action='serve', subtype='inferred', inferred=True,
                             possession=0, touch_in_possession=1, possession_len=1))
        failed_defence = False
        if poss and poss[-1].team.iloc[0] != r.winner and len(poss[-1]) <= 2 and len(poss) > 1:
            lt = poss[-1].team.iloc[0]
            x = pos[pos.frame.between(r.start_frame, r.end_frame) & (pos.team == lt)].x
            own = 'LEFT' if (x < 9).mean() > 0.5 else 'RIGHT'
            land = ev[(ev.rally == r.rally) & (ev.frame > play.frame.iloc[-1])].ball_side
            failed_defence = not len(land) or land.iloc[0] == own
        for k, g in enumerate(poss):
            acts = ['serve'] if serve_seen and k == 0 else possession_actions(len(g))
            last = k == len(poss) - 1
            if last and failed_defence:
                acts = ['reception', 'set'][:len(g)]
            for i, (d, a) in enumerate(zip(g.itertuples(), acts)):
                sub = None
                if a == 'reception':
                    sub = 'failed' if last and failed_defence else \
                        'after_serve' if k == (1 if serve_seen else 0) else 'after_attack'
                elif a == 'set' and last and failed_defence:
                    sub = 'failed'
                elif a == 'attack':
                    if last:
                        sub = 'kill' if d.team == r.winner else 'error'
                    elif k == len(poss) - 2 and g.team.iloc[0] == r.winner:
                        sub = 'failed_defence' if failed_defence else 'returned'
                    else:
                        sub = 'returned'
                rows.append(dict(touch=d.touch, frame=d.frame, rally=r.rally, set=r.set, player=d.player,
                                 team=d.team, body_part=d.body_part, in_play=True, action=a, subtype=sub,
                                 inferred=False, possession=k + 1, touch_in_possession=i + 1, possession_len=len(g),
                                 check=len(g) > 3 or (k == 0 and not serve_seen and d.team == server)))
    df = pd.DataFrame(rows).sort_values(['frame', 'touch'], na_position='first')
    df['time_sec'] = (df.frame / FPS).round(2)
    df['inferred'] = df.inferred.fillna(False).astype(bool)
    df['check'] = df.check.fillna(False).astype(bool)
    cols = ['touch', 'frame', 'time_sec', 'rally', 'set', 'player', 'team', 'body_part', 'in_play', 'action', 'subtype',
            'inferred', 'possession', 'touch_in_possession', 'possession_len', 'check']
    df[cols].to_csv(os.path.join(OUT, 'touch_actions_v3.csv'), index=False)
    print(df.groupby(['action', 'subtype'], dropna=False).size().to_string())
    print('serves seen:', int(((df.action == 'serve') & ~df.inferred).sum()), ' to check:', int(df.check.sum()))
    print(df[df.action != 'dead'].groupby('action').body_part.value_counts().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
