"""Player speed per rally and going into each touch (v3 video), from data/analysis/player_positions_v3.csv.

Per rally (rallies_v3video.csv start..end frame) and player: frames tracked, distance covered (sum of the smoothed
steps inside a track), mean speed, top speed (highest TOP_WIN-frame average, so one noisy frame cannot set it), time
above SPRINT speed, number of jump frames.
Per touch (touch_labels_final.csv, kind == player): the toucher's mean speed over the APPROACH frames before it and
their speed at the touch frame.

Lengths are court units (court lines = 18 x 9); x 8/9 for metres if the court is 16 x 8 (docs/ANALYSIS_PLAN.md).

inputs : data/analysis/player_positions_v3.csv, data/points/rallies_v3video.csv, data/contacts/touch_labels_final.csv
outputs: data/analysis/player_speed_rallies_v3.csv, data/analysis/touch_speed_v3.csv
usage: python scripts/analysis/player_speed.py"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
FPS = 25
TOP_WIN = 12         # frames (~0.5 s) averaged for the top speed
SPRINT = 4.0         # units/s counted as sprinting
APPROACH = 12        # frames before a touch for the approach speed


def main():
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'),
                      usecols=['frame', 'player', 'team', 'track', 'jump', 'x', 'y', 'speed'])
    rallies = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_v3video.csv'))
    pos = pos.sort_values(['player', 'frame'])
    same = (pos.track == pos.track.shift()) & (pos.frame.diff() == 1)
    pos['step'] = np.where(same, np.hypot(pos.x.diff(), pos.y.diff()), 0.0)
    pos['top'] = pos.groupby('track').speed.transform(lambda s: s.rolling(TOP_WIN, min_periods=TOP_WIN).mean())
    rows = []
    for r in rallies.itertuples():
        seg = pos[pos.frame.between(r.start_frame, r.end_frame)]
        for pl, g in seg.groupby('player'):
            rows.append(dict(rally=r.rally, set=r.set, player=pl, team=pl[:3], winner=r.winner,
                             frames=len(g), rally_frames=r.end_frame - r.start_frame + 1,
                             distance=g.step.sum(), mean_speed=g.speed.mean(), top_speed=g.top.max(),
                             sprint_sec=(g.speed > SPRINT).sum() / FPS, jump_frames=int(g.jump.sum())))
    ral = pd.DataFrame(rows)
    ral.round(3).to_csv(os.path.join(OUT, 'player_speed_rallies_v3.csv'), index=False)

    t = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'touch_labels_final.csv'))
    t = t[t.kind == 'player']
    by = {pl: g.set_index('frame') for pl, g in pos.groupby('player')}
    out = []
    for r in t.itertuples():
        g = by.get(r.player)
        if g is None:
            continue
        before = g.speed.reindex(range(r.frame - APPROACH, r.frame))
        out.append(dict(touch=r.touch, frame=r.frame, rally=r.rally, in_play=r.in_play, player=r.player,
                        body_part=r.body_part, approach_speed=before.mean(), approach_frames=int(before.notna().sum()),
                        speed_at_touch=g.speed.get(r.frame, np.nan)))
    pd.DataFrame(out).round(3).to_csv(os.path.join(OUT, 'touch_speed_v3.csv'), index=False)

    tot = ral.groupby('player').agg(distance=('distance', 'sum'), top_speed=('top_speed', 'max'),
                                    mean_speed=('mean_speed', 'mean'), sprint_sec=('sprint_sec', 'sum'))
    print(tot.round(2).to_string())


if __name__ == '__main__':
    main()
