"""Physical profile of every player at their chest touches.

One row per chest touch (the 99 hand-marked ones) joining what the other analyses already measured:
  - body: stature, posture, jump, stance width and its parts (towards / along the net, side-to-side / front-back),
    foot status (planted / moving / raised), player speed on the approach and at the touch (m/s)
  - ball: height at contact (m) and as a share of the player's stature, ball speed coming in (m/s, 3D, over the
    frames before the touch in the same fitted flight) and its descent angle (deg below horizontal)
  - place: distance from the net (m), distance from the nearest sideline (m), court square
  - use: action (reception / set / attack / dead = after the rally ended), for receptions and sets whether the
    own team touched the ball again,
    whether the team won the rally
Then a per-player summary (medians and shares) over the in-play touches.

Lengths are metres on the 16 x 8 m court (net at x = 8).

inputs : data/analysis/touch_actions_v3.csv, chest_stance_v3.csv, touch_speed_v3.csv, ball_height_v2.csv,
         player_positions_v3.csv, player_statures.json, point_landings.csv,
         data/experiments/chest_marks/marks/marks/*.json (v2 frame of each touch)
outputs: data/analysis/chest_profile.csv, data/analysis/chest_profile_by_player.csv
usage: python scripts/analysis/chest_profile.py"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, 'data', 'analysis')
FPS = 25.0
IN_FRAMES = 5        # frames before the touch used for the incoming ball speed


def main():
    acts = pd.read_csv(os.path.join(A, 'touch_actions_v3.csv'))
    acts = acts[acts.touch.notna()].copy()
    acts['touch'] = acts.touch.astype(int)
    stance = pd.read_csv(os.path.join(A, 'chest_stance_v3.csv'))
    speed = pd.read_csv(os.path.join(A, 'touch_speed_v3.csv'))
    ball = pd.read_csv(os.path.join(A, 'ball_height_v2.csv')).set_index('frame')
    pos = pd.read_csv(os.path.join(A, 'player_positions_v3.csv'),
                      usecols=['frame', 'player', 'x', 'y', 'square', 'posture'])
    stat = {k: v['stature_m'] for k, v in json.load(open(os.path.join(A, 'player_statures.json'))).items()}
    winners = pd.read_csv(os.path.join(A, 'point_landings.csv')).set_index('rally').winner
    marks = {int(m['touch']): m for m in
             (json.load(open(f)) for f in glob.glob(os.path.join(ROOT, 'data', 'experiments', 'chest_marks',
                                                                 'marks', 'marks', '*.json')))}

    d = stance.merge(acts[['touch', 'set', 'team', 'action', 'subtype', 'touch_in_possession', 'possession_len']],
                     on='touch', how='left')
    d = d.merge(speed[['touch', 'approach_speed', 'speed_at_touch']], on='touch', how='left')
    d = d.merge(pos.rename(columns={'posture': 'pos_posture'}), on=['frame', 'player'], how='left')

    rows = []
    for r in d.itertuples():
        m = marks.get(r.touch, {})
        f2 = m.get('v2_frame')
        h = v_in = angle = np.nan
        if f2 is not None and f2 in ball.index:
            b = ball.loc[f2]
            h = b.height_m
            # the flight that brings the ball in: the one of the last fitted frame before the touch
            prev = [f2 - k for k in range(IN_FRAMES + 3, 0, -1)
                    if f2 - k in ball.index and ball.loc[f2 - k].method == 'flight_fit']
            prev = [f for f in prev if prev and ball.loc[f].flight_id == ball.loc[prev[-1]].flight_id][-IN_FRAMES:]
            if len(prev) >= 2:
                p0, p1 = ball.loc[prev[0]], ball.loc[prev[-1]]
                dt = (prev[-1] - prev[0]) / FPS
                dxy = np.hypot(p1.court_x_m - p0.court_x_m, p1.court_y_m - p0.court_y_m)
                dz = p1.height_m - p0.height_m
                v_in = np.sqrt(dxy ** 2 + dz ** 2) / dt
                angle = np.degrees(np.arctan2(-dz, dxy))
        last = r.touch_in_possession == r.possession_len
        rows.append(dict(
            touch=r.touch, frame=r.frame, time_sec=r.time_sec, set=r.set, rally=r.rally, team=r.team,
            player=r.player, stature_m=stat.get(r.player),
            action=r.action, subtype=r.subtype,
            kept_ball=(not last) if r.action in ('reception', 'set') else np.nan,
            rally_won=(winners.get(r.rally) == r.team) if r.rally in winners.index else np.nan,
            ball_height_m=round(h, 2) if pd.notna(h) else np.nan,
            ball_height_share=round(h / stat[r.player], 2) if pd.notna(h) and r.player in stat else np.nan,
            ball_speed_in=round(v_in, 1) if pd.notna(v_in) else np.nan,
            ball_descent_deg=round(angle, 0) if pd.notna(angle) else np.nan,
            posture=r.posture, jump=r.jump, reliable=r.reliable,
            stance_m=r.stance, towards_net_m=r.towards_net, along_net_m=r.along_net,
            side_to_side_m=r.side_to_side, front_back_m=r.front_back,
            left_foot=r.left_foot, right_foot=r.right_foot,
            feet_planted=(r.left_foot == 'planted') + (r.right_foot == 'planted'),
            approach_speed=r.approach_speed, speed_at_touch=r.speed_at_touch,
            net_dist_m=round(abs(r.x - 8), 2) if pd.notna(r.x) else np.nan,
            sideline_dist_m=round(min(r.y, 8 - r.y), 2) if pd.notna(r.y) else np.nan,
            square=r.square))
    out = pd.DataFrame(rows).sort_values('touch')
    out.to_csv(os.path.join(A, 'chest_profile.csv'), index=False)

    g = out[out.action != 'dead'].groupby('player')   # in play only
    s = pd.DataFrame({
        'team': g.team.first(),
        'stature_m': g.stature_m.first(),
        'chest_touches': g.size(),
        'receptions': g.action.apply(lambda a: (a == 'reception').sum()),
        'sets': g.action.apply(lambda a: (a == 'set').sum()),
        'attacks': g.action.apply(lambda a: (a == 'attack').sum()),
        'kept_ball_pct': g.kept_ball.apply(lambda k: round(100 * k.dropna().astype(float).mean())),
        'rally_won_pct': g.rally_won.apply(lambda k: round(100 * k.dropna().astype(float).mean())),
        'ball_height_m': g.ball_height_m.median(),
        'ball_height_share': g.ball_height_share.median(),
        'ball_speed_in': g.ball_speed_in.median(),
        'ball_descent_deg': g.ball_descent_deg.median(),
        'stance_m': g.stance_m.median(),
        'towards_net_m': g.towards_net_m.median(),
        'along_net_m': g.along_net_m.median(),
        'side_to_side_m': g.side_to_side_m.median(),
        'front_back_m': g.front_back_m.median(),
        'both_feet_planted_pct': g.feet_planted.apply(lambda k: round(100 * (k == 2).mean())),
        'jump_pct': g.jump.apply(lambda k: round(100 * k.astype(float).mean())),
        'approach_speed': g.approach_speed.median(),
        'speed_at_touch': g.speed_at_touch.median(),
        'net_dist_m': g.net_dist_m.median(),
        'sideline_dist_m': g.sideline_dist_m.median(),
    }).round(2)
    s.to_csv(os.path.join(A, 'chest_profile_by_player.csv'))
    print(s.T.to_string())


if __name__ == '__main__':
    main()
