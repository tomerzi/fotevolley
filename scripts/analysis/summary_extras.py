"""Metrics the coach summary needs that the pipeline did not have yet.

1. ball_after_touch.csv   every in-play touch: highest ball point between this touch and the next one (apex), the
                          time to the next touch (air time), and who touches next.
2. set_partner.csv        every set: where the partner stands (distance, towards the net / along the net from the setter),
                          how fast he moves, and which way he faces (angle to the net, angle between his facing and
                          the direction to the ball).
3. set_foot_side.csv      every foot set: where the ball (= the foot) is relative to the setter's body, in his own
                          frame: right (+) / left (-) of the body centre and in front (+) / behind (-), metres.
                          side = left / middle / right with a 0.15 m middle band.

Positions: data/analysis/player_positions_v3.csv (v3 frames); facing: data/analysis/body_direction_full_match.csv
(v2 frames, shoulders first, then knees, then feet; facing unit vector fx, fy; right-hand side = (-fy, fx), as in
body_rays.py); ball: data/analysis/ball_height_v2.csv (v2 frames) and the touch table's ball court point.
usage: python scripts/analysis/summary_extras.py
"""
import numpy as np
import pandas as pd

OUT = 'data/analysis/'
MIDDLE = 0.15

t = pd.read_csv('data/analysis/touch_outcomes.csv')
m = pd.read_csv('data/contacts/touches_master.csv', usecols=['touch', 'ball_height_m', 'ball_court_x_m', 'ball_court_y_m'])
t = t.merge(m, on='touch', how='left').sort_values('frame').reset_index(drop=True)
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv').query('~deleted').set_index('v3_frame').v2_frame
t['v2_frame'] = t.v2_frame.fillna(t.frame.map(fm))   # the owner's added touch (v3 12700) has no v2 frame
bh = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'player', 'x', 'y', 'speed', 'posture'])
pos = pos.set_index(['frame', 'player']).sort_index()
bd = pd.read_csv('data/analysis/body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'fx', 'fy', 'angle_to_net_deg'])
bd = bd.groupby(['frame', 'player', 'part'])[['fx', 'fy', 'angle_to_net_deg']].mean()
PARTNER = {'Franklin': 'Hiltinho', 'Hiltinho': 'Franklin', 'Maor_Haas': 'Ron_Ben_Ishay', 'Ron_Ben_Ishay': 'Maor_Haas'}


def facing(v2, player, win=2):
    """Facing unit vector and angle to the net near v2 frame (shoulders > knees > feet)."""
    for part in ('shoulders', 'knees', 'feet'):
        rows = [bd.loc[(f, player, part)] for f in range(int(v2) - win, int(v2) + win + 1) if (f, player, part) in bd.index]
        if rows:
            r = pd.DataFrame(rows).mean()
            v = np.array([r.fx, r.fy]); n = np.linalg.norm(v)
            if n > 1e-6:
                return v / n, r.angle_to_net_deg, part
    return None, np.nan, None


def where(frame, player, win=2):
    for d in range(win + 1):
        for f in (frame - d, frame + d):
            if (f, player) in pos.index:
                return pos.loc[(f, player)]
    return None


# 1. ball after each touch
rows = []
for i, r in t.iterrows():
    nxt = t.iloc[i + 1] if i + 1 < len(t) and t.rally[i + 1] == r.rally else None
    v0 = int(r.v2_frame)
    v1 = int(nxt.v2_frame) if nxt is not None else v0 + 75
    seg = bh.loc[v0 + 1:v1 - 1] if v1 - v0 > 2 else bh.loc[v0:v0]
    k = seg.height_m.idxmax() if len(seg) else None
    rows.append(dict(touch=int(r.touch), rally=r.rally, set=r.set, player=r.player, team=r.team, action=r.action,
                     body_part=r.body_part, height_at_touch_m=r.ball_height_m,
                     apex_m=round(seg.height_m.max(), 2) if k is not None else np.nan,
                     apex_sigma_m=round(seg.loc[k, 'height_sigma_m'], 2) if k is not None else np.nan,
                     rise_m=round(seg.height_m.max() - r.ball_height_m, 2) if k is not None else np.nan,
                     air_time_s=round((nxt.frame - r.frame) / 25, 2) if nxt is not None else np.nan,
                     next_player=nxt.player if nxt is not None else None,
                     next_same_team=(nxt.team == r.team) if nxt is not None else None, success=r.success))
ba = pd.DataFrame(rows)
ba.to_csv(OUT + 'ball_after_touch.csv', index=False)

# 2. partner at each set, 3. foot side at foot sets
sp, fs = [], []
for r in t[t.action == 'set'].itertuples():
    me, pa = where(int(r.frame), r.player), where(int(r.frame), PARTNER[r.player])
    if me is None:
        continue
    net_dir = 1 if me.x < 8 else -1   # +x towards the net for a player on the left half
    if pa is not None:
        dx, dy = pa.x - me.x, pa.y - me.y
        f, ang, part = facing(r.v2_frame, PARTNER[r.player])
        to_ball = np.array([r.ball_court_x_m - pa.x, r.ball_court_y_m - pa.y]) if pd.notna(r.ball_court_x_m) else None
        look = np.nan
        if f is not None and to_ball is not None and np.linalg.norm(to_ball) > 0.2:
            look = np.degrees(np.arccos(np.clip(f @ to_ball / np.linalg.norm(to_ball), -1, 1)))
        sp.append(dict(touch=r.touch, rally=r.rally, set=r.set, setter=r.player, partner=PARTNER[r.player], team=r.team,
                       body_part=r.body_part, quality=r.quality, success=r.success,
                       distance_m=round(np.hypot(dx, dy), 2), partner_towards_net_m=round(dx * net_dir, 2),
                       partner_along_net_m=round(abs(dy), 2), partner_speed_ms=round(pa.speed, 2),
                       partner_moving=bool(pa.speed > 1.0), partner_angle_to_net_deg=round(ang, 1) if pd.notna(ang) else np.nan,
                       partner_facing_vs_ball_deg=round(look, 1) if pd.notna(look) else np.nan, facing_from=part))
    if r.body_part == 'foot' and pd.notna(r.ball_court_x_m):
        f, ang, part = facing(r.v2_frame, r.player)
        if f is None:
            continue
        v = np.array([r.ball_court_x_m - me.x, r.ball_court_y_m - me.y])
        right = np.array([-f[1], f[0]])
        lat, fwd = v @ right, v @ f
        fs.append(dict(touch=r.touch, rally=r.rally, set=r.set, player=r.player, team=r.team, foot=r.limb_side,
                       quality=r.quality, success=r.success, right_of_body_m=round(lat, 2), in_front_m=round(fwd, 2),
                       side='middle' if abs(lat) < MIDDLE else ('right' if lat > 0 else 'left'),
                       height_m=r.ball_height_m, facing_from=part))
sp, fs = pd.DataFrame(sp), pd.DataFrame(fs)
sp.to_csv(OUT + 'set_partner.csv', index=False)
fs.to_csv(OUT + 'set_foot_side.csv', index=False)

print('ball after touch', len(ba))
print(ba.groupby(['player', 'action'])[['apex_m', 'air_time_s']].median().round(2))
print('set partner', len(sp))
print(sp.groupby('partner')[['distance_m', 'partner_towards_net_m', 'partner_speed_ms', 'partner_angle_to_net_deg',
                             'partner_facing_vs_ball_deg']].median().round(2))
print('foot sets', len(fs))
print(pd.crosstab([fs.player, fs.foot], fs.side))
