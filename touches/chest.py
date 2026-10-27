"""Chest touches and the distance between the player's feet at that moment.

There is no labelled touch data in the project, so touches are detected:
  * the ball is on the player's chest in the picture: within 0.6 of the
    torso length of the chest point (35 % of the way from neck to hip),
  * the ball is moving (median image speed > 6 px/frame over +-4 frames),
    which drops a ball held in the hands before a serve,
  * no hand is on the ball (both confident wrists > 0.25 torso lengths away)
    and the ball is on the upper torso, not at the hips,
  * the ball changes flight within 3 frames (ball_height/ball_height.csv),
  * the ball is a real detection, not an interpolated box,
  * in 3-D the ball is at chest height (0.72 x stature +- 0.45 m) and
    within 1.2 m of the player's feet on the sand; if two players qualify
    in the same frame, the nearer one keeps the touch.
Consecutive candidate frames of one player form one touch; the frame
closest to the chest is the touch frame.
Feet distance: both ankles mapped to the sand with the calibrated camera.
A foot clearly in the air (ankle far above the other) is flagged, because
then the sand projection is not a foot position.
Writes touches/chest_touches.csv.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'ball_height')
from traj import Camera

cam = Camera()
STATURE = {k: v['stature_m'] for k, v in json.load(open('ball_height/players.json')).items()}
spd = pd.read_csv('player_speed/player_speed.csv', usecols=['frame', 'player', 'court_x_m', 'court_y_m'])
P = pd.read_csv('pose/vitpose_body25.csv')
b = pd.read_csv('correct_bbox/ball_boxes_corrected3.csv')
h = pd.read_csv('ball_height/ball_height.csv')
b['size'] = np.sqrt((b.x2 - b.x1) * (b.y2 - b.y1))

fl = h.flight_id.values
brk = np.zeros(len(h), bool)
brk[1:] = fl[1:] != fl[:-1]
step = np.hypot(b.center_x.diff(), b.center_y.diff())
moving = (step.rolling(9, center=True, min_periods=3).median() > 6).values
near_break = pd.Series(brk).rolling(7, center=True, min_periods=1).max().astype(bool).values

m = P.merge(b[['frame', 'center_x', 'center_y', 'size', 'source']], on='frame')
m = m[m.source.isin(['detected', 'rescued', 'manual'])]
cx = m.neck_x + 0.35 * (m.mid_hip_x - m.neck_x)
cy = m.neck_y + 0.35 * (m.mid_hip_y - m.neck_y)
torso = np.hypot(m.mid_hip_x - m.neck_x, m.mid_hip_y - m.neck_y)
m['d_chest'] = np.hypot(m.center_x - cx, m.center_y - cy) / torso
m['torso'] = torso
wr = np.fmin(np.where(m.r_wrist_s > 0.4, np.hypot(m.center_x - m.r_wrist_x, m.center_y - m.r_wrist_y), np.inf),
             np.where(m.l_wrist_s > 0.4, np.hypot(m.center_x - m.l_wrist_x, m.center_y - m.l_wrist_y), np.inf)) / torso
along = ((m.center_x - m.neck_x) * (m.mid_hip_x - m.neck_x) + (m.center_y - m.neck_y) * (m.mid_hip_y - m.neck_y)) / torso ** 2
ok = (wr > 0.25) & (along < 0.7) & (m.neck_s > 0.5) & (m.mid_hip_s > 0.5) & (m.d_chest < 0.6) & near_break[m.frame.values] & moving[m.frame.values]
c = m[ok].merge(spd, on=['frame', 'player']).merge(h[['frame', 'height_m', 'court_x_m', 'court_y_m']].rename(
    columns={'court_x_m': 'bx', 'court_y_m': 'by'}), on='frame')
c['chest_h'] = 0.72 * c.player.map(STATURE)
c['dxy'] = np.hypot(c.bx - c.court_x_m, c.by - c.court_y_m)
c = c[((c.height_m - c.chest_h).abs() < 0.45) & (c.dxy < 1.2)]
c = c.loc[c.groupby('frame').dxy.idxmin()].sort_values(['player', 'frame'])

rows = []
for pl, t in c.groupby('player'):
    grp = (t.frame.diff() > 3).cumsum()
    for _, e in t.groupby(grp):
        r = e.loc[e.d_chest.idxmin()]
        both = (r.r_ankle_s > 0.4) & (r.l_ankle_s > 0.4)
        R = cam.ground(np.array([[r.r_ankle_x, r.r_ankle_y + 6]]))[0]
        L = cam.ground(np.array([[r.l_ankle_x, r.l_ankle_y + 6]]))[0]
        dist = float(np.hypot(*(R - L)[:2])) if both else np.nan
        lift = abs(r.r_ankle_y - r.l_ankle_y) / r.torso  # ankle height gap in torso lengths
        rows.append(dict(frame=int(r.frame), time_sec=round(r.frame / 25, 2), player=pl,
                         feet_distance_m=round(dist, 2), foot_in_air=bool(lift > 0.35),
                         ball_height_m=float(h.height_m[int(r.frame)]),
                         ball_to_player_m=round(float(r.dxy), 2),
                         chest_offset=round(float(r.d_chest), 2), frames=len(e)))
out = pd.DataFrame(rows).sort_values('frame')
out.to_csv('touches/chest_touches.csv', index=False)
print(len(out), 'chest touches'); print(out.groupby('player').feet_distance_m.describe().round(2))
print('foot in air:', int(out.foot_in_air.sum()))
print(out.head(12).to_string())
