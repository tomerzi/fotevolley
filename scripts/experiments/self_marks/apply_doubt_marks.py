"""Apply the user's marks from the doubt-marking page (data/experiments/doubt_marks/marks_*.json, newest round wins
per item) to the analysis data:
- landings: a click on the ball at its landing frame -> court x, y on the sand (camera ray at z = 0.11 m); a click
  with "ברשת" -> the point on the net plane (x = 8). The verdict sets the kind (in_court / out / net). "לא רואים"
  keeps the old values. Source becomes "hand-marked (doubt page)".
- rally 39: the untagged touch at v2 frame 10506 is Maor Haas (user), which splits Brazil's possession.
Writes data/analysis/point_landings.csv and data/contacts/touches_master.csv in place (originals kept as *_before_doubt.csv).
"""
import os, sys, json, glob, shutil
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT); sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera()
BH = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
marks = {}
for f in sorted(glob.glob('data/experiments/doubt_marks/marks_round*.json')):
    marks.update(json.load(open(f)))
LP = 'data/analysis/point_landings.csv'
if not os.path.exists(LP.replace('.csv', '_before_doubt.csv')):
    shutil.copy(LP, LP.replace('.csv', '_before_doubt.csv'))
land = pd.read_csv(LP.replace('.csv', '_before_doubt.csv'))
KIND = {'בפנים': 'in_court', 'בחוץ': 'out', 'ברשת': 'net'}
n = 0
for k, m in marks.items():
    if not k.startswith('land_') or m.get('verdict') not in KIND:
        continue
    i = land.index[land.rally == m['rally']][0]
    land.loc[i, 'kind'] = KIND[m['verdict']]
    land.loc[i, 'source'] = 'hand-marked (doubt page)'
    if 'u' in m:
        d = cam.ray(np.array([[m['u'], m['v']]]))[0]
        if m['verdict'] == 'ברשת':
            # the camera stands on the net line, so the net plane cannot place the ball; use the ball's measured
            # height in that frame (flight fit, else 1 m) and keep x on the net
            hb = BH.height_m.get(int(m['frame']), np.nan)
            z = hb if np.isfinite(hb) else 1.0
            s = (-z - cam.C[2]) / d[2]; P = cam.C + s * d
            land.loc[i, ['x', 'y', 'height_m']] = [8.0, round(float(np.clip(P[1], -0.5, 8.5)), 2), round(z, 2)]
        else:
            s = (-0.11 - cam.C[2]) / d[2]; P = cam.C + s * d
            land.loc[i, ['x', 'y', 'height_m']] = [round(P[0], 2), round(P[1], 2), 0.0]
        land.loc[i, 'frame'] = int(m['frame'])
    elif land.loc[i, 'source'] != 'hand-marked landing':
        land.loc[i, ['x', 'y', 'height_m']] = [np.nan, np.nan, np.nan]   # kind known, place not: the old point was the ball at rally end
    n += 1
land.to_csv(LP, index=False)
print('landings updated:', n)
# rally 39
TP = 'data/contacts/touches_master.csv'
if not os.path.exists(TP.replace('.csv', '_before_doubt.csv')):
    shutil.copy(TP, TP.replace('.csv', '_before_doubt.csv'))
t = pd.read_csv(TP.replace('.csv', '_before_doubt.csv'))
if marks.get('touch_39'):
    i = t.index[(t.rally == 39) & (t.v2_frame == 10506)][0]
    t.loc[i, ['kind', 'player', 'team', 'in_play', 'action']] = ['player', 'Maor_Haas', 'ISR', True, 'attack']
    t.loc[i, 'action_subtype'] = 'returned'
    # possessions of rally 39 rebuilt from the team sequence
    r = t[(t.rally == 39) & (t.kind == 'player')].sort_values('v2_frame')
    pos, k, last = 0, 0, None
    for j in r.index:
        if t.loc[j, 'team'] != last:
            pos += 1; k = 0; last = t.loc[j, 'team']
        k += 1; t.loc[j, 'possession'] = pos; t.loc[j, 'touch_in_possession'] = k
    for p_, g in t[(t.rally == 39) & (t.kind == 'player')].groupby('possession'):
        t.loc[g.index, 'possession_len'] = len(g)
    j = r.index[r.v2_frame == 10522][0]; t.loc[j, 'action'] = 'reception'; t.loc[j, 'action_subtype'] = 'after_attack'
    t.to_csv(TP, index=False)
    print(t[t.rally == 39][['v2_frame', 'player', 'team', 'action', 'possession', 'touch_in_possession', 'possession_len']].to_string())
print(land[land.source == 'hand-marked (doubt page)'][['rally', 'winner', 'kind', 'x', 'y', 'height_m']].to_string())
