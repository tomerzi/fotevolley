"""Apply the user's marks from the doubt-marking page (data/experiments/doubt_marks/marks_*.json, newest round wins
per item) to the analysis data:
- landings: a click on the ball at its landing frame -> court x, y on the sand (camera ray at z = 0.11 m); a click
  with "ברשת" -> the point on the net plane (x = 8). The verdict sets the kind (in_court / out / net). "לא רואים"
  keeps the old values. Source becomes "hand-marked (doubt page)".
Writes data/analysis/point_landings.csv in place (original kept as point_landings_before_doubt.csv). The rally 39 touch
(v2 frame 10506) is now in the touch labels, not patched here.
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
# user's written notes override a click: rally 21 "ignore the out mark"; rally 13 "an Israeli player kicks the ball
# out" (no landing seen)
OVERRIDE = {21: None, 13: 'בחוץ'}
for r_, v_ in OVERRIDE.items():
    k_ = f'land_{r_:02d}'
    if v_ is None:
        marks.pop(k_, None)
    else:
        marks[k_] = dict(rally=r_, verdict=v_)
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
# rally 39 (the untagged touch at v3 frame 8966 is Ron_Ben_Ishay) now lives in the source labels
# (data/contacts/touch_labels_final.csv, touch 246; data/contacts/owner_decisions.csv); this script no longer
# patches data/contacts/touches_master.csv, which scripts/contacts/build_touch_table.py rebuilds.
print(land[land.source == 'hand-marked (doubt page)'][['rally', 'winner', 'kind', 'x', 'y', 'height_m']].to_string())

# ---- partner gap: 4 clicks = both feet of each defender at the opponent's attack -> sand plane (z = 0.03 m),
# each defender = the midpoint of his two feet. Clicks are paired as (1,2)(3,4); if (1,3)(2,4) gives much tighter
# pairs the user clicked across, and that pairing is used.
SP = 'data/analysis/team_spacing_v3.csv'
if not os.path.exists(SP.replace('.csv', '_before_doubt.csv')):
    shutil.copy(SP, SP.replace('.csv', '_before_doubt.csv'))
sp = pd.read_csv(SP.replace('.csv', '_before_doubt.csv'))
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
sp['v2'] = sp.frame.map(fm)
sp['gap_source'] = 'tracking'
rows = []
for k, m in marks.items():
    if not k.startswith('gap_'):
        continue
    _, r_, f_ = k.split('_'); r_, f_ = int(r_), int(f_)
    i = sp.index[(sp.rally == r_) & (sp.v2 == f_) & (sp.moment == 'attack')]
    if not len(i):
        continue
    i = i[0]
    if m.get('verdict') == 'cant' or len(m.get('pts', [])) != 4:
        sp.loc[i, 'gap_source'] = 'user: cannot tell'; continue
    uv = np.array([[p['u'], p['v']] for p in m['pts']], float)
    d = cam.ray(uv); P = cam.C + ((-0.03 - cam.C[2]) / d[:, 2])[:, None] * d; P = P[:, :2]
    a = np.linalg.norm(P[0] - P[1]) + np.linalg.norm(P[2] - P[3]); b = np.linalg.norm(P[0] - P[2]) + np.linalg.norm(P[1] - P[3])
    A, B = ((P[0] + P[1]) / 2, (P[2] + P[3]) / 2) if a <= b * 1.5 else ((P[0] + P[2]) / 2, (P[1] + P[3]) / 2)
    dist = float(np.linalg.norm(A - B))
    rows.append(dict(rally=r_, frame=f_, before=sp.loc[i, 'distance'], marked=round(dist, 3), stance=round(min(a, b) / 2, 2)))
    sp.loc[i, ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'distance']] = [round(A[0], 3), round(A[1], 3), round(B[0], 3), round(B[1], 3), round(dist, 3)]
    sp.loc[i, 'gap_source'] = 'user: 4 feet clicked'
sp.drop(columns='v2').to_csv(SP, index=False)
g = pd.DataFrame(rows)
if len(g):
    g['diff'] = g.marked - g.before
    print('gaps marked: %d   vs tracking: median |diff| %.2f m, max %.2f m; median stance (feet apart) %.2f m' % (len(g), g['diff'].abs().median(), g['diff'].abs().max(), g.stance.median()))
    print('crossed the 3.5 m line:', int(((g.marked > 3.5) != (g.before > 3.5)).sum()))
    g.to_csv('data/experiments/doubt_marks/gaps_marked.csv', index=False)
