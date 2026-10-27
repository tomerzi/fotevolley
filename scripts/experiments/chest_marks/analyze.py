"""Use the hand-marked chest touches to test each camera calibration.

Input: the marking page's database exported with ArtifactData list (out_dir) to
data/experiments/chest_marks/marks/marks/<touch>.json (one file per touch: frame, ball_x, ball_y, feet, chest).
For each calibration the marked ball pixel is put at the touching player's horizontal position (ground point from
data/analysis/player_positions_v3.csv) and its height is compared with 0.72 x that player's stature measured with
the same calibration (standing necks, see physics_calib/check.py). A right calibration gives a ratio ~0.72 on
touches with both feet on the sand. Writes data/experiments/chest_marks/result.csv.
"""
import os, sys, glob, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

CAMS = {'site (16x8 + Israeli 1.86 m)': ('data/analysis/camera_v2.npz', 'data/analysis/player_statures.json'),
        'physics + antenna (16x8)': ('data/experiments/physics_calib/camera.npz', 'data/experiments/physics_calib/player_statures.json'),
        '18x9 + net + Israeli 1.86 m': ('data/experiments/court_scale/s1.1250/camera.npz', 'data/experiments/court_scale/s1.1250/player_statures.json')}
marks = [json.load(open(f)) for f in glob.glob('data/experiments/chest_marks/marks/**/*.json', recursive=True)]
marks = [m.get('data', m) for m in marks]
marks = [m for m in marks if m.get('chest') and 'ball_x' in m]
print(len(marks), 'marked chest touches')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
v2_to_v3 = fm[~fm.deleted].set_index('v2_frame').v3_frame
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'player', 'x', 'y']).set_index(['frame', 'player'])
Hi = np.linalg.inv(np.load('data/mappings/homography_px_to_m.npy'))
rows = []
for name, (cp, sp) in CAMS.items():
    if not os.path.exists(cp):
        continue
    cam = Camera(cp)
    st = {k: v['stature_m'] for k, v in json.load(open(sp)).items()}
    for m in marks:
        try:
            x, y = pos.loc[(int(v2_to_v3[m['frame']]), m['player'])].values.ravel()[:2]
        except KeyError:
            continue
        q = Hi @ np.array([x, y, 1.0])
        g = cam.ground(np.array([q[:2] / q[2]]))[0]
        d = cam.ray(np.array([[m['ball_x'], m['ball_y']]]))[0]
        s = np.dot(g[:2] - cam.C[:2], d[:2]) / np.dot(d[:2], d[:2])
        hgt = -(cam.C + s * d)[2]
        rows.append(dict(calibration=name, touch=m['touch'], player=m['player'], feet=m.get('feet'),
                         ball_h=round(float(hgt), 3), stature=st.get(m['player']), ratio=round(float(hgt / st[m['player']]), 3)))
r = pd.DataFrame(rows)
r.to_csv('data/experiments/chest_marks/result.csv', index=False)
both = r[r.feet == 'both']
print(both.groupby('calibration').agg(n=('ratio', 'size'), ball_h=('ball_h', 'median'), ratio=('ratio', 'median')).round(3))
print('expected ratio ~0.72 (chest height / stature)')
