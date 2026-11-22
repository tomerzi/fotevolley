"""Facing the camera or back to it, per frame and player (v2 frames, ViTPose BODY_25 as body_rays.py uses), so that
body orientation is only used where it can be measured.

'camera' = the pose's left shoulder is on the image right and the face (nose / eyes) is seen; 'away' = back to the
camera (left shoulder on the image left, or no face seen): the pose model often swaps left and right then, and the eye
ray has no eyes to work from, so orientation_ok = False; 'side' = shoulders closer than 0.25 trunk lengths in the
image (side-on, the left / right order cannot be trusted, but eyes and feet still are).

Output: data/analysis/body_orientation_facing.csv (frame, player, facing, face_score, orientation_ok) and a summary
per player printed. Join it to body_orientation_0_15585.csv on (frame, player).
"""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OLD = {'BRA_A': 'Franklin', 'BRA_B': 'Hiltinho', 'ISR_A': 'Ron_Ben_Ishay', 'ISR_B': 'Maor_Haas'}
K = ['nose', 'l_eye', 'r_eye', 'neck', 'mid_hip', 'l_shoulder', 'r_shoulder']
p = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player'] + [f'{k}_{a}' for k in K for a in 'xys'])
p['player'] = p.player.replace(OLD)
sh = p.l_shoulder_x - p.r_shoulder_x
trunk = np.hypot(p.neck_x - p.mid_hip_x, p.neck_y - p.mid_hip_y).clip(lower=1)
face = p[['nose_s', 'l_eye_s', 'r_eye_s']].max(axis=1)
ok_sh = (p.l_shoulder_s >= 0.3) & (p.r_shoulder_s >= 0.3)
p['facing'] = np.select([~ok_sh | (sh.abs() / trunk < 0.25), (sh > 0) & (face >= 0.5), (sh < 0) | (face < 0.3)],
                        ['side', 'camera', 'away'], 'side')
p['face_score'] = face.round(2)
p['orientation_ok'] = p.facing != 'away'
p[['frame', 'player', 'facing', 'face_score', 'orientation_ok']].to_csv('data/analysis/body_orientation_facing.csv', index=False)
print(pd.crosstab(p.player, p.facing, normalize='index').round(2))
