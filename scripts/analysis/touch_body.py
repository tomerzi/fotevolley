"""Body at every touch from the joint positions (Sapiens2 keypoints, v3 video): joint angles, which way the player
faces the camera, and whether each foot stands on the sand.

Angles (degrees, median over frames t-1..t+1 of the touch; t = labelled touch frame):
  knee_l / knee_r     hip - knee - ankle            180 = straight leg, smaller = bent
  hip_l / hip_r       shoulder - hip - knee         180 = body straight, smaller = bent at the hip
  elbow_l / elbow_r   shoulder - elbow - wrist      180 = straight arm
  arm_l / arm_r       upper arm against the trunk   0 = arm along the body, 90 = arm out to the side / front
  trunk_lean          trunk (hip midpoint -> neck) against the true vertical at the player's spot (the camera's image
                      of a vertical line there, camera_v2.npz), 0 = upright; trunk_lean_dir = 'net' when the
                      shoulders are on the net side of the hips in the image, else 'back'
  neck                trunk against neck -> nose: head bent forward / back
All angles are measured in the image plane (one camera), so a limb pointing towards or away from the camera looks
shorter and its angle is not reliable. Each angle has a *_ok flag: the joints' scores >= S_OK and every segment it
uses at least FORESHORT x that player's usual (median) image length of the segment, scaled by the box height.

facing (from the pose's left / right labels and the face): 'camera' = the player's left shoulder is on the image
right and the face is seen; 'away' = back to the camera (left shoulder on the image left, nose / eyes weak, ears or
neck seen); 'side' = shoulders too close together in the image to tell. With the back to the camera the pose model
often mixes up left and right, so left / right angles and body orientation are not reliable there (orientation_ok).

feet: each foot's ground point (heel, big toe, small toe; ankle if none) over t-PLANT_DT..t+PLANT_DT, as in
chest_stance.py: 'planted' (moves less than PLANT_MOVE box heights), 'raised' (moves and is more than LIFT_FRAC leg
lengths above the other foot in the image), 'moving' (a step or a slide), 'unseen' (score too low: foot in the sand,
hidden). stance = the player's state: 'set' (both feet planted), 'one_foot' (one planted), 'moving' (none planted),
'air' (jump flag of player_positions_v3 or both feet raised); approach_speed from touch_speed_v3.csv.

Output: data/analysis/touch_body.csv (one row per player touch), data/analysis/touch_body_summary.csv (player x body
part: medians of the reliable angles, share of each stance and facing).
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
S_OK = 0.5
FORESHORT = 0.6
DT = 1
PLANT_DT = 2
PLANT_MOVE = 0.06
SET_SPEED = 1.3     # m/s at the touch: below = stood set (calibrated, see stance_cal)
SIDE_W = 0.4        # shoulder width / trunk length below this = side-on (calibrated on 60 owner labels: 72% in sample, 62% leave-one-out; was 0.25)
LIFT_FRAC = 0.5
OLD = {'BRA_A': 'Franklin', 'BRA_B': 'Hiltinho', 'ISR_A': 'Ron_Ben_Ishay', 'ISR_B': 'Maor_Haas'}
J = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow',
     'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
     'right_ankle', 'left_heel', 'right_heel', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe']
SEG = {'thigh': ('hip', 'knee'), 'shin': ('knee', 'ankle'), 'upper_arm': ('shoulder', 'elbow'), 'forearm': ('elbow', 'wrist')}

# ---------- camera (numpy port of scripts/ball_height/traj.py Camera.project) ----------
cam = np.load('data/analysis/camera_v2.npz')
K, dist, R, tv = cam['K'], cam['dist'].ravel(), cam['R'], cam['t'].ravel()
def project(X):
    Xc = (R @ np.asarray(X, float).T).T + tv
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2 = dist[:4]; k3 = dist[4] if len(dist) > 4 else 0
    r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]]

def ang(a, b, c):
    """angle at b (deg) of the image points a-b-c, arrays (n,2)."""
    u, v = a - b, c - b
    cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))

def vang(u, v):
    cs = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))

# ---------- touches and pose ----------
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[(t.kind == 'player') & t.player.notna()].copy()
need = set(f for f in t.frame for f in range(f - PLANT_DT, f + PLANT_DT + 1))
cols = ['frame', 'player', 'y1', 'y2'] + [f'{j}_{a}' for j in J for a in 'xys']
pose = pd.concat(c[c.frame.isin(need)] for c in pd.read_csv('data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv',
                                                           usecols=cols, chunksize=200000))
pose['player'] = pose.player.replace(OLD)
pose = pose.drop_duplicates(['frame', 'player']).set_index(['frame', 'player']).sort_index()
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'player', 'x', 'y', 'jump', 'airborne', 'posture'])
pos = pos.drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
spd = pd.read_csv('data/analysis/touch_speed_v3.csv').set_index('touch')
H = np.load('data/mappings/homography_px_to_m.npy')

# usual segment lengths per player (in box heights), from all loaded frames with good scores
P = lambda d, j: d[[f'{j}_x', f'{j}_y']].to_numpy(float)
S = lambda d, j: d[f'{j}_s'].to_numpy(float)
bh_all = (pose.y2 - pose.y1).to_numpy(float)
usual = {}
for seg, (a, b) in SEG.items():
    for s in ('left', 'right'):
        L = np.linalg.norm(P(pose, f'{s}_{a}') - P(pose, f'{s}_{b}'), axis=1) / bh_all
        ok = (S(pose, f'{s}_{a}') >= S_OK) & (S(pose, f'{s}_{b}') >= S_OK)
        sr = pd.Series(np.where(ok, L, np.nan), index=pose.index.get_level_values('player'))
        for p, v in sr.groupby(level=0).quantile(0.75).items():   # upper quartile ~ segment seen side-on
            usual[(p, seg, s)] = v

def body(d, p):
    """angles and flags for the pose rows d (one player)."""
    bh = (d.y2 - d.y1).to_numpy(float)
    out = {}
    def seg_ok(s, seg):
        a, b = SEG[seg]
        L = np.linalg.norm(P(d, f'{s}_{a}') - P(d, f'{s}_{b}'), axis=1) / bh
        return (S(d, f'{s}_{a}') >= S_OK) & (S(d, f'{s}_{b}') >= S_OK) & (L >= FORESHORT * usual.get((p, seg, s), np.inf))
    for s, k in (('left', 'l'), ('right', 'r')):
        out[f'knee_{k}'] = ang(P(d, f'{s}_hip'), P(d, f'{s}_knee'), P(d, f'{s}_ankle'))
        out[f'knee_{k}_ok'] = seg_ok(s, 'thigh') & seg_ok(s, 'shin')
        out[f'hip_{k}'] = ang(P(d, f'{s}_shoulder'), P(d, f'{s}_hip'), P(d, f'{s}_knee'))
        out[f'hip_{k}_ok'] = seg_ok(s, 'thigh') & (S(d, f'{s}_shoulder') >= S_OK)
        out[f'elbow_{k}'] = ang(P(d, f'{s}_shoulder'), P(d, f'{s}_elbow'), P(d, f'{s}_wrist'))
        out[f'elbow_{k}_ok'] = seg_ok(s, 'upper_arm') & seg_ok(s, 'forearm')
        trunk = (P(d, 'left_hip') + P(d, 'right_hip')) / 2 - P(d, 'neck')
        out[f'arm_{k}'] = vang(P(d, f'{s}_elbow') - P(d, f'{s}_shoulder'), trunk)
        out[f'arm_{k}_ok'] = seg_ok(s, 'upper_arm') & (S(d, 'neck') >= S_OK)
    hipm = (P(d, 'left_hip') + P(d, 'right_hip')) / 2
    tr = P(d, 'neck') - hipm
    # true vertical at the player's ground point, as the camera sees it
    fr = d.index.get_level_values('frame')
    g = np.array([pos.loc[(f, p), ['x', 'y']].to_numpy(float) if (f, p) in pos.index else [np.nan, np.nan] for f in fr])
    up = project(np.c_[g, np.full(len(g), -1.5)]) - project(np.c_[g, np.zeros(len(g))])   # Z down: -1.5 = 1.5 m up
    out['trunk_lean'] = vang(tr, up)
    toward_net = np.sign(8 - g[:, 0])                       # image x grows with court x
    out['trunk_lean_dir'] = np.where(np.sign(tr[:, 0]) == toward_net, 'net', 'back')
    hs = np.minimum(S(d, 'left_hip'), S(d, 'right_hip'))
    out['trunk_lean_ok'] = (hs >= S_OK) & (S(d, 'neck') >= S_OK) & np.isfinite(g[:, 0])
    out['neck'] = vang(P(d, 'nose') - P(d, 'neck'), tr)
    out['neck_ok'] = (S(d, 'nose') >= S_OK) & (S(d, 'neck') >= S_OK)
    # facing the camera?
    sh = P(d, 'left_shoulder')[:, 0] - P(d, 'right_shoulder')[:, 0]
    width = np.abs(sh) / np.maximum(np.linalg.norm(tr, axis=1), 1)
    face = np.maximum(S(d, 'nose'), np.maximum(S(d, 'left_eye'), S(d, 'right_eye')))
    out['facing'] = np.select([width < SIDE_W, (sh > 0) & (face >= S_OK), (sh < 0) | (face < 0.3)],
                              ['side', 'camera', 'away'], 'side')
    out['face_score'] = face
    return out

def foot(d, s):
    pts = np.full((len(d), 2), np.nan); n = np.zeros(len(d))
    acc = np.zeros((len(d), 2))
    for k in ('heel', 'big_toe', 'small_toe'):
        ok = S(d, f'{s}_{k}') >= 0.3
        acc += np.where(ok[:, None], P(d, f'{s}_{k}'), 0); n += ok
    pts = np.where((n > 0)[:, None], acc / np.maximum(n, 1)[:, None], np.nan)
    ank = (S(d, f'{s}_ankle') >= 0.3)[:, None] & (n == 0)[:, None]
    return np.where(ank, P(d, f'{s}_ankle'), pts)

rows = []
for r in t.itertuples():
    p = r.player
    idx = [(f, p) for f in range(r.frame - DT, r.frame + DT + 1) if (f, p) in pose.index]
    row = dict(touch=r.touch, frame=r.frame, v2_frame=r.v2_frame, rally=r.rally, set=r.set, in_play=r.in_play, player=p,
               team=r.team, body_part=r.body_part, limb_side=r.limb_side, action=r.action)
    if idx:
        b = body(pose.loc[idx], p)
        for k, v in b.items():
            if k.endswith('_ok'):
                row[k] = bool(np.mean(v) >= 0.5)
            elif k in ('facing', 'trunk_lean_dir'):
                row[k] = pd.Series(v).mode().iloc[0]
            else:
                row[k] = round(float(np.nanmedian(v)), 1) if np.isfinite(v).any() else np.nan
        row['orientation_ok'] = row['facing'] != 'away'
    # feet
    fidx = [(f, p) for f in range(r.frame - PLANT_DT, r.frame + PLANT_DT + 1) if (f, p) in pose.index]
    st = {}
    if len(fidx) >= 3:
        d = pose.loc[fidx]; bh = float(np.median(d.y2 - d.y1))
        fp = {s: foot(d, s) for s in ('left', 'right')}
        for s in ('left', 'right'):
            a = fp[s]
            if np.isfinite(a[:, 0]).sum() < 3:
                st[s] = 'unseen'; continue
            mv = np.nanmax(np.linalg.norm(a - np.nanmedian(a, 0), axis=1)) / bh
            o = fp['right' if s == 'left' else 'left']
            hip = np.nanmedian(P(d, f'{s}_hip')[:, 1]); leg = max(np.nanmedian(a[:, 1]) - hip, 1)
            higher = np.nanmedian(o[:, 1]) - np.nanmedian(a[:, 1]) > LIFT_FRAC * leg if np.isfinite(o[:, 1]).any() else False
            st[s] = 'planted' if mv < PLANT_MOVE else ('raised' if higher else 'moving')
    row['left_foot'], row['right_foot'] = st.get('left', 'unseen'), st.get('right', 'unseen')
    jump = bool(pos.loc[(r.frame, p), 'jump']) if (r.frame, p) in pos.index else False
    row['jump'] = jump
    feet = [row['left_foot'], row['right_foot']]
    npl = feet.count('planted')
    row['stance'] = ('air' if jump or feet.count('raised') == 2 else
                     'unknown' if feet.count('unseen') == 2 else
                     'set' if npl == 2 else 'one_foot' if npl == 1 else 'moving')
    row['approach_speed'] = spd.approach_speed.get(r.touch, np.nan)
    row['speed_at_touch'] = spd.speed_at_touch.get(r.touch, np.nan)
    # jump = the jump / both-feet-raised test, else 'set' when the body moves slower than SET_SPEED at the touch.
    # Checked on the owner's 69 labelled touches (feet_isr_labels + doubts): 52% over set / moving / jump; the jump
    # test finds about half the jumps (small jumps barely move the ground point). NOT reliable: use stance_label.
    row['stance_cal'] = ('jump' if row['stance'] == 'air' else
                         'set' if row['speed_at_touch'] < SET_SPEED else 'moving')
    rows.append(row)
out = pd.DataFrame(rows)
# the owner's own labels where given (ground truth): stance and facing
lab = pd.concat([pd.read_csv('data/experiments/feet_isr_labels/answers_final.csv'), pd.read_csv('data/experiments/doubts/answers.csv')])
lab = lab[lab.id.str[0].isin(['f', 's', 'c'])].copy()
lab['touch'] = lab.id.str[1:].astype(int)
st_lab = lab[lab.stance.isin(['set', 'moving', 'jump'])].drop_duplicates('touch', keep='last').set_index('touch').stance
fc_lab = lab[lab.facing.isin(['camera', 'away', 'side'])].drop_duplicates('touch', keep='last').set_index('touch').facing
out['stance_label'] = out.touch.map(st_lab)
out['facing_label'] = out.touch.map(fc_lab)
out.to_csv('data/analysis/touch_body.csv', index=False)

ANG = ['knee_l', 'knee_r', 'hip_l', 'hip_r', 'elbow_l', 'elbow_r', 'arm_l', 'arm_r', 'trunk_lean', 'neck']
summ = []
for (p, bp), g in out[out.in_play].groupby(['player', 'body_part']):
    s = dict(player=p, body_part=bp, n=len(g))
    for a in ANG:
        okc = g[f'{a}_ok'] if f'{a}_ok' in g else pd.Series(True, index=g.index)
        okc = okc.fillna(False).astype(bool)
        if a not in ('trunk_lean', 'neck'):
            okc &= g.facing.ne('away')
        s[a] = round(g.loc[okc, a].median(), 1) if okc.any() else np.nan
        s[f'{a}_n'] = int(okc.sum())
    s['lean_back_share'] = round((g.trunk_lean_dir == 'back').mean(), 2)
    for k in ('set', 'moving', 'jump'):
        s[f'stable_{k}'] = round((g.stance_cal == k).mean(), 2)
    for k in ('set', 'one_foot', 'moving', 'air', 'unknown'):
        s[f'stance_{k}'] = round((g.stance == k).mean(), 2)
    for k in ('camera', 'side', 'away'):
        s[f'facing_{k}'] = round((g.facing == k).mean(), 2)
    s['approach_speed'] = round(g.approach_speed.median(), 2)
    summ.append(s)
summ = pd.DataFrame(summ)
summ.to_csv('data/analysis/touch_body_summary.csv', index=False)
print(out.stance.value_counts().to_dict(), out.facing.value_counts().to_dict())
print({a: int(out[f'{a}_ok'].sum()) for a in ANG}, 'of', len(out))
print(summ[summ.body_part == 'chest'][['player', 'n', 'knee_l', 'knee_r', 'knee_l_n', 'knee_r_n', 'hip_l', 'hip_r',
                                       'trunk_lean', 'trunk_lean_n', 'lean_back_share', 'arm_l', 'arm_r', 'stance_set', 'stance_one_foot', 'stance_moving', 'stance_air', 'facing_away']].to_string(index=False))
