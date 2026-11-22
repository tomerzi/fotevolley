"""Partner gap at the opponent's attack from the pose joints (user: "the joints marked in the project are the most
certain"). For every attack moment, each defender's ground point = the midpoint of his two feet, each foot = its heel
and big toe (ViTPose, conf >= 0.5) on the sand plane (z = 0.03 m), else the ankle (z = 0.08 m). A defender in the
air (or with no confident foot) takes the median over the nearest 6 grounded frames within +-0.5 s.
Final gap per moment: pose when both defenders have it; else the user's 4-click marks; else tracking.
Compares pose with the user's clicks and with tracking, and rewrites data/analysis/team_spacing_v3.csv.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT); sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera()
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Ron_Ben_Ishay': 'ISR', 'Maor_Haas': 'ISR'}
C = 0.5
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
base = pd.read_csv('data/analysis/team_spacing_v3_before_doubt.csv')
cur = pd.read_csv('data/analysis/team_spacing_v3.csv')       # tracking + user clicks
base['v2'] = base.frame.map(fm); cur['v2'] = cur.frame.map(fm)
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'airborne']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
KP = ['r_heel', 'l_heel', 'r_big_toe', 'l_big_toe', 'r_ankle', 'l_ankle']
pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player'] + [f'{k}_{c}' for k in KP for c in 'xys']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])


def plane(uv, z):
    d = cam.ray(np.asarray(uv, float).reshape(-1, 2)); return (cam.C + ((-z - cam.C[2]) / d[:, 2])[:, None] * d)[:, :2]


def foot(r, s):
    pts = [(r[f'{s}_{k}_x'], r[f'{s}_{k}_y']) for k in ('heel', 'big_toe') if r[f'{s}_{k}_s'] >= C]
    if pts:
        return plane(pts, 0.03).mean(0)
    if r[f'{s}_ankle_s'] >= C:
        return plane([(r[f'{s}_ankle_x'], r[f'{s}_ankle_y'])], 0.08)[0]
    return None


def ground_point(f, pl):
    try:
        r = pose.loc[(f, pl)]
    except KeyError:
        return None
    a, b = foot(r, 'r'), foot(r, 'l')
    if a is None or b is None:
        return None
    return (a + b) / 2


def grounded(f, pl):
    try:
        return not bool(pos.loc[(f, pl)].airborne)
    except KeyError:
        return False


def robust(f, pl):
    if grounded(f, pl):
        g = ground_point(f, pl)
        if g is not None:
            return g, 'at the moment'
    near = sorted([k for k in range(f - 12, f + 13) if k != f], key=lambda k: abs(k - f))
    got = [g for g in (ground_point(k, pl) for k in near if grounded(k, pl)) if g is not None][:6]
    return (np.median(got, axis=0), 'grounded frames around it') if got else (None, 'no confident feet')


rows = []
for i, r in cur.iterrows():
    if r.moment != 'attack':
        continue
    pls = [p for p in TEAM if TEAM[p] == r.defending_team]
    (A, ha), (B, hb) = robust(int(r.v2), pls[0]), robust(int(r.v2), pls[1])
    d_pose = float(np.linalg.norm(A - B)) if A is not None and B is not None else np.nan
    d_track = base.loc[i, 'distance']
    d_user = r.distance if r.gap_source == 'user: 4 feet clicked' else np.nan
    rows.append(dict(idx=i, rally=r.rally, frame=int(r.v2), team=r.defending_team, conceded=r.conceded, d_pose=d_pose, d_user=d_user, d_track=d_track,
                     how=ha if ha == hb else f'{ha} / {hb}', A=A, B=B))
d = pd.DataFrame(rows)
u = d.dropna(subset=['d_pose', 'd_user'])
print('pose available: %d/%d moments' % (d.d_pose.notna().sum(), len(d)))
print('pose vs your clicks (%d): median |diff| %.2f m, 90%% within %.2f m, bias %+.2f m' % (len(u), (u.d_pose - u.d_user).abs().median(), (u.d_pose - u.d_user).abs().quantile(.9), (u.d_pose - u.d_user).median()))
t = d.dropna(subset=['d_pose'])
print('pose vs tracking (%d): median |diff| %.2f m, bias %+.2f m' % (len(t), (t.d_pose - t.d_track).abs().median(), (t.d_pose - t.d_track).median()))
big = u[(u.d_pose - u.d_user).abs() > 0.5]
print('pose and your clicks differ by more than 0.5 m:'); print(big[['rally', 'frame', 'team', 'd_pose', 'd_user', 'd_track', 'how']].round(2).to_string())
# final values
sp = cur.copy()
for x in d.itertuples():
    if np.isfinite(x.d_pose):
        sp.loc[x.idx, ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'distance']] = [round(x.A[0], 3), round(x.A[1], 3), round(x.B[0], 3), round(x.B[1], 3), round(x.d_pose, 3)]
        sp.loc[x.idx, 'gap_source'] = 'pose joints'
sp.drop(columns='v2').to_csv('data/analysis/team_spacing_v3.csv', index=False)
d.drop(columns=['A', 'B']).round(3).to_csv('data/experiments/doubt_marks/gaps_pose_vs_user.csv', index=False)
print(sp[sp.moment == 'attack'].gap_source.value_counts().to_string())
a = sp[sp.moment == 'attack']
for tm in ('ISR', 'BRA'):
    i_ = a[a.defending_team == tm]; f_ = i_[i_.distance > 3.5]; n_ = i_[i_.distance <= 3.5]
    print(tm, 'concede >3.5: %d/%d = %.0f%%   <=3.5: %d/%d = %.0f%%' % (f_.conceded.sum(), len(f_), 100 * f_.conceded.mean(), n_.conceded.sum(), len(n_), 100 * n_.conceded.mean()))
a = a.copy(); a['t'] = pd.qcut(a.distance, 3, labels=['close', 'mid', 'wide'])
print(a.groupby('t', observed=True).agg(n=('conceded', 'size'), concede=('conceded', 'mean'), gap=('distance', 'median')).round(2).to_string())
