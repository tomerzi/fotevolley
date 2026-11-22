"""EXPERIMENT: re-measure the partner gap at the opponent's attack for the 60 doubtful moments (a defender hidden or
airborne, or the gap within 3.2-3.8 m of the 3.5 m line) without hand marks.
Feet from the pose (ViTPose heels + big toes, conf >= 0.3, on the sand plane z = 0.03 m; ankles at 0.08 m otherwise);
a defender in the air takes the median of his feet over the nearest 6 grounded frames within +-0.5 s.
Validated first on the 66 clean moments (agreement with the tracked positions). Then each doubtful gap gets an error
estimate and a verdict: clearly above / clearly below 3.5 m (more than 2 sigma away) or still undecided.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT); sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera()
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
sp = pd.read_csv('data/analysis/team_spacing_v3.csv').query("moment == 'attack'").reset_index(drop=True)
sp['v2'] = sp.frame.map(fm).astype(int)
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'team', 'x', 'y', 'occluded', 'airborne']).drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
KP = ['r_heel', 'l_heel', 'r_big_toe', 'l_big_toe', 'r_ankle', 'l_ankle']
pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player'] + [f'{k}_{c}' for k in KP for c in 'xys']).drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])


def plane(uv, z):
    d = cam.ray(np.asarray(uv, float).reshape(-1, 2)); return cam.C + ((-z - cam.C[2]) / d[:, 2])[:, None] * d


def feet(f, pl):
    try:
        r = pose.loc[(f, pl)]
    except KeyError:
        return None
    pts = [(r[f'{k}_x'], r[f'{k}_y']) for k in KP[:4] if r[f'{k}_s'] >= 0.3]
    if len(pts) >= 2:
        return plane(pts, 0.03)[:, :2].mean(0)
    pts = [(r[f'{k}_x'], r[f'{k}_y']) for k in KP[4:] if r[f'{k}_s'] >= 0.3]
    return plane(pts, 0.08)[:, :2].mean(0) if pts else None


def grounded(f, pl):
    try:
        q = pos.loc[(f, pl)]
        return not bool(q.airborne)
    except KeyError:
        return False


def feet_robust(f, pl):
    if grounded(f, pl):
        return feet(f, pl), 'pose at the moment'
    near = sorted(range(f - 12, f + 13), key=lambda k: abs(k - f))
    got = [feet(k, pl) for k in near if k != f and grounded(k, pl)]
    got = [g for g in got if g is not None][:6]
    return (np.median(got, axis=0), 'grounded frames around it') if got else (None, 'none')


def flag(r):
    try:
        g = [pos.loc[(r.v2, p)] for p in TEAM if TEAM[p] == r.defending_team]
        return any(bool(q.occluded) or bool(q.airborne) for q in g)
    except KeyError:
        return True


rows = []
for r in sp.itertuples():
    pls = [p for p in TEAM if TEAM[p] == r.defending_team]
    F = [feet_robust(r.v2, p) for p in pls]
    hidden = flag(r)
    doubtful = hidden or 3.2 <= r.distance <= 3.8
    if any(f[0] is None for f in F):
        rows.append(dict(rally=r.rally, team=r.defending_team, frame=r.v2, d_track=r.distance, d_pose=np.nan, doubtful=doubtful, hidden=hidden, how='no pose')); continue
    (a, ha), (b, hb) = F
    d = float(np.hypot(*(a - b)))
    rows.append(dict(rally=r.rally, team=r.defending_team, frame=r.v2, conceded=r.conceded, d_track=r.distance, d_pose=d, doubtful=doubtful, hidden=hidden, how=ha if ha == hb else ha + ' / ' + hb,
                     far_y=min(a[1], b[1])))
d = pd.DataFrame(rows)
d['diff'] = d.d_pose - d.d_track
clean = d[~d.doubtful & d.d_pose.notna()]
print('clean moments: %d, pose vs tracking: median |diff| %.2f m, 90%% within %.2f m, bias %+.2f m' % (len(clean), clean['diff'].abs().median(), clean['diff'].abs().quantile(.9), clean['diff'].median()))
sig = clean['diff'].abs().quantile(.68)   # 1-sigma of a single gap measurement, from the two independent methods
dd = d[d.doubtful].copy()
dd['best'] = np.where(dd.hidden, dd.d_pose, (dd.d_pose + dd.d_track) / 2)
s_best = np.where(dd.hidden, sig, sig / np.sqrt(2))
dd['sigma'] = s_best
dd['verdict'] = np.where(dd.best - 2 * dd.sigma > 3.5, 'clearly above 3.5', np.where(dd.best + 2 * dd.sigma < 3.5, 'clearly below 3.5', 'undecided'))
dd['flipped'] = (dd.best > 3.5) != (dd.d_track > 3.5)
print('doubtful moments: %d (no pose: %d)' % (len(dd), dd.d_pose.isna().sum()))
print(dd.verdict.value_counts().to_string()); print('side of 3.5 m changed vs before:', int(dd.flipped.sum()))
os.makedirs('data/experiments/self_marks', exist_ok=True)
d.round(3).to_csv('data/experiments/self_marks/gap_all.csv', index=False)
dd.round(3).to_csv('data/experiments/self_marks/gap_doubtful.csv', index=False)
# effect on the headline: ISR concede rate above / below 3.5 m with the new distances
new = d.set_index(['rally', 'frame']).d_track.copy()
for x in dd.itertuples():
    if np.isfinite(x.best):
        new.loc[(x.rally, x.frame)] = x.best
d['d_new'] = new.values
for col in ('d_track', 'd_new'):
    i = d[d.team == 'ISR']; far = i[i[col] > 3.5]; near = i[i[col] <= 3.5]
    print(col, 'ISR concede >3.5: %d/%d = %.0f%%   <=3.5: %d/%d = %.0f%%' % (far.conceded.sum(), len(far), 100 * far.conceded.mean(), near.conceded.sum(), len(near), 100 * near.conceded.mean()))
# threshold-free: logistic-ish view = concede rate by gap tercile (ISR+BRA)
d['tercile'] = pd.qcut(d.d_new, 3, labels=['close', 'mid', 'wide'])
print(d.groupby('tercile', observed=True).agg(n=('conceded', 'size'), concede=('conceded', 'mean'), gap=('d_new', 'median')).round(2))
