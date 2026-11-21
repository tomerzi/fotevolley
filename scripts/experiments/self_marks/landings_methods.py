"""EXPERIMENT: three ways to find a landing without hand marks, each scored on the 24 hand-marked landings.
M1 flight extrapolation (landings_auto.py); M2 first frame after the last touch with ball height <= 0.3 m (any
height method); M3 bounce in the image: the first sharp change of the ball's image velocity after the last touch,
placed on the sand by the camera ray at z = 0.11 m.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT); sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera()
h = pd.read_csv('data/analysis/ball_height_v2.csv').set_index('frame')
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv').set_index('frame')
ral = pd.read_csv('data/points/rallies_v2.csv').set_index('rally')
t = pd.read_csv('data/contacts/touches_master.csv'); t = t[t.kind == 'player']; lt = t.groupby('rally').v2_frame.max()
land = pd.read_csv('data/analysis/point_landings.csv')
m1 = pd.read_csv('data/experiments/self_marks/landings_auto.csv').set_index('rally')


def ground(u, v, z=0.11):
    d = cam.ray(np.array([[u, v]]))[0]; s = (-z - cam.C[2]) / d[2]; P = cam.C + s * d
    return P[0], P[1]


rows = []
for r in land.itertuples():
    a = int(lt.get(r.rally, ral.start_frame[r.rally])) + 4; e = int(ral.end_frame[r.rally]) + 30
    w = h.loc[a:e]
    low = w[(w.height_m <= 0.3) & w.court_x_m.notna()]
    m2 = (low.court_x_m.iloc[0], low.court_y_m.iloc[0], low.index[0]) if len(low) else (np.nan, np.nan, np.nan)
    bb = b.loc[a:e, ['center_x', 'center_y']].dropna()
    m3 = (np.nan, np.nan, np.nan)
    if len(bb) > 6:
        f = bb.index.to_numpy(); u = bb.center_x.to_numpy(); v = bb.center_y.to_numpy()
        ok = np.diff(f) == 1
        vel = np.c_[np.diff(u), np.diff(v)]
        for i in range(1, len(vel)):
            if ok[i] and ok[i - 1] and vel[i - 1, 1] > 2 and np.hypot(*(vel[i] - vel[i - 1])) > max(6, 0.6 * np.hypot(*vel[i - 1])):
                X, Y = ground(u[i], v[i]); m3 = (X, Y, f[i]); break
    rows.append(dict(rally=r.rally, source=r.source, kind=r.kind, x=r.x, y=r.y, m1x=m1.auto_x.get(r.rally), m1y=m1.auto_y.get(r.rally), m1c=m1.conf.get(r.rally),
                     m2x=m2[0], m2y=m2[1], m2f=m2[2], m3x=m3[0], m3y=m3[1], m3f=m3[2]))
d = pd.DataFrame(rows)
for k in ('m1', 'm2', 'm3'):
    d[k + 'e'] = np.hypot(d[k + 'x'] - d.x, d[k + 'y'] - d.y)
hand = d[d.source == 'hand-marked landing']
for k in ('m1', 'm2', 'm3'):
    e = hand[k + 'e']
    print(k, 'found %d/24  median %.2f m  within 0.5 m: %d  within 1 m: %d' % (e.notna().sum(), e.median(), (e < .5).sum(), (e < 1).sum()))
# agreement of M2 and M3 as a self-check
d['m23'] = np.hypot(d.m2x - d.m3x, d.m2y - d.m3y)
hh = d[d.source == 'hand-marked landing']
for thr in (0.5, 1.0):
    sel = hh[hh.m23 < thr]
    print(f'M2 and M3 agree within {thr} m: {len(sel)}/24, their mean vs hand: median %.2f max %.2f' % (np.hypot((sel.m2x + sel.m3x) / 2 - sel.x, (sel.m2y + sel.m3y) / 2 - sel.y).median(), np.hypot((sel.m2x + sel.m3x) / 2 - sel.x, (sel.m2y + sel.m3y) / 2 - sel.y).max()))
d.round(3).to_csv('data/experiments/self_marks/landings_methods.csv', index=False)
print(d[d.source == 'rally end'][['rally', 'kind', 'm1x', 'm1y', 'm1c', 'm2x', 'm2y', 'm2f', 'm3x', 'm3y', 'm3f', 'm23']].round(2).to_string())
