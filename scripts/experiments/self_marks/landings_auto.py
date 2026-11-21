"""EXPERIMENT: find where each point ended without hand marks.
After the rally's last player touch, take the ball's fitted flights (data/analysis/ball_height_v2.csv). The landing is
where the last flight reaches the sand (ball centre at 0.11 m): the first fitted frame at or below it, otherwise the
flight extrapolated with real gravity (x, y straight in time, height a parabola with g = 9.81).
Validated on the 24 hand-marked landings first (held out: never used here), then applied to the 22 points that only
had the rally-end fallback. Writes data/experiments/self_marks/landings_auto.csv.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
G, RB = 9.81, 0.11
h = pd.read_csv('data/analysis/ball_height_v2.csv')
ral = pd.read_csv('data/points/rallies_v2.csv').set_index('rally')
t = pd.read_csv('data/contacts/touches_master.csv'); t = t[t.kind == 'player']
last_touch = t.groupby('rally').v2_frame.max()
land = pd.read_csv('data/analysis/point_landings.csv')
rows = []
for r in land.itertuples():
    lt = int(last_touch.get(r.rally, ral.loc[r.rally, 'start_frame']))
    end = int(ral.loc[r.rally, 'end_frame'])
    w = h[(h.frame > lt) & (h.frame <= end + 50) & (h.method == 'flight_fit')]
    if not len(w):
        rows.append(dict(rally=r.rally, auto_x=np.nan, auto_y=np.nan, auto_frame=np.nan, how='no flight after last touch')); continue
    fid = w.flight_id.iloc[0]   # the flight leaving the last touch
    g = w[w.flight_id == fid].sort_values('frame')
    low = g[g.height_m <= RB + 0.05]
    if len(low):
        q = low.iloc[0]; rows.append(dict(rally=r.rally, auto_x=q.court_x_m, auto_y=q.court_y_m, auto_frame=q.frame, how='fitted flight reaches sand')); continue
    tt = (g.frame.to_numpy() - g.frame.iloc[0]) / 25.0
    if len(g) < 4:
        rows.append(dict(rally=r.rally, auto_x=np.nan, auto_y=np.nan, auto_frame=np.nan, how='flight too short')); continue
    ax = np.polyfit(tt, g.court_x_m, 1); ay = np.polyfit(tt, g.court_y_m, 1)
    # height = h0 + v t - g t^2 / 2  -> fit h0, v with g fixed
    A = np.c_[np.ones_like(tt), tt]; h0, v = np.linalg.lstsq(A, g.height_m + 0.5 * G * tt ** 2, rcond=None)[0]
    disc = v * v + 2 * G * (h0 - RB)
    if disc < 0:
        rows.append(dict(rally=r.rally, auto_x=np.nan, auto_y=np.nan, auto_frame=np.nan, how='no ground crossing')); continue
    tl = (v + np.sqrt(disc)) / G
    gap = tl * 25 - (g.frame.iloc[-1] - g.frame.iloc[0])
    rows.append(dict(rally=r.rally, auto_x=np.polyval(ax, tl), auto_y=np.polyval(ay, tl), auto_frame=g.frame.iloc[0] + tl * 25,
                     how='extrapolated to the sand', gap_frames=gap))
a = land.merge(pd.DataFrame(rows), on='rally')
a['err_m'] = np.hypot(a.auto_x - a.x, a.auto_y - a.y)
a['conf'] = np.where(a.how == 'fitted flight reaches sand', 'high', np.where(a.gap_frames <= 6, 'medium', 'low'))
inside = (a.auto_x.between(-0.1, 16.1)) & (a.auto_y.between(-0.1, 8.1))
a.loc[(a.kind == 'out') & inside, 'conf'] = 'low'   # says out, lands inside: something else stopped it
a.loc[(a.kind == 'in_court') & ~inside, 'conf'] = 'low'
hand = a[a.source == 'hand-marked landing']
print('held-out check vs hand marks: n', hand.err_m.notna().sum(), 'of', len(hand), ' median err %.2f m  p80 %.2f m' % (hand.err_m.median(), hand.err_m.quantile(.8)))
print(hand.groupby('conf').err_m.agg(['size','median','max']).round(2))
trk = a[a.source == 'ball track']; print('vs ball-track landings: median %.2f m' % trk.err_m.median())
fb = a[a.source == 'rally end']
print(fb[['rally', 'kind', 'x', 'y', 'auto_x', 'auto_y', 'auto_frame', 'gap_frames', 'conf']].round(2).to_string())
os.makedirs('data/experiments/self_marks', exist_ok=True)
a.round(3).to_csv('data/experiments/self_marks/landings_auto.csv', index=False)
