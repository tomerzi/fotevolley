"""Per-shot sun direction from ball <-> shadow epipolar geometry (see sun.py).

usage: python sun_fit.py camera.json shadow_candidates.csv sun_per_shot.csv
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera
from data import load_ball
import sun


def peaks(cam, pr, n_peaks=10):
    acc, ga, gb = sun.hough(cam, pr)
    a2 = acc.copy()
    out = []
    for _ in range(n_peaks):
        i = np.unravel_index(np.argmax(a2), a2.shape)
        if a2[i] <= 0:
            break
        s, inl, cond = sun.refine(cam, pr, (ga[i[0]], gb[i[1]]))
        out.append(dict(s=s, n=len(set(pr['frame'][inl])), cond=cond))
        a2[max(0, i[0] - 12):i[0] + 13, max(0, i[1] - 12):i[1] + 13] = 0
    return out


def fit_all(cam, ball, cands, verbose=False):
    shots, prs, pks = [], {}, {}
    for shot, g in ball[ball.obs].groupby('shot'):
        c = cands[(cands.frame >= g.frame.min()) & (cands.frame <= g.frame.max())]
        pr = sun.pairs_for_shot(cam, g, c)
        shots.append((shot, g.frame.nunique()))
        if pr is not None and len(pr['frame']) >= 10:
            prs[shot] = pr
            pks[shot] = peaks(cam, pr)
    # global consensus: the sun hardly moves during the match, false solutions scatter
    best = [max(p, key=lambda r: r['n'] * min(1, r['cond'] / 2)) for p in pks.values() if p]
    strong = np.array([r['s'] for r in best if r['n'] >= 40 and r['cond'] >= 4])
    cons = np.median(strong, 0)
    if verbose:
        print('consensus sun slope', cons, 'from', len(strong), 'shots')
    rows = []
    for shot, n_obs in shots:
        r = dict(shot=shot, n_obs=n_obs, a=np.nan, b=np.nan, n=0, cond=0.0)
        if shot in pks:
            near = [p for p in pks[shot] if np.linalg.norm(p['s'] - cons) < 0.15 and p['cond'] > 0.5]
            if near:
                p = max(near, key=lambda q: q['n'])
            else:
                s, inl, cond = sun.refine(cam, prs[shot], cons, tol=0.05)
                p = dict(s=s, n=len(set(prs[shot]['frame'][inl])), cond=cond)
            if p['n'] >= 6 and np.linalg.norm(p['s'] - cons) < 0.15:
                r.update(a=p['s'][0], b=p['s'][1], n=p['n'], cond=p['cond'])
        rows.append(r)
    d = pd.DataFrame(rows)
    # smooth over neighbouring shots (weights: number of supporting frames), keep own estimate when it is strong
    w = np.where(d.n >= 6, np.minimum(d.n, 150) * np.minimum(1, d.cond / 4), 0.0)
    for col in 'ab':
        v = d[col].values
        sm = np.full(len(d), np.nan)
        for i in range(len(d)):
            lo, hi = max(0, i - 4), min(len(d), i + 5)
            ww, vv = w[lo:hi], v[lo:hi]
            ok = ww > 0
            if ok.sum() >= 2:
                med = np.median(vv[ok])
                ok &= np.abs(vv - med) < 0.05
                sm[i] = np.sum(ww[ok] * vv[ok]) / np.sum(ww[ok])
        sm = pd.Series(sm).interpolate(limit_direction='both').values
        d[col + '_smooth'] = sm
        strong_own = (d.n >= 50) & (d.cond >= 6) & (np.abs(v - sm) < 0.04)
        d[col + '_fit'] = np.where(strong_own, 0.5 * (v + sm), sm)
    return d


if __name__ == '__main__':
    cam = Camera.from_json(sys.argv[1])
    ball = load_ball()
    cands = pd.read_csv(sys.argv[2])
    d = fit_all(cam, ball, cands, verbose=True)
    d.to_csv(sys.argv[3], index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
    d['elev_deg'] = np.degrees(np.arctan(1 / np.hypot(d.a_fit, d.b_fit)))
    d['azim_deg'] = np.degrees(np.arctan2(d.a_fit, d.b_fit))
    print(d.round(3).to_string())
