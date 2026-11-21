"""Data for the match-analysis interface (site/analysis_ui/): the court-story layers plus three metrics.

Set win probability (ISR's chance to win the set before every rally)
  Score-only model: the serving team wins a rally with the match's pooled serving rate (shrunk toward 0.5 with 5
  rallies of prior weight), the same for both teams, so the curve reflects the score and the serve, not hindsight. Sets go to 18, win by 2. The rally winner serves next.
  P(score, server) by dynamic programming; deuce in closed form. Win-probability added (WPA) per rally = change
  in the rally winner's set win probability.

Touch value (how much a touch raised the toucher's team's chance to win the rally)
  State before a touch = (touch number in the possession 1/2/3, ball zone by distance from the net: net < 2 m,
  mid 2-5 m, back > 5 m). V(state) = share of rallies won by the team about to make that touch, shrunk toward
  V(touch number) with 8 touches of prior weight. After the touch: the next touch's state (V for the same team,
  1 - V for the opponent), or 1 / 0 if the rally ended with this team winning / losing.
  Touch value = V(after) - V(before), in percentage points of rally-win chance.

Attack quality (chance that an attack wins the point, from how it was struck)
  Ridge logistic regression on body part (foot / head / other), angle to the line (< 15, 15-35, > 35 deg),
  contact height (< 2.3, 2.3-2.6, > 2.6 m) and length of the shot. Fitted on this match's 126 attacks only.
  Quality = the model's probability; the outcome itself is not an input.
usage: python scripts/site/build_analysis_ui.py -> site/analysis_ui/data.json
"""
import os, json
from functools import lru_cache
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
cs = json.load(open('site/court_story/data.json'))
out = {k: cs[k] for k in ('rallies', 'sideout', 'heatz', 'move', 'rose', 'lines', 'landz', 'spacez', 'netz', 'stature', 'chest')}
isr_left = {int(k): v for k, v in cs['isr_left'].items()}
r2 = lambda v, n=3: None if v is None or not np.isfinite(v) else round(float(v), n)
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}

# ------------------------------------------------------------------ set win probability
ral = pd.read_csv('data/points/rallies_v2.csv')
act = pd.read_csv('data/analysis/touch_actions_v3.csv')
ral['server'] = ral.rally.map(act[act.action == 'serve'].groupby('rally').team.first())
k_a = int(((ral.server == 'ISR') & (ral.winner == 'ISR')).sum()); n_a = int((ral.server == 'ISR').sum())
k_b = int(((ral.server == 'BRA') & (ral.winner == 'ISR')).sum()); n_b = int((ral.server == 'BRA').sum())
# score-only model (no hindsight about which team is better): both teams get the match's pooled serving rate
k_srv = int((ral.server == ral.winner).sum()); n_srv = len(ral)
p_srv = (k_srv + 5 * 0.5) / (n_srv + 5)
A = p_srv; B = 1 - p_srv
TARGET = 18


@lru_cache(None)
def wp(i, b, srv):
    """ISR's chance to win the set from score i-b with `srv` serving next."""
    if i >= TARGET and i - b >= 2:
        return 1.0
    if b >= TARGET and b - i >= 2:
        return 0.0
    if i >= TARGET - 1 and b >= TARGET - 1 and i == b:   # deuce: two in a row needed
        # from deuce with server s: win two straight or come back to deuce
        def two(s):
            p = A if s == 'ISR' else B
            win_first = p * A + (1 - p) * 0   # ISR wins rally (then serves), then wins again with ISR serving
            # states after one rally: adv ISR (ISR serving) or adv BRA (BRA serving)
            return p, win_first
        pI, _ = A, None
        # solve x_s = P(win from deuce, server s): x_s = p_s*(A + (1-A)*x_B) + (1-p_s)*(B*x_I)
        # unknowns x_I (ISR serving), x_B (BRA serving)
        M = np.array([[1 - A * (1 - A) * 0 - 0, 0], [0, 1]], float)
        # write explicitly
        a11 = 1 - (1 - A) * B; a12 = -A * (1 - A)
        a21 = -(1 - B) * B; a22 = 1 - B * (1 - A)
        rhs = np.array([A * A, B * A])
        xI, xB = np.linalg.solve(np.array([[a11, a12], [a21, a22]]), rhs)
        return float(xI if srv == 'ISR' else xB)
    p = A if srv == 'ISR' else B
    return p * wp(i + 1, b, 'ISR') + (1 - p) * wp(i, b + 1, 'BRA')


wpl = []
for s_, g in ral.groupby('set'):
    i = b = 0
    for x in g.sort_values('rally').itertuples():
        before = wp(i, b, x.server)
        i, b = x.ISR_score_after, x.BRA_score_after
        nxt = 'ISR' if x.winner == 'ISR' else 'BRA'
        after = wp(i, b, nxt)
        wpl.append(dict(r=int(x.rally), s=int(s_), sv=x.server, w=x.winner, i=int(i), b=int(b), before=r2(before), after=r2(after),
                        wpa=r2((after - before) if x.winner == 'ISR' else (before - after))))
out['wp'] = dict(model=dict(p_serve=r2(p_srv), raw=f'{k_srv}/{n_srv} rallies won by the server', target=TARGET), rallies=wpl)

# ------------------------------------------------------------------ touch value
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[(t.kind == 'player') & t.in_play & t.player.notna()].sort_values('v2_frame').copy()
win = ral.set_index('rally').winner
t['won'] = (t.rally.map(win) == t.team).astype(int)
t['k'] = t.touch_in_possession.clip(1, 3).fillna(1).astype(int)


def nz(xx, yy, r):
    return (xx, yy) if isr_left.get(int(r), True) else (16 - xx, 8 - yy)


t['nx'], t['ny'] = zip(*[nz(a, b_, r) if np.isfinite(a) else (np.nan, np.nan) for a, b_, r in zip(t.ball_court_x_m, t.ball_court_y_m, t.rally)])
dn = (t.nx - 8).abs()
t['zone'] = np.where(dn < 2, 'net', np.where(dn < 5, 'mid', 'back'))
Vk = t.groupby('k').won.mean()
cnt = t.groupby(['k', 'zone']).won.agg(['sum', 'size'])
V = {(k, z): (row['sum'] + 8 * Vk[k]) / (row['size'] + 8) for (k, z), row in cnt.iterrows()}
vals = []
rows = list(t.itertuples())
for j, x in enumerate(rows):
    vb = V[(x.k, x.zone)]
    nx_ = rows[j + 1] if j + 1 < len(rows) and rows[j + 1].rally == x.rally else None
    if nx_ is None:
        va = float(x.won)
    else:
        vn = V[(nx_.k, nx_.zone)]
        va = vn if nx_.team == x.team else 1 - vn
    vals.append(dict(p=x.player, tm=x.team, a=x.action, s=x.action_subtype if isinstance(x.action_subtype, str) else '', b=x.body_part,
                     r=int(x.rally), f=int(x.v2_frame), x=r2(x.nx, 2), y=r2(x.ny, 2), h=r2(x.ball_height_m, 2), k=int(x.k), zone=x.zone,
                     vb=r2(vb), va=r2(va), val=r2(va - vb)))
out['touches'] = vals
out['touch_model'] = {f'{k}-{z}': r2(v) for (k, z), v in V.items()}

# ------------------------------------------------------------------ attack quality
az = [a for a in cs['attackz']]
def feats(a):
    b = a['b']; ang = a['ang']; h = a['h'] if a['h'] is not None else 2.4; L = a.get('len') or np.hypot(a['x1'] - a['x0'], a['y1'] - a['y0'])
    return [1.0, b == 'foot', b == 'head', 15 <= ang < 35, ang >= 35, h < 2.3, h > 2.6, (L - 6) / 3]
FN = ['intercept', 'foot', 'head', 'angle 15-35', 'angle > 35', 'contact < 2.3 m', 'contact > 2.6 m', 'length (per 3 m)']
X = np.array([feats(a) for a in az], float); y = np.array([a['s'] in ('kill', 'failed_defence') for a in az], float)
w = np.zeros(X.shape[1]); lam = 2.0
for _ in range(3000):
    p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + lam * np.r_[0, w[1:]]
    Hs = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
    w -= np.linalg.solve(Hs, g)
    if np.abs(g).max() < 1e-8:
        break
q = 1 / (1 + np.exp(-X @ w))
# leave-one-out style honesty check: in-sample Brier vs base-rate Brier
brier = float(np.mean((q - y) ** 2)); base = float(np.mean((y.mean() - y) ** 2))
for a, qq in zip(az, q):
    a['q'] = r2(qq)
out['attacks'] = az
out['attack_model'] = dict(coef={n: r2(c) for n, c in zip(FN, w)}, brier=r2(brier), brier_base=r2(base), n=len(az), rate=r2(y.mean()))
os.makedirs('site/analysis_ui', exist_ok=True)
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating,)):
        return clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
json.dump(clean(out), open('site/analysis_ui/data.json', 'w'), separators=(',', ':'), allow_nan=False)
print('WP model', out['wp']['model']); print('touch V', out['touch_model']); print('attack model', out['attack_model'])
tv = pd.DataFrame(vals); print(tv.groupby(['p']).val.agg(['sum', 'mean', 'size']).round(3)); print(tv.groupby('a').val.mean().round(3))
wpd = pd.DataFrame(wpl); print(wpd.sort_values('wpa', key=abs, ascending=False).head(5).to_string())
print(os.path.getsize('site/analysis_ui/data.json') // 1024, 'KB')
