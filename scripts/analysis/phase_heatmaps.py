"""Heat maps of every player by game phase, in the player's own view of his half (v3 video).

Phases (per team, every frame of every rally; the team that has the ball = team of the last in-play touch,
data/contacts/touches_master.csv, hand-labelled player per touch):
  serve      the serving team, from the rally start to the receiver's first touch (the main camera picks most rallies
             up late, so this phase is short and only shows where the server's partner stands)
  reception  the receiving team over the same frames, and later in the rally while the opponent's attack is on its way
             (from the opponent's attack touch to our first touch)
  build      our ball, touches before the attack (reception done, set on the way)
  attack     from our attack touch until the opponent touches the ball (the attacker covering after the hit)
  defence    the opponent has the ball and has not attacked yet (we set up against their attack)

Only measured frames are used (accuracy first): the ground point comes from the pose (ankles, planted foot or hips),
not from a box bottom, the player is not occluded by his teammate or cut by the image border, and the frame is not a
gap bridged by the smoother. Smoothed positions (player_positions_v3.csv, Kalman + RTS) are used for those frames.

Own view: x' = distance from the own baseline (0) to the net (8), y' = across the court as the player sees it facing
the net (0 = his left sideline, 8 = his right). Each frame adds a Gaussian of SIGMA m (position uncertainty ~0.1-0.2 m
plus some smoothing) on a 0.1 m grid; maps are in seconds per square metre. Points outside the lines are kept
(players often stand behind the baseline).

Outputs data/analysis/phase_heatmaps.json (grids), data/analysis/phase_heatmaps_summary.csv (per player x phase:
seconds, mean distance from net, mean side position, share in the left half, area holding 50% / 90% of the time) and
data/analysis/phase_heatmaps_pairs.csv (per team x phase: overlap of the two partners' maps, 0 = separate zones,
1 = the same zone), data/analysis/phase_heatmaps.png.
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS = 25
SIGMA = 0.25
CELL = 0.1
XR = (-2.0, 8.5)    # own view: from 2 m behind the baseline to just past the net
YR = (-1.5, 9.5)
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Ron_Ben_Ishay': 'ISR', 'Maor_Haas': 'ISR'}
PAIRS = {'ISR': ('Ron_Ben_Ishay', 'Maor_Haas'), 'BRA': ('Franklin', 'Hiltinho')}
PHASES = ['serve', 'reception', 'build', 'attack', 'defence']
GOOD_SRC = {'ankles', 'ankle', 'planted', 'hips'}

ral = pd.read_csv('data/points/rallies_v3video.csv')
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[(t.kind == 'player') & t.in_play & t.team.notna()].sort_values('frame')
pos = pd.read_csv('data/analysis/player_positions_v3.csv',
                  usecols=['frame', 'player', 'team', 'src', 'occluded', 'edge', 'raw_x', 'x', 'y'])
pos = pos.drop_duplicates(['frame', 'player'])

# ---------- phase of each team in every rally frame ----------
rows = []
for r in ral.itertuples():
    g = t[t.rally == r.rally]
    if g.empty:
        continue
    f0, f1 = int(r.start_frame), int(r.end_frame)
    recv = g.team.iloc[0]
    srv = 'ISR' if recv == 'BRA' else 'BRA'
    fr = np.arange(f0, f1 + 1)
    ph = {tm: np.empty(len(fr), dtype=object) for tm in ('ISR', 'BRA')}
    first = int(g.frame.iloc[0])
    pre = fr < first
    ph[srv][pre] = 'serve'; ph[recv][pre] = 'reception'
    tf, tt, ta = g.frame.to_numpy(), g.team.to_numpy(), g.action.to_numpy()
    for i in range(len(tf)):
        a, b = tf[i], (tf[i + 1] if i + 1 < len(tf) else f1 + 1)
        sel = (fr >= a) & (fr < b)
        own, opp = tt[i], ('ISR' if tt[i] == 'BRA' else 'BRA')
        if ta[i] == 'attack':
            ph[own][sel] = 'attack'; ph[opp][sel] = 'reception'
        else:
            ph[own][sel] = 'build'; ph[opp][sel] = 'defence'
    # side of each team in this rally (for the own view)
    pr = pos[(pos.frame >= f0) & (pos.frame <= f1)]
    left = pr.groupby('team').x.median().lt(8).to_dict()
    for tm in ('ISR', 'BRA'):
        rows.append(pd.DataFrame(dict(frame=fr, team=tm, rally=r.rally, set=r.set, phase=ph[tm],
                                      left=left.get(tm, tm == 'ISR'), won=r.winner == tm)))
phase = pd.concat(rows)
d = pos.merge(phase, on=['frame', 'team'])
d['strict'] = d.src.isin(GOOD_SRC) & ~d.occluded & ~d.edge & d.raw_x.notna()
# own view
d['xo'] = np.where(d.left, d.x, 16 - d.x)                  # 0 = own baseline, 8 = net
d['yo'] = np.where(d.left, d.y, 8 - d.y)                   # facing +x on the left half, the far sideline (y = 0) is on the left
print('rally frames per player:', len(d) // 4, ' measured (strict):', f'{d.strict.mean():.1%}')
s = d[d.strict]

# ---------- maps ----------
gx = np.arange(XR[0], XR[1] + 1e-9, CELL); gy = np.arange(YR[0], YR[1] + 1e-9, CELL)
def kde(g):
    if len(g) == 0:
        return np.zeros((len(gx), len(gy)))
    H, _, _ = np.histogram2d(g.xo, g.yo, [np.r_[gx - CELL / 2, gx[-1] + CELL / 2], np.r_[gy - CELL / 2, gy[-1] + CELL / 2]])
    k = np.arange(-int(3 * SIGMA / CELL), int(3 * SIGMA / CELL) + 1) * CELL
    w = np.exp(-k ** 2 / (2 * SIGMA ** 2)); w /= w.sum()
    H = np.apply_along_axis(lambda v: np.convolve(v, w, 'same'), 0, H)
    H = np.apply_along_axis(lambda v: np.convolve(v, w, 'same'), 1, H)
    return H / FPS / CELL ** 2          # seconds per m^2
def area(H, q):
    v = np.sort(H.ravel())[::-1]; c = np.cumsum(v) / max(v.sum(), 1e-9)
    return round(float((np.searchsorted(c, q) + 1) * CELL ** 2), 2)

maps, summ = {}, []
for p in TEAM:
    maps[p] = {}
    for ph_ in ['all'] + PHASES:
        g = s[s.player == p] if ph_ == 'all' else s[(s.player == p) & (s.phase == ph_)]
        H = kde(g); maps[p][ph_] = H
        summ.append(dict(player=p, team=TEAM[p], phase=ph_, seconds=round(len(g) / FPS, 1), frames=len(g),
                         from_net_m=round(8 - g.xo.mean(), 2) if len(g) else None,
                         side_m=round(g.yo.mean(), 2) if len(g) else None,
                         left_half=round((g.yo < 4).mean(), 2) if len(g) else None,
                         area50_m2=area(H, .5), area90_m2=area(H, .9)))
summ = pd.DataFrame(summ)
pairs = []
for tm, (a, b) in PAIRS.items():
    for ph_ in ['all'] + PHASES:
        A, B = maps[a][ph_], maps[b][ph_]
        if A.sum() == 0 or B.sum() == 0:
            continue
        A, B = A / A.sum(), B / B.sum()
        pairs.append(dict(team=tm, phase=ph_, overlap=round(float(np.sum(np.sqrt(A * B))), 3),
                          a=a, b=b,
                          a_left=summ.query('player == @a and phase == @ph_').left_half.iloc[0],
                          b_left=summ.query('player == @b and phase == @ph_').left_half.iloc[0],
                          a_from_net=summ.query('player == @a and phase == @ph_').from_net_m.iloc[0],
                          b_from_net=summ.query('player == @b and phase == @ph_').from_net_m.iloc[0]))
pairs = pd.DataFrame(pairs)
summ.to_csv('data/analysis/phase_heatmaps_summary.csv', index=False)
pairs.to_csv('data/analysis/phase_heatmaps_pairs.csv', index=False)
json.dump(dict(cell=CELL, sigma=SIGMA, x0=XR[0], y0=YR[0], nx=len(gx), ny=len(gy), units='seconds per m2',
               axes='x = distance from own baseline (net at 8), y = across, 0 = player\'s left',
               maps={p: {k: np.round(v, 2).tolist() for k, v in m.items()} for p, m in maps.items()}),
          open('data/analysis/phase_heatmaps.json', 'w'), separators=(',', ':'))
print(summ.to_string(index=False)); print(pairs.to_string(index=False))

# ---------- figure ----------
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 6, figsize=(18, 13))
    for i, p in enumerate(TEAM):
        for j, ph_ in enumerate(['all'] + PHASES):
            H = maps[p][ph_]; a_ = ax[i, j]
            a_.imshow(H, origin='lower', extent=[YR[0], YR[1], XR[0], XR[1]], cmap='magma', aspect='equal',
                      vmax=np.quantile(H[H > 0], .995) if (H > 0).any() else 1)
            a_.plot([0, 8, 8, 0, 0], [0, 0, 8, 8, 0], color='w', lw=1); a_.axhline(8, color='c', lw=2)
            sec = summ.query('player == @p and phase == @ph_').seconds.iloc[0]
            a_.set_title(f'{p} - {ph_} ({sec:.0f} s)', fontsize=9); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('Time per m2 by phase, own view (net at top, player\'s left on the left)')
    fig.tight_layout(); fig.savefig('data/analysis/phase_heatmaps.png', dpi=80)
except ImportError:
    print('matplotlib missing: no figure')
