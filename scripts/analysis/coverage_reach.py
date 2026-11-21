"""Court coverage from the existing data: heat maps by game phase, and a reach model of what the defenders could
cover at each opponent attack.

Reach model: at the attack frame each defender stands at his feet point (team_spacing_v3: pose joints). He reaches a
spot if   distance - REACH <= SPEED * (flight - REACT)   where flight = frames from the attack to the next touch or
landing. REACT = 0.30 s (the pace-reaction peak measured in this project), REACH = 1.0 m (a step and a leg / dive).
SPEED is calibrated on the attacks the defence DID return: the smallest speed at which 95% of the returning
touches are inside the reachable area. The open area = cells of the defending half (inside the lines) nobody reaches.
Outputs data/analysis/coverage_reach.json (maps on 0.5 m cells in the fixed frame: ISR left, BRA right).
"""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS, REACT, REACH = 25, 0.30, 1.0
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
PL = ['Maor_Haas', 'Ron_Ben_Ishay', 'Franklin', 'Hiltinho']
r3 = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 3)
ral = pd.read_csv('data/points/rallies_v2.csv')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
rid = np.full(int(ral.end_frame.max()) + 2, -1)
for x in ral.itertuples():
    rid[x.start_frame:x.end_frame + 1] = x.rally
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'team', 'x', 'y']).drop_duplicates(['v2_frame', 'player'])
pos = pos[pos.v2_frame < len(rid)]; pos['rally'] = rid[pos.v2_frame.to_numpy()]; pos = pos[pos.rally > 0]
isr_left = pos[pos.team == 'ISR'].groupby('rally').x.mean().lt(8).to_dict()
flip = lambda r: not isr_left.get(int(r), True)
def nz(x, y, r):
    return (16 - x, 8 - y) if flip(r) else (x, y)

# ---------- phases: who has the ball (team of the last in-play touch) ----------
t = pd.read_csv('data/contacts/touches_master.csv'); t = t[(t.kind == 'player') & t.in_play & t.team.notna()].sort_values('v2_frame')
poss = np.full(len(rid), '', dtype=object)
for r_, g in t.groupby('rally'):
    fr = g.v2_frame.to_numpy(); tm = g.team.to_numpy(); e = int(ral.set_index('rally').end_frame[r_])
    for i in range(len(fr)):
        poss[fr[i]:(fr[i + 1] if i + 1 < len(fr) else e + 1)] = tm[i]
pos['poss'] = poss[pos.v2_frame.to_numpy()]
pos['phase'] = np.where(pos.poss == '', 'serve', np.where(pos.poss == pos.team, 'own ball', 'opponent ball'))
win = ral.set_index('rally').winner
pos['won'] = pos.rally.map(win) == pos.team
xx, yy = zip(*[nz(a, b, r) for a, b, r in zip(pos.x, pos.y, pos.rally)]); pos['nx'], pos['ny'] = xx, yy
XB = np.arange(-1, 17.01, 0.5); YB = np.arange(-1, 9.01, 0.5)
def hmap(g):
    H, _, _ = np.histogram2d(g.nx, g.ny, [XB, YB]); return H.T / max(len(g), 1)   # share of frames per cell
heat = {}
for p in PL:
    g = pos[pos.player == p]
    heat[p] = {k: np.round(hmap(s) * 1000, 2).tolist() for k, s in
               [('all', g), ('own ball', g[g.phase == 'own ball']), ('opponent ball', g[g.phase == 'opponent ball']), ('won', g[g.won]), ('lost', g[~g.won])]}
# partner overlap (Bhattacharyya coefficient of the two heat maps) and own-half area at 50 / 90 %
def area(H, q):
    v = np.sort(np.asarray(H).ravel())[::-1]; c = np.cumsum(v) / v.sum(); return float((np.searchsorted(c, q) + 1) * 0.25)
team = {}
for tm, (a, b) in {'ISR': ('Maor_Haas', 'Ron_Ben_Ishay'), 'BRA': ('Franklin', 'Hiltinho')}.items():
    team[tm] = {}
    for k in ('all', 'own ball', 'opponent ball', 'won', 'lost'):
        A = np.array(heat[a][k]); B = np.array(heat[b][k]); A = A / A.sum(); B = B / B.sum()
        team[tm][k] = dict(overlap=r3(np.sum(np.sqrt(A * B))), area50_a=area(A, .5), area50_b=area(B, .5), area90_a=area(A, .9), area90_b=area(B, .9))

# ---------- reach model at each opponent attack ----------
sp = pd.read_csv('data/analysis/team_spacing_v3.csv').query("moment == 'attack'").copy()
sp['v2'] = sp.frame.map(fm)
land = pd.read_csv('data/analysis/point_landings.csv').set_index('rally')
tt = t.set_index('touch')
ev = []
for x in sp.itertuples():
    if x.attack_touch not in tt.index:
        continue
    a = tt.loc[x.attack_touch]; f0 = int(a.v2_frame)
    nxt = t[(t.rally == x.rally) & (t.v2_frame > f0)]
    if len(nxt) and np.isfinite(nxt.iloc[0].ball_court_x_m):
        n0 = nxt.iloc[0]; dest = (n0.ball_court_x_m, n0.ball_court_y_m); f1 = int(n0.v2_frame); how = 'touched'
    elif x.rally in land.index and np.isfinite(land.loc[x.rally, 'x']) and np.isfinite(land.loc[x.rally, 'frame']):
        dest = (land.loc[x.rally, 'x'], land.loc[x.rally, 'y']); f1 = int(land.loc[x.rally, 'frame']); how = land.loc[x.rally, 'kind']
    else:
        continue
    fl = (f1 - f0) / FPS
    if not (0.2 < fl < 3):
        continue
    ev.append(dict(rally=int(x.rally), team=x.defending_team, outcome=x.outcome, conceded=bool(x.conceded), flight=fl, how=how,
                   p=[(x.p1_x, x.p1_y), (x.p2_x, x.p2_y)], dest=dest, gap=x.distance))
ev = pd.DataFrame(ev)
# calibrate SPEED on returned attacks: the returning touch must be reachable by one of the two
def need_speed(e):
    d = min(np.hypot(e.dest[0] - px, e.dest[1] - py) for px, py in e.p)
    return max(0.0, d - REACH) / max(e.flight - REACT, 0.05)
ev['need'] = ev.apply(need_speed, axis=1)
ret = ev[ev.outcome == 'returned']
SPEED = float(np.quantile(ret.need, 0.95))
print(f'{len(ev)} attacks with a known destination and flight; speed calibrated on {len(ret)} returns: {SPEED:.2f} m/s (95%)')
ev['reachable'] = ev.need <= SPEED
cells_x = np.arange(-0.75, 16.76, 0.5); cells_y = np.arange(-0.75, 8.76, 0.5)
GX, GY = np.meshgrid(cells_x, cells_y)
hole_sum = {'ISR': np.zeros_like(GX), 'BRA': np.zeros_like(GX)}; n_att = {'ISR': 0, 'BRA': 0}
open_area = []
for e in ev.itertuples():
    left_isr = isr_left.get(e.rally, True)
    # defending half in raw coordinates
    dl = (e.team == 'ISR') == left_isr
    half = (GX < 8.0) if dl else (GX > 8.0)
    half &= (GX > 0) & (GX < 16) & (GY > 0) & (GY < 8)   # inside the lines: a ball landing out is the attacker's error
    reach = np.zeros_like(GX, bool)
    for px, py in e.p:
        reach |= np.hypot(GX - px, GY - py) - REACH <= SPEED * max(e.flight - REACT, 0)
    hole = half & ~reach
    open_area.append(hole.sum() * 0.25)
    h = hole.astype(float)
    if flip(e.rally):
        h = h[::-1, ::-1]   # to the fixed frame
    hole_sum[e.team] += h; n_att[e.team] += 1
ev['open_m2'] = open_area
holes = {tm: np.round(hole_sum[tm] / max(n_att[tm], 1), 3).tolist() for tm in hole_sum}
# does the open area matter? concede rate when the ball went to an unreachable spot, and by open area
res = ev.groupby('reachable').conceded.agg(['mean', 'size']).round(3)
print(res)
ev['open_bin'] = pd.cut(ev.open_m2, [-0.1, 2, 6, 100], labels=['< 2', '2-6', '> 6'])
print(ev.groupby(['team', 'open_bin'], observed=True).conceded.agg(['mean', 'size']).round(2))
print(ev.groupby('team').open_m2.median())
print({tm: {k: team[tm][k]['overlap'] for k in team[tm]} for tm in team})
out = dict(cell=0.5, x0=-1, y0=-1, heat=heat, team=team, holes=dict(x0=-1, y0=-1, cell=0.5, maps=holes, n=n_att),
           model=dict(speed=round(SPEED, 2), react=REACT, reach=REACH, n_returns=len(ret)),
           attacks=[dict(r=e.rally, tm=e.team, o=e.outcome, c=e.conceded, fl=round(e.flight, 2), open=round(e.open_m2, 2), reach=bool(e.reachable),
                         dest=list(np.round(nz(e.dest[0], e.dest[1], e.rally), 2)), p=[list(np.round(nz(a, b, e.rally), 2)) for a, b in e.p]) for e in ev.itertuples()])
json.dump(out, open('data/analysis/coverage_reach.json', 'w'), separators=(',', ':'))
print('written', os.path.getsize('data/analysis/coverage_reach.json') // 1024, 'KB')
