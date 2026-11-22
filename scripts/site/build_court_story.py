"""Collect the project's analysis data into one JSON for the court-story page (site/court_story/).
Everything is computed here from the repo's CSVs; the page only draws.
usage: python scripts/site/build_court_story.py  -> site/court_story/data.json
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS = 25
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
PL = ['Maor_Haas', 'Ron_Ben_Ishay', 'Franklin', 'Hiltinho']
r2 = lambda v, n=2: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)
out = {}

# ---- score flow --------------------------------------------------------------------------------------------------
ral = pd.read_csv('data/points/rallies_v2.csv')
act = pd.read_csv('data/analysis/touch_actions_v3.csv')
server = act[act.action == 'serve'].groupby('rally').team.first()
ral['server'] = ral.rally.map(server)
out['rallies'] = [dict(r=int(x.rally), s=int(x.set), w=x.winner, sv=x.server, i=int(x.ISR_score_after), b=int(x.BRA_score_after),
                       d=r2(x.duration_sec, 1), t=r2(x.start_time_sec, 1)) for x in ral.itertuples()]
inplay = np.zeros(int(ral.end_frame.max()) + 2, bool)
for x in ral.itertuples():
    inplay[x.start_frame:x.end_frame + 1] = True
so = {}
for tm in ('ISR', 'BRA'):
    rec = ral[ral.server != tm]
    so[tm] = dict(receive=int(len(rec)), sideout=int((rec.winner == tm).sum()),
                  serve=int((ral.server == tm).sum()), serve_won=int(((ral.server == tm) & (ral.winner == tm)).sum()))
out['sideout'] = so

# ---- positions: heatmaps (play time), mean spot, speed --------------------------------------------------------
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y', 'speed']).drop_duplicates(['v2_frame', 'player'])
pos = pos[pos.v2_frame < len(inplay)]
pp = pos[inplay[pos.v2_frame.to_numpy()]]
# teams change ends between sets: rotate the right half by 180 deg so every player plays on the left half (net at x=8)
XB = np.arange(-1, 8.01, 0.5); YB = np.arange(-1, 9.01, 0.5)
heat = {}
for p in PL:
    g = pp[pp.player == p].copy()
    rgt = g.x > 8
    g.loc[rgt, 'x'] = 16 - g.loc[rgt, 'x']; g.loc[rgt, 'y'] = 8 - g.loc[rgt, 'y']
    H, _, _ = np.histogram2d(g.x, g.y, [XB, YB])
    H = H / H.max()
    heat[p] = dict(grid=np.round(H.T * 99).astype(int).tolist(), mx=r2(g.x.mean()), my=r2(g.y.mean()),
                   sx=r2(g.x.std()), sy=r2(g.y.std()))
out['heat'] = dict(x0=-1, y0=-1, cell=0.5, nx=len(XB) - 1, ny=len(YB) - 1, players=heat)

mv = pd.read_csv('data/analysis/movement_by_set.csv')
spd = pd.read_csv('data/analysis/player_speed_rallies_v3.csv')
cov = pd.read_csv('data/analysis/coverage_match_v3.csv').set_index('player')
mov = {}
for p in PL:
    g = spd[spd.player == p]
    m = mv[mv.player == p]
    mov[p] = dict(dist=r2(g.distance.sum(), 0), top_kmh=r2(m.top_speed_kmh.max(), 1), m_per_min=r2((m.distance_m.sum() / m.rally_sec.sum()) * 60, 0),
                  sprint_pct=r2((m.sprint_pct * m.rally_sec).sum() / m.rally_sec.sum(), 1), area90=r2(cov.loc[p, 'area90'], 1),
                  won_dist=r2(g[g.winner == TEAM[p]].distance.mean(), 1), lost_dist=r2(g[g.winner != TEAM[p]].distance.mean(), 1))
out['move'] = mov

# ---- touches: body parts, heights ----------------------------------------------------------------------------------
t = pd.read_csv('data/contacts/touches_master.csv')
t = t[t.kind == 'player']
stat = {k: v['stature_m'] for k, v in json.load(open('data/analysis/player_statures.json')).items()}
out['stature'] = {p: r2(stat[p]) for p in PL}
out['touches'] = [dict(p=x.player, a=x.action, s=x.action_subtype if isinstance(x.action_subtype, str) else '', b=x.body_part,
                       h=r2(x.ball_height_m), x=r2(x.ball_court_x_m), y=r2(x.ball_court_y_m), r=int(x.rally), f=int(x.v2_frame))
                  for x in t.itertuples() if isinstance(x.player, str) and x.action in ('reception', 'set', 'attack')]

# ---- attacks: from the touch to where the ball went next (next touch or landing) ---------------------------------
land = pd.read_csv('data/analysis/point_landings.csv').set_index('rally')
tt = t[t.action.isin(['reception', 'set', 'attack', 'dead'])].sort_values('v2_frame')
att = []
for x in tt[tt.action == 'attack'].itertuples():
    nxt = tt[(tt.rally == x.rally) & (tt.v2_frame > x.v2_frame)]
    to, how = None, None
    if len(nxt) and np.isfinite(nxt.iloc[0].ball_court_x_m):
        n0 = nxt.iloc[0]; to = (n0.ball_court_x_m, n0.ball_court_y_m); how = 'touched'
    elif x.rally in land.index and np.isfinite(land.loc[x.rally, 'x']):
        to = (land.loc[x.rally, 'x'], land.loc[x.rally, 'y']); how = land.loc[x.rally, 'kind']
    if to is None or not np.isfinite(x.ball_court_x_m):
        continue
    x0, y0, x1, y1 = x.ball_court_x_m, x.ball_court_y_m, to[0], to[1]
    if x1 < x0:  # rotate 180 deg so every attack goes left -> right
        x0, y0, x1, y1 = 16 - x0, 8 - y0, 16 - x1, 8 - y1
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) < 1.0:   # did not cross the net (blocked / fell short)
        continue
    ang = float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # 0 = straight down the line, 90 = along the net
    att.append(dict(p=x.player, tm=x.team, b=x.body_part, s=x.action_subtype, x0=r2(x0), y0=r2(y0),
                    x1=r2(x1), y1=r2(y1), h=r2(x.ball_height_m), ang=r2(ang, 1), len=r2(np.hypot(dx, dy)), how=how))
out['attacks'] = att

# ---- ball flights, side view (x vs height), in play, every 2nd frame -------------------------------------------
bh = pd.read_csv('data/analysis/ball_height_v2.csv')
bh = bh[(bh.frame < len(inplay)) & bh.flight_id.notna() & (bh.method == 'flight_fit')]
bh = bh[inplay[bh.frame.to_numpy()]]
fl, cross = [], []
for fid, g in bh.groupby('flight_id'):
    g = g.sort_values('frame')
    if len(g) < 6:
        continue
    xs, hs = g.court_x_m.to_numpy(), g.height_m.to_numpy()
    sel = g.iloc[::2]
    fl.append(dict(dir=int(np.sign(xs[-1] - xs[0])), pts=[[r2(a, 2), r2(b, 2)] for a, b in zip(sel.court_x_m, sel.height_m)]))
    for i in range(len(xs) - 1):
        if (xs[i] - 8) * (xs[i + 1] - 8) < 0:
            w = (8 - xs[i]) / (xs[i + 1] - xs[i])
            cross.append(dict(h=r2(hs[i] + w * (hs[i + 1] - hs[i])), dir=int(np.sign(xs[i + 1] - xs[i])), f=int(g.frame.iloc[i])))
out['flights'] = fl
out['net_cross'] = cross

# ---- partner spacing and formation -------------------------------------------------------------------------------
sp = pd.read_csv('data/analysis/team_spacing_v3.csv')
sp = sp[sp.moment == 'attack']
out['spacing'] = [dict(tm=x.defending_team, d=r2(x.distance), c=bool(x.conceded), p1=[r2(x.p1_x), r2(x.p1_y)], p2=[r2(x.p2_x), r2(x.p2_y)])
                  for x in sp.itertuples() if np.isfinite(x.distance)]
fm = pd.read_csv('data/analysis/formation_points.csv')
out['formation'] = fm.groupby(['team', 'formation', 'result']).size().unstack(fill_value=0).reset_index().to_dict('records')

# ---- landings -----------------------------------------------------------------------------------------------------
lr = pd.read_csv('data/analysis/point_landings.csv')
out['landings'] = [dict(r=int(x.rally), w=x.winner, l=x.loser, x=r2(x.x), y=r2(x.y), k=x.kind, p=x.last_touch_player, b=x.last_touch_part)
                   for x in lr.itertuples()]

# ---- body orientation (whole match, play time): shoulders and eyes, 15-degree rose -------------------------------
bo = pd.read_csv('data/analysis/body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'angle_to_net_deg'])
bo = bo[bo.frame < len(inplay)]
bo = bo[inplay[bo.frame.to_numpy()]]
bo = bo.groupby(['player', 'part', 'frame']).angle_to_net_deg.mean().reset_index()
rose = {}
B = np.arange(-180, 181, 15)
for p in PL:
    rose[p] = {}
    for part in ('shoulders', 'eyes', 'feet'):
        v = bo[(bo.player == p) & (bo.part == part)].angle_to_net_deg
        h = np.histogram(v, B)[0]
        rose[p][part] = dict(bins=(h / h.sum()).round(4).tolist(), square=r2((v.abs() < 30).mean() * 100, 0),
                             side=r2(((v.abs() >= 60) & (v.abs() <= 120)).mean() * 100, 0), back=r2((v.abs() > 150).mean() * 100, 0))
out['rose'] = rose

# ---- line distances (play time) ------------------------------------------------------------------------------------
ls = pd.read_csv('data/analysis/line_distances_summary_players.csv').set_index('player')
out['lines'] = {p: dict(base=r2(ls.loc[p, 'baseline_median_m']), far=r2(ls.loc[p, 'far sideline_median_m']),
                        near=r2(ls.loc[p, 'near sideline_median_m']), net=r2(ls.loc[p, 'net_median_m']), out=r2(ls.loc[p, 'outside_pct'], 1))
                for p in PL}

# ---- reaction (first 30 s) -----------------------------------------------------------------------------------------
ex = pd.read_csv('data/analysis/pace_reaction_excess_0_749.csv')
out['reaction'] = ex.to_dict('records')

# ---- chest touches -------------------------------------------------------------------------------------------------
ch = pd.read_csv('data/analysis/chest_profile_by_player.csv').set_index('player')
out['chest'] = {p: dict(n=int(ch.loc[p, 'chest_touches']), h=r2(ch.loc[p, 'ball_height_m']), won=r2(ch.loc[p, 'point_won_pct'], 0),
                        descent=r2(ch.loc[p, 'ball_descent_deg'], 0), vin=r2(ch.loc[p, 'ball_speed_in'], 1)) for p in PL if p in ch.index}

# ================= one fixed court frame: ISR always on the left half, BRA on the right =================
# teams change ends every 6 points; a rally with ISR on the right is rotated 180 deg (x -> 16-x, y -> 8-y).
pos_all = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'team', 'x', 'y']).drop_duplicates(['v2_frame', 'player'])
rid = np.full(len(inplay), -1)
for x in ral.itertuples():
    rid[x.start_frame:x.end_frame + 1] = x.rally
pos_all = pos_all[pos_all.v2_frame < len(inplay)]
pos_all['rally'] = rid[pos_all.v2_frame.to_numpy()]
pos_all = pos_all[pos_all.rally > 0]
isr_left = pos_all[pos_all.team == 'ISR'].groupby('rally').x.mean().lt(8).to_dict()
out['isr_left'] = {int(k): bool(v) for k, v in isr_left.items()}


def nz(xx, yy, rally):
    """to the fixed frame (ISR left)"""
    if isr_left.get(int(rally), True):
        return xx, yy
    return 16 - xx, 8 - yy


X2 = np.arange(-1, 17.01, 0.5); Y2 = np.arange(-1, 9.01, 0.5)
hz = {}
for p in PL:
    g = pos_all[pos_all.player == p]
    flip = ~g.rally.map(isr_left).fillna(True).astype(bool)
    xx = np.where(flip, 16 - g.x, g.x); yy = np.where(flip, 8 - g.y, g.y)
    H, _, _ = np.histogram2d(xx, yy, [X2, Y2]); H = H / H.max()
    hz[p] = dict(grid=np.round(H.T * 99).astype(int).tolist(), mx=r2(np.mean(xx)), my=r2(np.mean(yy)), sx=r2(np.std(xx)), sy=r2(np.std(yy)))
out['heatz'] = dict(x0=-1, y0=-1, cell=0.5, players=hz)

# touches in the fixed frame
tz = []
for x in t.itertuples():
    if not isinstance(x.player, str) or x.action not in ('reception', 'set', 'attack') or not np.isfinite(x.ball_court_x_m):
        continue
    a, b = nz(x.ball_court_x_m, x.ball_court_y_m, x.rally)
    tz.append(dict(p=x.player, a=x.action, s=x.action_subtype if isinstance(x.action_subtype, str) else '', b=x.body_part, h=r2(x.ball_height_m), x=r2(a), y=r2(b), r=int(x.rally)))
out['touchz'] = tz

# attacks in the fixed frame (attacker on his own half)
az = []
for x in tt[tt.action == 'attack'].itertuples():
    nxt = tt[(tt.rally == x.rally) & (tt.v2_frame > x.v2_frame)]
    to = None
    if len(nxt) and np.isfinite(nxt.iloc[0].ball_court_x_m):
        to = (nxt.iloc[0].ball_court_x_m, nxt.iloc[0].ball_court_y_m)
    elif x.rally in land.index and np.isfinite(land.loc[x.rally, 'x']):
        to = (land.loc[x.rally, 'x'], land.loc[x.rally, 'y'])
    if to is None or not np.isfinite(x.ball_court_x_m):
        continue
    x0, y0 = nz(x.ball_court_x_m, x.ball_court_y_m, x.rally); x1, y1 = nz(to[0], to[1], x.rally)
    if abs(x1 - x0) < 1.0:
        continue
    ang = float(np.degrees(np.arctan2(abs(y1 - y0), abs(x1 - x0))))
    az.append(dict(p=x.player, tm=x.team, b=x.body_part, s=x.action_subtype, x0=r2(x0), y0=r2(y0), x1=r2(x1), y1=r2(y1), h=r2(x.ball_height_m), ang=r2(ang, 1), r=int(x.rally)))
out['attackz'] = az

# landings in the fixed frame
lz = []
for x in lr.itertuples():
    if np.isfinite(x.x):
        a, b = nz(x.x, x.y, x.rally)
        lz.append(dict(r=int(x.rally), w=x.winner, x=r2(a), y=r2(b), k=x.kind, p=x.last_touch_player, b=x.last_touch_part))
out['landz'] = lz

# partner spacing at the opponent's attack, fixed frame
sz = []
for x in pd.read_csv('data/analysis/team_spacing_v3.csv').query("moment == 'attack'").itertuples():
    if not np.isfinite(x.distance):
        continue
    a1, b1 = nz(x.p1_x, x.p1_y, x.rally); a2, b2 = nz(x.p2_x, x.p2_y, x.rally)
    sz.append(dict(tm=x.defending_team, d=r2(x.distance), c=bool(x.conceded), x1=r2(a1), y1=r2(b1), x2=r2(a2), y2=r2(b2), r=int(x.rally)))
out['spacez'] = sz

# net crossings: where along the net (y) and how high, fixed frame; from = team that sent it
bh2 = pd.read_csv('data/analysis/ball_height_v2.csv')
bh2 = bh2[(bh2.frame < len(inplay)) & bh2.flight_id.notna() & (bh2.method == 'flight_fit')]
bh2 = bh2[inplay[bh2.frame.to_numpy()]]
nzc = []
for fid, g in bh2.groupby('flight_id'):
    g = g.sort_values('frame'); xs, ys, hs, fr = g.court_x_m.to_numpy(), g.court_y_m.to_numpy(), g.height_m.to_numpy(), g.frame.to_numpy()
    for i in range(len(xs) - 1):
        if (xs[i] - 8) * (xs[i + 1] - 8) < 0:
            w = (8 - xs[i]) / (xs[i + 1] - xs[i]); yy = ys[i] + w * (ys[i + 1] - ys[i]); h = hs[i] + w * (hs[i + 1] - hs[i])
            rl = int(rid[fr[i]])
            if rl <= 0:
                continue
            left = isr_left.get(rl, True)
            yy2 = yy if left else 8 - yy
            moving_right = xs[i + 1] > xs[i]
            sender = ('ISR' if moving_right else 'BRA') if left else ('BRA' if moving_right else 'ISR')
            nzc.append(dict(y=r2(yy2), h=r2(h), fr=sender, r=rl))
out['netz'] = nzc
os.makedirs('site/court_story', exist_ok=True)
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o
s = json.dumps(_clean(out), separators=(',', ':'), allow_nan=False)
open('site/court_story/data.json', 'w').write(s)
print(len(s) // 1024, 'KB')
print(json.dumps(dict(sideout=so, move=mov, lines=out['lines'], chest=out['chest']), indent=0)[:3000])
a = pd.DataFrame(att); print(a.groupby(['tm', 'b', 's']).size().unstack(fill_value=0)); print(a.groupby('tm').ang.describe())
a['cross'] = a.ang > 20; print(a.groupby(['tm', 'cross', 's']).size().unstack(fill_value=0))
c = pd.DataFrame(cross); print(c.groupby('dir').h.describe())
sp2 = pd.DataFrame(out['spacing']); sp2['bin'] = pd.cut(sp2.d, [0, 2.5, 3, 3.5, 4, 9]); print(sp2.groupby(['tm', 'bin']).c.agg(['mean', 'size']))
print({p: {k: (v['square'], v['side'], v['back']) for k, v in rose[p].items()} for p in PL})
