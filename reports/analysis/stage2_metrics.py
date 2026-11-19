import pandas as pd, numpy as np, json, sys
SP = '/tmp/claude-0/-home-user-fotevolley/d4e6fc41-e0bc-5bac-ac65-a3014176e779/scratchpad/'
R = '/home/user/fotevolley/'
FPS = 25
s = pd.read_pickle(SP + 'stage1.pkl'); df = s['df']; rallies = s['rallies']
C = pd.read_pickle(SP + 'turns.pkl')
B = pd.read_csv(R + 'correct_bbox/ball_boxes_corrected3.csv').set_index('frame')
okb = B.source.isin(['detected', 'rescued', 'interpolated', 'manual'])
bx = B.center_x.where(okb); by = B.center_y.where(okb)
NET_PX = 908.0
H = np.load(R + 'homography_px_to_m.npy'); Hi = np.linalg.inv(H)
def m2p(X, Y):
    v = Hi @ np.array([X, Y, 1.0]); return v[:2] / v[2]
def ppm(X, Y):
    return np.linalg.norm(m2p(X + .5, Y) - m2p(X - .5, Y))
PL = ['ISR_A', 'ISR_B', 'BRA_A', 'BRA_B']
partner = {'ISR_A': 'ISR_B', 'ISR_B': 'ISR_A', 'BRA_A': 'BRA_B', 'BRA_B': 'BRA_A'}
TS = pd.read_csv(R + 'team_sides_full_video.csv').set_index('frame')
isr_side = TS.blue_side.str[0]

D = df.set_index(['frame', 'player']).sort_index()

def ang3(a, b, c):
    v1 = a - b; v2 = c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0 or np.isnan(n): return np.nan
    return np.degrees(np.arccos(np.clip(v1 @ v2 / n, -1, 1)))

def P(r, k):
    if r.get(k + '_s', 0) is None or not (r.get(k + '_s', 0) > 0.3): return np.array([np.nan, np.nan])
    return np.array([r[k + '_x'], r[k + '_y']])

# ---------------- contacts & sequence ----------------
C = C[(C.d < 0.05) & (~C.low)].copy()
C['team'] = C.player.str[:3]
def rally_of(f):
    for rr in rallies:
        if rr['start'] <= f <= rr['end']: return rr['n']
    return None
C['rally'] = C.frame.map(rally_of)
C = C.dropna(subset=['rally']).sort_values('frame')
# drop same-player duplicates within 10 frames
keep = []
for _, r in C.iterrows():
    if keep and keep[-1]['player'] == r.player and r.frame - keep[-1]['frame'] < 10: continue
    keep.append(r.to_dict())
C = pd.DataFrame(keep)

rows = []
for rr in rallies:
    e = C[C.rally == rr['n']].sort_values('frame')
    seg = df[(df.frame >= rr['start']) & (df.frame <= rr['end'])]
    behind = seg[seg.dnet > 7.9]
    server = None; serve_f = None
    if len(behind):
        cnt = behind.groupby('player').size()
        if cnt.max() >= 5:
            server = cnt.idxmax(); serve_f = int(behind[behind.player == server].frame.max())
    rr['server'] = server; rr['serve_f'] = serve_f
    lt = None; tn = 0; first = True
    rl = []
    el = e.to_dict('records')
    for i, c in enumerate(el):
        f = int(c['frame']); own = (isr_side[f] if f in isr_side.index else 'L') if c['team'] == 'ISR' else ('R' if (isr_side[f] if f in isr_side.index else 'L') == 'L' else 'L')
        opp = 'R' if own == 'L' else 'L'
        pf = int(el[i - 1]['frame']) if i > 0 else f - 40
        nf = int(el[i + 1]['frame']) if i + 1 < len(el) else f + 40
        def side_of(xx):
            return 'L' if xx < NET_PX - 40 else ('R' if xx > NET_PX + 40 else 'N')
        before = [side_of(bx[ff]) for ff in range(max(pf, f - 40), f - 2) if ff in bx.index and bx[ff] == bx[ff]]
        after = [side_of(bx[ff]) for ff in range(f + 2, min(nf, f + 45)) if ff in bx.index and bx[ff] == bx[ff]]
        came = (opp in before) or (i > 0 and el[i - 1]['team'] != c['team'])
        sent = (opp in after) or (i + 1 < len(el) and el[i + 1]['team'] != c['team'])
        if first and server is not None and c['player'] == server and (serve_f is None or f <= serve_f + 15):
            kind = 'serve'
        elif came and sent: kind = 'one_touch'
        elif came: kind = 'pass'
        elif sent: kind = 'attack'
        else: kind = 'set'
        if i == 0 and kind in ('attack', 'one_touch') and server is None and c['team'] != (lt or c['team']): pass
        first = False
        d = dict(c); d.update(kind=kind, tn=0, came=came, sent=sent); rl.append(d)
    rows += rl
    rr['n_contacts'] = len(rl)
    rr['last_contact'] = rl[-1]['player'] if rl else None
    rr['last_kind'] = rl[-1]['kind'] if rl else None
    starts = [x for x in [serve_f, (rl[0]['frame'] if rl else None)] if x is not None]
    rr['play_start'] = min(starts) if starts else None
    rr['play_end'] = min(rr['end'], rl[-1]['frame'] + 40) if rl else None
EV = pd.DataFrame(rows)
_dn = df.set_index(['frame', 'player']).dnet
EV['dnet_c'] = [(_dn.get((int(f), p), np.nan)) for f, p in zip(EV.frame, EV.player)]
EV.loc[(EV.kind == 'attack') & (EV.dnet_c > 6.5), 'kind'] = 'serve'
EV.loc[(EV.kind == 'one_touch') & (EV.dnet_c <= 1.6), 'kind'] = 'block'
print(EV.groupby(['player', 'kind']).size().unstack(fill_value=0), file=sys.stderr)

inplay = np.zeros(int(df.frame.max()) + 2, bool)
for rr in rallies:
    if rr['play_start'] is not None and rr['play_end'] is not None and rr['play_end'] > rr['play_start']:
        inplay[rr['play_start']:rr['play_end'] + 1] = True
df['inplay'] = inplay[df.frame.astype(int)]
print('inplay sec', inplay.sum() / FPS, file=sys.stderr)

OUT = {}
# ---------------- score / points ----------------
pts = []
for rr in rallies:
    pts.append(dict(n=rr['n'], set=rr['set'], winner=rr['winner'], isr=rr['isr'], bra=rr['bra'], t=round(rr['end'] / FPS, 1),
                    isr_side=rr['isr_side'], server=rr['server'], batch=rr['batch'], etype=rr['etype'],
                    dur=round((rr['play_end'] - rr['play_start']) / FPS, 1) if rr['play_start'] is not None and rr['play_end'] and rr['play_end'] > rr['play_start'] else None,
                    nc=rr['n_contacts'], last=rr['last_contact'], last_kind=rr['last_kind']))
OUT['points'] = pts
P_ = pd.DataFrame(pts)
OUT['score_summary'] = dict(
    isr_total=int((P_.winner == 'ISR').sum()), bra_total=int((P_.winner == 'BRA').sum()),
    set1=dict(isr=int(((P_.set == 1) & (P_.winner == 'ISR')).sum()), bra=int(((P_.set == 1) & (P_.winner == 'BRA')).sum())),
    set2=dict(isr=int(((P_.set == 2) & (P_.winner == 'ISR')).sum()), bra=int(((P_.set == 2) & (P_.winner == 'BRA')).sum())),
    by_side={sd: dict(isr=int(((P_.isr_side == sd) & (P_.winner == 'ISR')).sum()), bra=int(((P_.isr_side == sd) & (P_.winner == 'BRA')).sum())) for sd in 'LR'},
)
# runs
runs = []; cur = None; ln = 0; st = None
for p in pts:
    if p['winner'] == cur: ln += 1
    else:
        if cur: runs.append(dict(team=cur, len=ln, from_n=st))
        cur = p['winner']; ln = 1; st = p['n']
runs.append(dict(team=cur, len=ln, from_n=st))
OUT['runs'] = sorted(runs, key=lambda r: -r['len'])[:6]
# sideout vs breakpoint: who served
so = {'ISR': [0, 0], 'BRA': [0, 0]}  # [won when receiving, total receiving]
srv = {}
for p in pts:
    if not p['server']: continue
    st = p['server'][:3]; rec = 'BRA' if st == 'ISR' else 'ISR'
    so[rec][1] += 1; so[rec][0] += p['winner'] == rec
    srv.setdefault(p['server'], [0, 0]); srv[p['server']][1] += 1; srv[p['server']][0] += p['winner'] == st
OUT['sideout'] = so; OUT['serve_by_player'] = srv
# point by rally length
dd = P_.dropna(subset=['dur'])
OUT['by_len'] = {lab: dict(isr=int(((dd.winner == 'ISR') & m).sum()), bra=int(((dd.winner == 'BRA') & m).sum()))
                 for lab, m in [('short', dd.dur < 4), ('mid', (dd.dur >= 4) & (dd.dur < 8)), ('long', dd.dur >= 8)]}
# last contact
lc = P_.dropna(subset=['last'])
OUT['last_contact'] = {pl: dict(won=int(((lc['last'] == pl) & (lc.winner == pl[:3])).sum()), lost=int(((lc['last'] == pl) & (lc.winner != pl[:3])).sum())) for pl in PL}

# ---------------- movement ----------------
mv = {}
ip = df[df.inplay]
for pl in PL:
    g = ip[ip.player == pl]
    sp = g.speed.dropna()
    secs = len(sp) / FPS
    dist = sp.sum() / FPS
    # sprints: runs over 3.5 m/s for >=0.4s
    fast = (g.speed > 3.5).astype(int).values
    nspr = 0; run = 0
    for v in fast:
        if v: run += 1
        else:
            if run >= 10: nspr += 1
            run = 0
    mv[pl] = dict(avg_speed=round(sp.mean(), 2), p95=round(sp.quantile(.95), 2), max=round(sp.quantile(.995), 2),
                  m_per_min=round(dist / secs * 60, 1), dist_total=round(dist, 0), sprints=nspr, sprints_per_min=round(nspr / secs * 60, 2),
                  still_pct=round((sp < 0.5).mean() * 100, 1), secs=round(secs, 0),
                  dnet_mean=round(g.dnet.mean(), 2), dnet_med=round(g.dnet.median(), 2),
                  near_net_pct=round((g.dnet < 2.5).mean() * 100, 1), back_pct=round((g.dnet > 5.5).mean() * 100, 1),
                  y_mean=round(g.court_y_s.mean(), 2))
OUT['move'] = mv
# speed histogram
OUT['speed_hist'] = {pl: np.histogram(ip[ip.player == pl].speed.dropna(), bins=[0, .5, 1, 2, 3, 4, 5, 9])[0].tolist() for pl in PL}

# heatmaps: own-half normalized. depth = dnet (0..8), lateral = y (0..8). For left side y as is; right side mirror y so that the
# view is "from behind own baseline": take everything as seen with net on top.
heat = {}
for pl in PL:
    g = ip[ip.player == pl].dropna(subset=['court_x_s', 'court_y_s'])
    dep = g.dnet.clip(0, 9.99); lat = g.court_y_s.clip(-0.99, 8.99)
    # mirror lateral for right side so that "left of player facing net" consistent
    lat = np.where(g.court_x_s < 8, 8 - lat, lat)
    h, _, _ = np.histogram2d(dep, lat, bins=[np.arange(0, 10.01, 1), np.arange(-1, 9.01, 1)])
    heat[pl] = (h / h.sum() * 100).round(1).tolist()
OUT['heat'] = heat
# lateral split: who plays the "left" (from own perspective)
# partner distance
pdist = {}
for team in ['ISR', 'BRA']:
    a = ip[ip.player == team + '_A'].set_index('frame')[['court_x_s', 'court_y_s']]
    b = ip[ip.player == team + '_B'].set_index('frame')[['court_x_s', 'court_y_s']]
    j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner').dropna()
    dd_ = np.hypot(j.court_x_s_a - j.court_x_s_b, j.court_y_s_a - j.court_y_s_b)
    dx = (j.court_x_s_a - j.court_x_s_b).abs(); dy = (j.court_y_s_a - j.court_y_s_b).abs()
    # side-by-side vs front-back
    sbs = ((dy > dx)).mean()
    pdist[team] = dict(mean=round(dd_.mean(), 2), med=round(dd_.median(), 2), p10=round(dd_.quantile(.1), 2), p90=round(dd_.quantile(.9), 2),
                       side_by_side_pct=round(sbs * 100, 1), too_close_pct=round((dd_ < 1.5).mean() * 100, 1), far_pct=round((dd_ > 6).mean() * 100, 1),
                       hist=np.histogram(dd_, bins=[0, 1.5, 3, 4.5, 6, 20])[0].tolist(), frames=len(dd_))
    pdist[team]['series'] = j.index.tolist()[:0]
OUT['pdist'] = pdist

# ---------------- head / gaze ----------------
def head_dir(r):
    """+1 facing image right, -1 facing image left, 0 camera/unknown, 2 away from camera"""
    ns = r.get('nose_s', 0); res = r.get('r_ear_s', 0); les = r.get('l_ear_s', 0)
    if not (ns == ns): return np.nan
    ears = [r[e + '_x'] for e in ['r_ear', 'l_ear'] if r.get(e + '_s', 0) > 0.3]
    sw = abs(r.get('r_shoulder_x', np.nan) - r.get('l_shoulder_x', np.nan))
    hsz = max(r['h'] * 0.07, 4)
    if ns < 0.3 and ears: return 2
    if not ears or ns < 0.3: return np.nan
    dx = r['nose_x'] - np.mean(ears)
    if abs(dx) < 0.35 * hsz: return 0
    return 1 if dx > 0 else -1

pose = df[df.nose_s.notna()].copy()
pose['hd'] = [head_dir(r) for r in pose.to_dict('records')]
pose['ball_x'] = pose.frame.map(bx); pose['ball_y'] = pose.frame.map(by)
pose['bside'] = np.where(pose.ball_x < NET_PX - 40, 'L', np.where(pose.ball_x > NET_PX + 40, 'R', 'N'))
pose['own'] = np.where(pose.court_x_s < 8, 'L', 'R')
pose['net_dir'] = np.where(pose.own == 'L', 1, -1)
pose['ball_dir'] = np.sign(pose.ball_x - pose.nose_x)
pose = pose[pose.inplay]
gaze = {}
for pl in PL:
    g = pose[pose.player == pl].dropna(subset=['hd'])
    res = {}
    for ctx, m in [('ball_opp', g.bside != g.own), ('ball_own', g.bside == g.own)]:
        m = m & (g.bside != 'N') & g.ball_x.notna()
        gg = g[m]
        if len(gg) < 20: continue
        lat = gg.hd.isin([1, -1])
        res[ctx] = dict(n=len(gg),
                        to_net=round(((gg.hd == gg.net_dir)).mean() * 100, 1),
                        to_back=round(((gg.hd == -gg.net_dir)).mean() * 100, 1),
                        camera=round((gg.hd == 0).mean() * 100, 1),
                        away=round((gg.hd == 2).mean() * 100, 1),
                        to_ball=round(((gg.hd == gg.ball_dir) & lat).mean() * 100, 1))
        # scanning: head direction switches per second within continuous stretches
        sw = 0; secs = 0
        for _, run in gg.groupby((gg.frame.diff() != 1).cumsum()):
            if len(run) < 12: continue
            hd = run.hd.rolling(5, center=True, min_periods=3).median().round()
            ch = (hd.diff().fillna(0) != 0).sum()
            sw += ch; secs += len(run) / FPS
        res[ctx]['switch_per_sec'] = round(sw / secs, 2) if secs else None
        res[ctx]['secs'] = round(secs, 1)
    gaze[pl] = res
OUT['gaze'] = gaze

# ---------------- body / anthropometry (estimates) ----------------
body = {}
for pl in PL:
    g = df[(df.player == pl) & df.nose_s.notna()].copy()
    g = g[(g.nose_s > .5) & (g.r_ankle_s > .5) & (g.l_ankle_s > .5) & (g.r_knee_s > .5) & (g.l_knee_s > .5) & (g.speed < 0.6)]
    recs = []
    for r in g.to_dict('records'):
        hip = np.array([r['mid_hip_x'], r['mid_hip_y']])
        kr = ang3(np.array([r['r_hip_x'], r['r_hip_y']]), np.array([r['r_knee_x'], r['r_knee_y']]), np.array([r['r_ankle_x'], r['r_ankle_y']]))
        kl = ang3(np.array([r['l_hip_x'], r['l_hip_y']]), np.array([r['l_knee_x'], r['l_knee_y']]), np.array([r['l_ankle_x'], r['l_ankle_y']]))
        if min(kr, kl) < 168: continue
        neck = np.array([r['neck_x'], r['neck_y']])
        trunk = abs(np.degrees(np.arctan2(neck[0] - hip[0], hip[1] - neck[1])))
        if trunk > 8: continue
        ank = (r['r_ankle_y'] + r['l_ankle_y']) / 2
        hpx = (ank - r['nose_y'])
        sc = ppm(r['court_x'], r['court_y'])
        recs.append(dict(h_m=hpx / sc, sh=abs(r['r_shoulder_x'] - r['l_shoulder_x']) / sc, leg=(ank - r['mid_hip_y']) / max(hpx, 1),
                         arm=(np.hypot(r['r_shoulder_x'] - r['r_elbow_x'], r['r_shoulder_y'] - r['r_elbow_y']) + np.hypot(r['r_elbow_x'] - r['r_wrist_x'], r['r_elbow_y'] - r['r_wrist_y'])) / max(hpx, 1)))
    R_ = pd.DataFrame(recs)
    body[pl] = dict(n=len(R_), nose_to_ankle_m=round(R_.h_m.median(), 2), leg_ratio=round(R_.leg.median(), 3), arm_ratio=round(R_.arm.median(), 3),
                    shoulder_px_m=round(R_.sh.quantile(.9), 2))
OUT['body'] = body
print(body, file=sys.stderr)

# ---------------- angles at contacts ----------------
def angles_at(f, pl):
    try: r = D.loc[(f, pl)]
    except KeyError: return None
    r = r.to_dict()
    if not (r.get('nose_s', 0) == r.get('nose_s', 0)): return None
    def pt(k):
        return np.array([r[k + '_x'], r[k + '_y']]) if r.get(k + '_s', 0) > 0.3 else np.array([np.nan, np.nan])
    o = {}
    # hitting arm = higher wrist
    rw, lw = pt('r_wrist'), pt('l_wrist')
    side = 'r' if (np.nan_to_num(rw[1], nan=1e9) < np.nan_to_num(lw[1], nan=1e9)) else 'l'
    sh, el, wr = pt(side + '_shoulder'), pt(side + '_elbow'), pt(side + '_wrist')
    hip = pt(side + '_hip')
    o['elbow'] = ang3(sh, el, wr)
    o['shoulder'] = ang3(hip, sh, el)  # arm elevation vs trunk
    o['knee_r'] = ang3(pt('r_hip'), pt('r_knee'), pt('r_ankle')); o['knee_l'] = ang3(pt('l_hip'), pt('l_knee'), pt('l_ankle'))
    o['knee'] = np.nanmin([o['knee_r'], o['knee_l']])
    o['hip'] = np.nanmean([ang3(pt('neck'), pt('r_hip'), pt('r_knee')), ang3(pt('neck'), pt('l_hip'), pt('l_knee'))])
    nk, mh = pt('neck'), pt('mid_hip')
    o['trunk_lean'] = abs(np.degrees(np.arctan2(nk[0] - mh[0], mh[1] - nk[1])))
    sc = ppm(r['court_x'], r['court_y'])
    o['wrist_above_head_m'] = (r['nose_y'] - wr[1]) / sc if wr[1] == wr[1] else np.nan
    o['dnet'] = abs(r['court_x'] - 8) if r.get('court_x') == r.get('court_x') else np.nan
    o['speed'] = r.get('speed', np.nan)
    o['feet_y'] = r['feet_y']; o['sc'] = sc
    return o

def jump_height(f, pl):
    # feet rise vs median feet_y in [-40,-15] frames
    try:
        g = df[(df.player == pl) & (df.frame >= f - 40) & (df.frame <= f + 3)].set_index('frame')
        base = g.loc[f - 40:f - 15].feet_y.median()
        top = g.loc[f - 4:f + 3].feet_y.min()
        sc = ppm(g.loc[f, 'court_x'], g.loc[f, 'court_y'])
        return max(0, (base - top) / sc)
    except Exception: return np.nan

def min_knee_before(f, pl):
    vals = []
    for ff in range(f - 25, f - 2):
        a = angles_at(ff, pl)
        if a: vals.append(a['knee'])
    return np.nanmin(vals) if vals and not all(np.isnan(vals)) else np.nan

ang_rows = []
for _, e in EV.iterrows():
    best = None
    for ff in range(int(e.frame) - 3, int(e.frame) + 4):
        aa = angles_at(ff, e.player)
        if not aa or aa['wrist_above_head_m'] != aa['wrist_above_head_m']: continue
        key = aa['wrist_above_head_m'] if e.kind in ('set', 'attack', 'serve', 'one_touch') else -aa['wrist_above_head_m']
        if best is None or key > best[0]: best = (key, aa)
    a = best[1] if best else angles_at(int(e.frame), e.player)
    if not a: continue
    a.update(player=e.player, kind=e.kind, frame=int(e.frame), rally=e.rally)
    if e.kind in ('attack', 'serve', 'set'):
        a['jump'] = jump_height(int(e.frame), e.player)
        a['load_knee'] = min_knee_before(int(e.frame), e.player)
    # partner snapshot
    pp = partner[e.player]
    try:
        me = D.loc[(int(e.frame), e.player)]; pr = D.loc[(int(e.frame), pp)]
        a['partner_dist'] = float(np.hypot(me.court_x_s - pr.court_x_s, me.court_y_s - pr.court_y_s))
        a['partner_dnet'] = float(abs(pr.court_x_s - 8))
        # partner movement over next 0.8s toward net?
        pr2 = D.loc[(int(e.frame) + 20, pp)]
        a['partner_to_net_m'] = float(abs(pr.court_x_s - 8) - abs(pr2.court_x_s - 8))
        a['partner_speed'] = float(df[(df.player == pp) & (df.frame.between(int(e.frame) - 5, int(e.frame) + 20))].speed.mean())
        hd = head_dir(pr.to_dict()) if pr.nose_s == pr.nose_s else np.nan
        own = 1 if pr.court_x_s < 8 else -1
        a['partner_head'] = {1 * own: 'net', -1 * own: 'back', 0: 'camera', 2: 'away'}.get(hd, None) if hd == hd else None
        # partner looking at ball?
        bxx = bx.get(int(e.frame), np.nan)
        a['partner_look_ball'] = bool(hd in (1, -1) and np.sign(bxx - pr.nose_x) == hd) if hd == hd and bxx == bxx else None
    except KeyError:
        pass
    ang_rows.append(a)
A = pd.DataFrame(ang_rows)

def summ(g, cols):
    return {c: (round(float(g[c].median()), 2) if c in g and g[c].notna().sum() else None) for c in cols}
OUT['angles'] = {}
for pl in PL:
    OUT['angles'][pl] = {}
    for k in ['serve', 'pass', 'set', 'attack', 'one_touch', 'block']:
        g = A[(A.player == pl) & (A.kind == k)]
        d_ = summ(g, ['elbow', 'shoulder', 'knee', 'hip', 'trunk_lean', 'wrist_above_head_m', 'dnet', 'jump', 'load_knee', 'speed'])
        d_['n'] = len(g)
        OUT['angles'][pl][k] = d_
# partner while I set / partner during my pass
OUT['partner_during'] = {}
for pl in PL:
    OUT['partner_during'][pl] = {}
    for k in ['pass', 'set', 'attack', 'serve']:
        g = A[(A.player == pl) & (A.kind == k)]
        if not len(g): continue
        hd = g.partner_head.value_counts(normalize=True).mul(100).round(0).to_dict() if 'partner_head' in g else {}
        OUT['partner_during'][pl][k] = dict(n=len(g), dist=round(g.partner_dist.median(), 2) if g.partner_dist.notna().any() else None,
                                            partner_dnet=round(g.partner_dnet.median(), 2) if g.partner_dnet.notna().any() else None,
                                            partner_to_net=round(g.partner_to_net_m.median(), 2) if g.partner_to_net_m.notna().any() else None,
                                            partner_speed=round(g.partner_speed.median(), 2) if g.partner_speed.notna().any() else None,
                                            approach_pct=round((g.partner_to_net_m > 0.3).mean() * 100, 0) if g.partner_to_net_m.notna().any() else None,
                                            look_ball_pct=round(g.partner_look_ball.dropna().astype(float).mean() * 100, 0) if g.partner_look_ball.notna().any() else None,
                                            head=hd)
OUT['contacts'] = EV.groupby(['player', 'kind']).size().unstack(fill_value=0).to_dict('index')

# defense setup when opponent attacks: positions of my team at opponent's attack frame
dfn = {}
for team in ['ISR', 'BRA']:
    opp = 'BRA' if team == 'ISR' else 'ISR'
    at = EV[(EV.team == opp) & (EV.kind == 'attack')]
    rs = []
    for f in at.frame:
        try:
            a = D.loc[(int(f), team + '_A')]; b = D.loc[(int(f), team + '_B')]
        except KeyError: continue
        da, db = abs(a.court_x_s - 8), abs(b.court_x_s - 8)
        rs.append(dict(a=da, b=db, blocker=team + ('_A' if da < db else '_B'), split=abs(da - db),
                       dist=np.hypot(a.court_x_s - b.court_x_s, a.court_y_s - b.court_y_s), one_at_net=min(da, db) < 1.5))
    rs = pd.DataFrame(rs)
    if len(rs):
        dfn[team] = dict(n=len(rs), one_at_net_pct=round(rs.one_at_net.mean() * 100, 0), dist=round(rs.dist.median(), 2),
                         blocker=rs.blocker.value_counts().to_dict(), front=round(rs[['a', 'b']].min(axis=1).median(), 2), back=round(rs[['a', 'b']].max(axis=1).median(), 2))
OUT['defense'] = dfn
OUT['meta'] = dict(fps=FPS, frames=int(df.frame.max()) + 1, inplay_sec=round(inplay.sum() / FPS, 0), n_contacts=len(EV),
                   n_rallies_with_play=int(sum(1 for p in pts if p['dur'])))
json.dump(OUT, open(SP + 'report.json', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (o.item() if hasattr(o, 'item') else str(o)))
A.to_csv(SP + 'angles.csv', index=False); EV.to_csv(SP + 'contacts.csv', index=False)
print(json.dumps({k: OUT[k] for k in ['score_summary', 'sideout', 'serve_by_player', 'by_len', 'last_contact', 'move', 'pdist', 'gaze', 'defense', 'meta']}, indent=0, default=str), file=sys.stderr)

# ---- extra: phase split (ball own side vs opp side) ----
ip2 = df[df.inplay].copy()
ip2['ball_x'] = ip2.frame.map(bx)
ip2['bside'] = np.where(ip2.ball_x < NET_PX - 40, 'L', np.where(ip2.ball_x > NET_PX + 40, 'R', 'N'))
ip2['own'] = np.where(ip2.court_x_s < 8, 'L', 'R')
ph = {}
for pl in PL:
    g = ip2[ip2.player == pl]
    o = g[(g.bside == g.own)]; x_ = g[(g.bside != g.own) & (g.bside != 'N')]
    ph[pl] = dict(dnet_own=round(o.dnet.median(), 2), dnet_opp=round(x_.dnet.median(), 2), speed_own=round(o.speed.mean(), 2), speed_opp=round(x_.speed.mean(), 2))
OUT['phase'] = ph
pp = {}
for team in ['ISR', 'BRA']:
    a = ip2[ip2.player == team + '_A'].set_index('frame'); b = ip2[ip2.player == team + '_B'].set_index('frame')
    j = a[['court_x_s', 'court_y_s', 'bside', 'own']].join(b[['court_x_s', 'court_y_s']], lsuffix='_a', rsuffix='_b', how='inner').dropna()
    j['d'] = np.hypot(j.court_x_s_a - j.court_x_s_b, j.court_y_s_a - j.court_y_s_b)
    pp[team] = dict(own=round(j[j.bside == j.own].d.median(), 2), opp=round(j[(j.bside != j.own) & (j.bside != 'N')].d.median(), 2))
    # per-rally partner distance mean -> who won
    pp[team]['all_mean'] = round(j.d.mean(), 2)
OUT['pdist_phase'] = pp
# partner distance per point won/lost
pw = {}
for team in ['ISR', 'BRA']:
    a = ip2[ip2.player == team + '_A'].set_index('frame'); b = ip2[ip2.player == team + '_B'].set_index('frame')
    j = a[['court_x_s', 'court_y_s']].join(b[['court_x_s', 'court_y_s']], lsuffix='_a', rsuffix='_b', how='inner').dropna()
    j['d'] = np.hypot(j.court_x_s_a - j.court_x_s_b, j.court_y_s_a - j.court_y_s_b)
    won = []; lost = []
    for rr in rallies:
        if rr['play_start'] is None or rr['play_end'] is None: continue
        m = j.loc[rr['play_start']:rr['play_end']].d.mean()
        (won if rr['winner'] == team else lost).append(m)
    pw[team] = dict(won=round(np.nanmean(won), 2), lost=round(np.nanmean(lost), 2))
OUT['pdist_wl'] = pw
# speed won/lost per player
sw = {}
for pl in PL:
    g = ip2[ip2.player == pl].set_index('frame')
    won = []; lost = []
    for rr in rallies:
        if rr['play_start'] is None or rr['play_end'] is None: continue
        m = g.loc[rr['play_start']:rr['play_end']].speed.mean()
        (won if rr['winner'] == pl[:3] else lost).append(m)
    sw[pl] = dict(won=round(np.nanmean(won), 2), lost=round(np.nanmean(lost), 2))
OUT['speed_wl'] = sw
json.dump(OUT, open(SP + 'report.json', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps({k: OUT[k] for k in ['phase', 'pdist_phase', 'pdist_wl', 'speed_wl']}, default=str))
