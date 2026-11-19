import pandas as pd, numpy as np, json, sys
R = '/home/user/fotevolley/'
FPS = 25
P = pd.read_csv(R + 'correct_bbox/player_boxes_v9.csv')
P = P[P.player != 'player']
POSE = pd.read_csv(R + 'pose/vitpose_body25.csv')
B = pd.read_csv(R + 'correct_bbox/ball_boxes_corrected3.csv')
PT = pd.read_csv(R + 'point_events_FINAL_VERIFIED.csv')
H = np.load(R + 'homography_px_to_m.npy'); Hi = np.linalg.inv(H)
NET_PX = 908.0
PLAYERS = ['ISR_A', 'ISR_B', 'BRA_A', 'BRA_B']

def m2p(X, Y):
    v = Hi @ np.array([X, Y, 1.0]); return v[:2] / v[2]

def ppm(X, Y):
    return np.linalg.norm(m2p(X + .5, Y) - m2p(X - .5, Y))

# ---------- merge ----------
df = P.merge(POSE.drop(columns=['team']), on=['frame', 'player'], how='left')
df = df.sort_values(['player', 'frame']).reset_index(drop=True)
df['h'] = df.y2 - df.y1

# smoothed court coords + speed
out = []
for pl, g in df.groupby('player'):
    g = g.set_index('frame').reindex(range(g.frame.min(), g.frame.max() + 1))
    g['player'] = pl
    for c in ['court_x', 'court_y']:
        s = g[c].interpolate(limit=5)
        s = s.rolling(7, center=True, min_periods=3).median().rolling(5, center=True, min_periods=2).mean()
        g[c + '_s'] = s
    vx = g.court_x_s.diff() * FPS; vy = g.court_y_s.diff() * FPS
    g['speed'] = np.hypot(vx, vy)
    g.loc[g.speed > 8.5, 'speed'] = np.nan  # tracking jumps
    g['vx'] = vx; g['vy'] = vy
    out.append(g.reset_index().rename(columns={'index': 'frame'}))
df = pd.concat(out, ignore_index=True)
df = df[df.team.notna() | df.court_x_s.notna()]
df['team'] = df.player.str[:3]
df['dnet'] = (df.court_x_s - 8).abs()
df['side'] = np.where(df.court_x_s < 8, 'L', 'R')

# ---------- ball ----------
B = B.set_index('frame')
B['ok'] = B.source.isin(['detected', 'rescued', 'interpolated', 'manual'])
B.loc[~B.ok, ['center_x', 'center_y']] = np.nan
bx = B.center_x; by = B.center_y
B['ball_side'] = np.where(bx < NET_PX - 30, 'L', np.where(bx > NET_PX + 30, 'R', 'N'))

# ---------- rallies ----------
PT = PT.sort_values(['frame', 'point_number'])
pts = PT.drop_duplicates('point_number')
rallies = []
prev = 0
for _, r in pts.iterrows():
    rallies.append(dict(n=int(r.point_number), start=prev + 1, end=int(r.frame), winner=r.country,
                        isr_side=r.blue_side[0], batch=r.event_type == 'BATCH_POINT',
                        set=1 if r.point_number <= 31 else 2, isr=int(r.ISR_score_after), bra=int(r.BRA_score_after),
                        etype=r.event_type))
    prev = int(r.frame)

def isr_side_at(f):
    for rr in rallies:
        if rr['start'] <= f <= rr['end']: return rr['isr_side']
    return rallies[-1]['isr_side']

# per-frame isr side from team sides file (more exact)
TS = pd.read_csv(R + 'team_sides_full_video.csv').set_index('frame')
isr_side_f = TS.blue_side.str[0]

# ---------- contacts ----------
kp = lambda n: [n + '_x', n + '_y', n + '_s']
byf = {f: g for f, g in df[df.x1.notna()].groupby('frame')}
cands = []
for f in B.index[B.ok]:
    if f not in byf: continue
    cx, cy = bx[f], by[f]
    g = byf[f]
    best = None
    for _, r in g.iterrows():
        hh = max(r.h, 60)
        ds = []
        for w in ['r_wrist', 'l_wrist']:
            if r.get(w + '_s', 0) > 0.3:
                ds.append(np.hypot(cx - r[w + '_x'], cy - r[w + '_y']))
        if not ds: continue
        d = min(ds) / hh
        if d < 0.22 and (best is None or d < best[1]):
            best = (r.player, d)
    if best: cands.append((f, best[0], best[1]))
C = pd.DataFrame(cands, columns=['frame', 'player', 'd'])

# ball direction change score
def dir_change(f, k=4):
    a = np.array([bx.get(f - k, np.nan), by.get(f - k, np.nan)]); b0 = np.array([bx.get(f, np.nan), by.get(f, np.nan)])
    c = np.array([bx.get(f + k, np.nan), by.get(f + k, np.nan)])
    v1 = b0 - a; v2 = c - b0
    if np.isnan(v1).any() or np.isnan(v2).any(): return np.nan
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 3 or n2 < 3: return 180.0 if max(n1, n2) > 10 else np.nan
    return np.degrees(np.arccos(np.clip(v1 @ v2 / n1 / n2, -1, 1)))

events = []
if len(C):
    C['grp'] = ((C.frame.diff() > 3) | (C.player != C.player.shift())).cumsum()
    for _, g in C.groupby('grp'):
        r = g.loc[g.d.idxmin()]
        f = int(r.frame)
        best_ang = np.nanmax([dir_change(ff) for ff in range(f - 3, f + 4)] + [-1])
        events.append(dict(frame=f, player=r.player, d=r.d, ang=best_ang))
E = pd.DataFrame(events)
E = E[E.ang >= 25].copy()
# merge same player within 12 frames
E = E.sort_values('frame')
keep = []; last = {}
for _, r in E.iterrows():
    if keep and keep[-1]['player'] == r.player and r.frame - keep[-1]['frame'] < 12: continue
    keep.append(r.to_dict())
E = pd.DataFrame(keep)
E['team'] = E.player.str[:3]

# assign rallies and touch numbers
def rally_of(f):
    for rr in rallies:
        if rr['start'] <= f <= rr['end']: return rr['n']
    return None
E['rally'] = E.frame.map(rally_of)
# serve detection: player beyond baseline before first contact
rows = []
for rr in rallies:
    e = E[E.rally == rr['n']].sort_values('frame')
    seg = df[(df.frame >= rr['start']) & (df.frame <= rr['end'])]
    # server: player with most extreme |x-8| > 8.0 in first part
    behind = seg[seg.dnet > 8.0]
    server = None; serve_f = None
    if len(behind):
        cnt = behind.groupby('player').size()
        server = cnt.idxmax()
        bs = behind[behind.player == server]
        serve_f = int(bs.frame.max())
    rr['server'] = server; rr['serve_f'] = serve_f
    tn = 0; lt = None
    first = True
    for _, c in e.iterrows():
        if serve_f is not None and c.frame < serve_f - 50: continue
        if first and server is not None and c.player == server:
            kind = 'serve'; tn = 0; lt = c.team
        else:
            if c.team != lt: tn = 1
            else: tn += 1
            lt = c.team
            kind = {1: 'pass', 2: 'set', 3: 'attack'}.get(tn, 'extra')
        first = False
        rows.append(dict(c, kind=kind, tn=tn))
    if len(e):
        rr['last_contact'] = rows[-1]['player'] if rows and rows[-1]['rally'] == rr['n'] else None
        rr['last_contact_f'] = rows[-1]['frame'] if rows and rows[-1]['rally'] == rr['n'] else None
    else:
        rr['last_contact'] = None; rr['last_contact_f'] = None
    rr['play_start'] = serve_f if serve_f else (int(e.frame.min()) if len(e) else None)
    rr['play_end'] = min(rr['end'], int(rr['last_contact_f']) + 50) if rr['last_contact_f'] else None
EV = pd.DataFrame(rows)
print('contacts', len(EV), EV.kind.value_counts().to_dict(), file=sys.stderr)
print(EV.groupby(['player', 'kind']).size().unstack(fill_value=0), file=sys.stderr)

# fix 'extra' (4th touches) -> treat as attack if 3 already? leave
# in-play mask
inplay = np.zeros(int(df.frame.max()) + 2, bool)
for rr in rallies:
    if rr['play_start'] and rr['play_end'] and rr['play_end'] > rr['play_start']:
        inplay[rr['play_start']:rr['play_end'] + 1] = True
df['inplay'] = inplay[df.frame.astype(int)]
dur = [(rr['play_end'] - rr['play_start']) / FPS for rr in rallies if rr['play_start'] and rr['play_end'] and rr['play_end'] > rr['play_start']]
print('rally durations', np.round(np.percentile(dur, [10, 50, 90]), 1), len(dur), file=sys.stderr)
for rr in rallies[:12]: print(rr, file=sys.stderr)
pd.to_pickle(dict(df=df, EV=EV, rallies=rallies), '/tmp/claude-0/-home-user-fotevolley/d4e6fc41-e0bc-5bac-ac65-a3014176e779/scratchpad/stage1.pkl')
