"""Look for ball touches missing from the touch table, by physics only (independent of the touch detector).

Tests, inside each point (rally start -> verified end frame, v3 frames):
1. impulse: a new gravity flight (ball_height_v2 flight_id) that starts more than GAP frames from any labelled
   player touch and whose upward speed jumps by >= MIN_DVZ m/s against the ball coming in (the end of the previous
   flight), and rises >= MIN_RISE m. A split of one continuous flight has no jump; a sand bounce barely rises (the
   sand absorbs the ball). A ball hidden for a while counts when it was falling and comes back higher and rising.
   Flights that start within CUT frames of a camera cut (ball_boxes_v3video.csv `shot`) are dropped: the tracker
   jumps to the ball's place in the new shot, which looks like a push.
2. hang time: two consecutive in-play touches more than MAX_AIR s apart while the ball never reached the sand.
3. net parity: between consecutive in-play touches by different teams the ball must cross the net (x = 8) an odd
   number of times; same team -> even.
4. rule: more than 3 touches in one possession.

inputs : data/analysis/ball_height_v2.csv, data/contacts/touches_master.csv, data/contacts/ball_contacts_v3.csv,
         data/points/rallies_v3video.csv, data/points/point_end_verified.csv, data/mappings/frame_mapping_v3.csv,
         data/boxes/ball_boxes_v3video.csv (camera cuts)
output : data/experiments/missing_touches/candidates.csv
"""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
FPS, G = 25, 9.81
MIN_RISE, MIN_DVZ, GAP, MAX_AIR, NET_BAND, CUT = 0.40, 3.0, 5, 3.2, 0.4, 3

b = pd.read_csv('data/analysis/ball_height_v2.csv')
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted]
b['f'] = b.frame.map(dict(zip(fm.v2_frame, fm.v3_frame)))
b = b[b.f.notna()].astype({'f': int}).set_index('f').sort_index()
t = pd.read_csv('data/contacts/touches_master.csv')
ev = pd.read_csv('data/contacts/ball_contacts_v3.csv')
ral = pd.read_csv('data/points/rallies_v3video.csv').set_index('rally')
pe = pd.read_csv('data/points/point_end_verified.csv').set_index('point')
shot = pd.read_csv('data/boxes/ball_boxes_v3video.csv', usecols=['frame', 'shot']).set_index('frame').shot
cuts = shot.index[shot.diff().fillna(0) != 0].values
lab = t[(t.kind == 'player')]
real = lab.frame.values


def near_label(f):
    d = np.abs(real - f)
    return int(d.min()) if len(d) else 999


rows = []
for r, R in ral.iterrows():
    end = int(pe.frame.get(r)) if r in pe.index and pd.notna(pe.frame.get(r)) else int(R.end_frame)
    end = min(int(R.end_frame), end + 5)
    w = b.loc[int(R.start_frame):end]
    # 1. impulses at flight starts
    fl = w[w.flight_id >= 0]
    for fid, g in fl.groupby('flight_id'):
        if len(g) < 4:
            continue
        s = int(g.index.min()); h = g.height_m.values
        k = min(4, len(h) - 1)
        vz0 = (h[k] - h[0]) / (k / FPS) + 0.5 * G * (k / FPS)      # upward speed at the start (gravity added back)
        rise = float(np.nanmax(h) - h[0])
        prev = b.loc[s - 40:s - 1]; prev = prev[prev.height_m.notna()]
        before_h = float(prev.height_m.iloc[-1]) if len(prev) else np.nan
        hidden = int(s - prev.index[-1] - 1) if len(prev) else None
        vz_in = np.nan
        if len(prev) >= 3 and hidden is not None and hidden <= 2:
            q = prev.iloc[-5:]; dt = (q.index[-1] - q.index[0]) / FPS
            vz_in = (q.height_m.iloc[-1] - q.height_m.iloc[0]) / dt - 0.5 * G * dt
        if np.isfinite(vz_in):
            jump = vz0 - vz_in >= MIN_DVZ
        else:   # hidden or no ball before: it must come back higher than it went and rising
            jump = np.isfinite(before_h) and h[0] > before_h + MIN_RISE and vz0 > 0
        at_cut = bool(len(cuts)) and int(np.abs(cuts - s).min()) <= CUT
        if jump and rise >= MIN_RISE and near_label(s) > GAP and not at_cut:
            e = ev[(ev.frame - s).abs() <= GAP]
            rows.append(dict(rally=r, frame=s, test='impulse', rise_m=round(rise, 2), vz0=round(vz0, 1), vz_in=round(vz_in, 1) if np.isfinite(vz_in) else None,
                             h_start=round(float(h[0]), 2), h_before=round(before_h, 2) if before_h == before_h else None,
                             hidden_frames=hidden, nearest_label=near_label(s),
                             detector=('; '.join(f"{x.frame}:{x.event_type}{'/'+x.player if isinstance(x.player, str) else ''}" for x in e.itertuples())) or None,
                             x=round(float(g.court_x_m.iloc[0]), 1)))
    # 2-4 between consecutive in-play touches
    tt = t[(t.rally == r) & (t.kind == 'player') & (t.in_play == True)].sort_values('frame')  # noqa: E712
    for a, c in zip(tt.itertuples(), tt.iloc[1:].itertuples()):
        seg = b.loc[a.frame:c.frame]; x = seg.court_x_m.dropna()
        side = pd.Series(np.where(x > 8 + NET_BAND, 1, np.where(x < 8 - NET_BAND, -1, 0)), index=x.index)
        side = side[side != 0]                       # hysteresis: only clear sides count
        cross = int((side.diff().abs() > 0).sum())
        dt = (c.frame - a.frame) / FPS
        low = bool((seg.height_m < 0.15).any())
        diff_team = a.team != c.team
        if dt > MAX_AIR and not low:
            rows.append(dict(rally=r, frame=a.frame, test='hang_time', note=f'{a.player}->{c.player} {dt:.1f}s in the air'))
        if len(x) > 0.5 * len(seg) and (cross % 2 == 1) != diff_team:
            rows.append(dict(rally=r, frame=a.frame, test='net_parity',
                             note=f'{a.player}({a.touch})->{c.player}({c.touch}): {cross} net crossings'))
    pl = tt.possession_len.dropna()
    if (pl > 3).any():
        rows.append(dict(rally=r, frame=int(tt[tt.possession_len > 3].frame.iloc[0]), test='over_3', note='more than 3 touches'))
out = pd.DataFrame(rows).sort_values(['rally', 'frame'])
os.makedirs('data/experiments/missing_touches', exist_ok=True)
out.to_csv('data/experiments/missing_touches/candidates.csv', index=False)
print(out.test.value_counts().to_string())

# Result (2026-11-24): the impulse test finds both touches the owner already knew were missing (point 27 at 5127,
# point 56 at 12737). Of the 20 other suspects, 3 were camera cuts (now filtered) and the owner checked the other 17
# (data/experiments/missing_touches/review_items.json): none is a touch. Net-parity flags all sit on a touch at the
# net (x within 0.4 m); the one hang-time flag is rally 34's replayed point.
