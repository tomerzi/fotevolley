"""Reaction to a change of pace: when one player speeds up or slows down, how long until each other player changes
his own pace, and how strongly. Split by who reacts: the teammate or an opponent.

Speed: from the ground positions (data/analysis/player_positions_v3.csv, 25 fps, m), lightly smoothed (5-frame
centred mean), tracking jumps above 9 m/s dropped. Acceleration a = d(speed)/dt.
Pace change: a burst where |a| >= 3 m/s^2 on 3 frames in a row with the same sign and the speed changes by >= 0.8 m/s
over the burst; it starts at the last frame before with |a| < 1 m/s^2 (or a sign change). Only frames inside the 60
rallies (data/points/rallies_v2.csv) count; bursts closer than 0.3 s to the previous one of the same player are merged.
Reaction: for each pace change of player A, the first pace change of each other player B that starts 0.04-1.0 s
later (later than 1.0 s = no reaction). Because players also change pace for their own reasons (the ball), a part of
these pairs is chance; the chance level is estimated by shifting B's changes by a random offset (2-6 s) 200 times.
Also per pair (A -> B): the cross-correlation of |a| at lags 0..1 s; the lag of its peak is the typical delay.
Writes data/analysis/pace_changes_<s>_<e>.csv, pace_reaction_<s>_<e>.csv, a chart (--chart) and a clip (--video).
usage: python scripts/analysis/pace_reaction.py [START END] [--chart OUT.png] [--video OUT.mp4]
"""
import os, sys, subprocess
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
args = [a for a in sys.argv[1:] if not a.startswith('--')]
F0, F1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, 749)
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
CHART, VIDEO = opt('--chart'), opt('--video')
FPS = 25
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Ron_Ben_Ishay': 'ISR', 'Maor_Haas': 'ISR'}
ORDER = ['Ron_Ben_Ishay', 'Maor_Haas', 'Franklin', 'Hiltinho']
NAMES = {'Franklin': 'Franklin', 'Hiltinho': 'Hiltinho', 'Ron_Ben_Ishay': 'Ron Ben Ishay', 'Maor_Haas': 'Maor Haas'}
A_HI, A_LO, DV, MAXLAG = 3.0, 1.0, 0.8, 1.0

ral = pd.read_csv('data/points/rallies_v2.csv')
inplay = np.zeros(F1 + 1, bool)
for r in ral.itertuples():
    inplay[max(r.start_frame, F0):min(r.end_frame, F1) + 1] = True
p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player'])
SP, ACC = {}, {}
for pl, g in p.groupby('player'):
    xy = g.set_index('v2_frame').reindex(range(F0, F1 + 1))[['x', 'y']].to_numpy()
    v = np.gradient(xy, axis=0) * FPS
    s = np.linalg.norm(v, axis=1); s[s > 9] = np.nan
    s = pd.Series(s).rolling(5, center=True, min_periods=3).mean().to_numpy()
    SP[pl] = s; ACC[pl] = np.gradient(s) * FPS

ev = []
for pl in TEAM:
    s, a = SP[pl], ACC[pl]
    n = len(a); i = 0; last = -99
    while i < n - 3:
        sg = np.sign(a[i])
        if inplay[F0 + i] and np.all(np.abs(a[i:i + 3]) >= A_HI) and np.all(np.sign(a[i:i + 3]) == sg):
            k = i
            while k > 0 and abs(a[k - 1]) >= A_LO and np.sign(a[k - 1]) == sg:
                k -= 1
            e = i
            while e < n - 1 and abs(a[e + 1]) >= A_LO and np.sign(a[e + 1]) == sg:
                e += 1
            dv = s[e] - s[k]
            if abs(dv) >= DV and k - last >= 0.3 * FPS:
                ev.append(dict(player=pl, team=TEAM[pl], frame=F0 + k, time_s=round((F0 + k) / FPS, 2),
                               kind='speeds up' if dv > 0 else 'slows down', speed_from=round(s[k], 2), speed_to=round(s[e], 2),
                               dur_s=round((e - k) / FPS, 2), peak_acc=round(float(np.nanmax(np.abs(a[k:e + 1]))), 1)))
                last = k
            i = e + 1
        else:
            i += 1
ev = pd.DataFrame(ev).sort_values('frame').reset_index(drop=True)
ev.to_csv(f'data/analysis/pace_changes_{F0}_{F1}.csv', index=False)
print(ev.groupby(['player', 'kind']).size().unstack(fill_value=0).to_string(), '\n')

byp = {pl: ev[ev.player == pl].frame.to_numpy() for pl in TEAM}


def first_after(fr_b, t):
    c = fr_b[(fr_b >= t + 1) & (fr_b <= t + MAXLAG * FPS)]
    return c[0] if len(c) else None


rows = []
for e in ev.itertuples():
    for pl in TEAM:
        if pl == e.player:
            continue
        b = first_after(byp[pl], e.frame)
        rel = 'teammate' if TEAM[pl] == e.team else 'opponent'
        r = dict(frame=e.frame, time_s=e.time_s, leader=e.player, leader_kind=e.kind, player=pl, relation=rel)
        if b is not None:
            eb = ev[(ev.player == pl) & (ev.frame == b)].iloc[0]
            r.update(react_frame=b, reaction_s=round((b - e.frame) / FPS, 2), react_kind=eb.kind,
                     same_direction=eb.kind == e.kind, react_dv=round(eb.speed_to - eb.speed_from, 2), react_acc=eb.peak_acc)
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(f'data/analysis/pace_reaction_{F0}_{F1}.csv', index=False)

# chance level: shift B's events by a random offset, same counting
rng = np.random.default_rng(0)
L = F1 - F0 + 1
chance = {'teammate': [], 'opponent': []}
BINS = np.arange(1, int(MAXLAG * FPS) + 3, 2) - 0.5  # 0.08 s bins
lag_ch = {'teammate': np.zeros(len(BINS) - 1), 'opponent': np.zeros(len(BINS) - 1)}
for _ in range(200):
    sh = {pl: np.sort((byp[pl] - F0 + rng.integers(2 * FPS, 6 * FPS)) % L + F0) for pl in TEAM}
    hits = {'teammate': [0, 0], 'opponent': [0, 0]}
    for e in ev.itertuples():
        for pl in TEAM:
            if pl == e.player:
                continue
            rel = 'teammate' if TEAM[pl] == e.team else 'opponent'
            hits[rel][1] += 1
            b = first_after(sh[pl], e.frame)
            hits[rel][0] += b is not None
            if b is not None:
                lag_ch[rel] += np.histogram([b - e.frame], BINS)[0] / 200
    for k in hits:
        chance[k].append(hits[k][0] / max(hits[k][1], 1))

summ = d.groupby('relation').agg(pairs=('player', 'size'), reacted=('reaction_s', 'count'), median_s=('reaction_s', 'median'),
                                 q25=('reaction_s', lambda v: v.quantile(.25)), q75=('reaction_s', lambda v: v.quantile(.75)),
                                 same_dir_pct=('same_direction', lambda v: v.dropna().astype(bool).mean() * 100))
summ['reacted_pct'] = summ.reacted / summ.pairs * 100
summ['chance_pct'] = [np.mean(chance[k]) * 100 for k in summ.index]
print(summ.round(2).to_string(), '\n')
# excess over chance per delay bin: where real answers pile up
exc = []
for rel in ('teammate', 'opponent'):
    real = np.histogram(d[(d.relation == rel)].reaction_s.dropna() * FPS, BINS)[0]
    for b0, b1, r_, c_ in zip(BINS[:-1], BINS[1:], real, lag_ch[rel]):
        exc.append(dict(relation=rel, lag_from_s=round((b0 + 0.5) / FPS, 2), lag_to_s=round((b1 - 0.5) / FPS, 2), real=int(r_),
                        chance=round(c_, 1), excess=round(r_ - c_, 1)))
exc = pd.DataFrame(exc)
exc.to_csv(f'data/analysis/pace_reaction_excess_{F0}_{F1}.csv', index=False)
print(exc.to_string(), '\n')
per = d.groupby(['player', 'relation']).agg(n=('reaction_s', 'count'), of=('player', 'size'), median_s=('reaction_s', 'median')).round(2)
print(per.to_string(), '\n')

# cross-correlation of |a| (play frames), lags 0..1 s, per ordered pair
xc = []
for A in TEAM:
    for B in TEAM:
        if A == B:
            continue
        a1, a2 = np.abs(ACC[A]), np.abs(ACC[B])
        best = (None, -1)
        cs = []
        for lag in range(0, int(MAXLAG * FPS) + 1):
            x, y = a1[:len(a1) - lag], a2[lag:]
            m = inplay[F0:F0 + len(x)] & np.isfinite(x) & np.isfinite(y)
            c = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 50 else np.nan
            cs.append(c)
            if c > best[1]:
                best = (lag, c)
        xc.append(dict(leader=A, player=B, relation='teammate' if TEAM[A] == TEAM[B] else 'opponent',
                       peak_lag_s=round(best[0] / FPS, 2), peak_corr=round(best[1], 2), corr_at_0=round(cs[0], 2)))
xc = pd.DataFrame(xc)
xc.to_csv(f'data/analysis/pace_xcorr_{F0}_{F1}.csv', index=False)
print(xc.to_string())

if CHART:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'teammate': '#2a78d6', 'opponent': '#eb6834'}
    TITLE = {'teammate': 'After the teammate changes pace', 'opponent': 'After an opponent changes pace'}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, rel in zip(axs, ('teammate', 'opponent')):
        g = exc[exc.relation == rel]
        xm = (g.lag_from_s + g.lag_to_s) / 2
        ax.bar(xm, g.real, width=0.068, color=COL[rel], zorder=3, label=f'answers in the match ({rel})')
        ax.step(np.r_[g.lag_from_s - 0.02, g.lag_to_s.iloc[-1] + 0.02], np.r_[g.chance, g.chance.iloc[-1]], where='post',
                color='#1d1d1b', lw=1.6, zorder=4, label='expected by chance' if rel == 'teammate' else None)
        top = g.loc[g.excess.idxmax()]
        ax.annotate(f'peak {top.lag_from_s:.2f}-{top.lag_to_s:.2f} s\n{int(top.real)} vs {top.chance:.0f} by chance',
                    ((top.lag_from_s + top.lag_to_s) / 2, top.real), xytext=(16, -2), textcoords='offset points', fontsize=8.5,
                    color='#1d1d1b', va='top')
        ax.set_title(TITLE[rel], loc='left', fontsize=10, color='#1d1d1b')
        ax.set_xlabel('delay until the player changes his own pace (s)', color='#52514c')
        ax.grid(axis='y', color='#e5e4dd', lw=0.6)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.spines['left'].set_color('#b9b8b0'); ax.spines['bottom'].set_color('#b9b8b0'); ax.tick_params(colors='#52514c', labelsize=8)
    axs[0].set_ylabel('number of answers', color='#52514c')
    fig.legend(loc='upper right', ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f'Reaction to a change of pace, first {(F1 - F0 + 1) / FPS:.0f} s', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(CHART, dpi=130, facecolor='#fcfcfb'); print('chart:', CHART)

if VIDEO:
    import cv2
    sys.path.insert(0, 'scripts/ball_height')
    from traj import Camera
    cam = Camera()
    W, H, S = 960, 540, 0.5
    BGR = {'teammate': (214, 120, 42), 'opponent': (52, 104, 235)}
    pos = p.set_index(['v2_frame', 'player'])
    HOLD = 15
    vid = d.dropna(subset=['react_frame'])
    vid = vid[vid.reaction_s.between(0.2, 0.4)].sort_values('reaction_s')

    def feet(pl, f):
        try:
            x, y = pos.loc[(f, pl)].values.ravel()[:2]
        except KeyError:
            return None
        return (cam.project(np.array([[x, y, 0.0]]))[0] * S).astype(int) if np.isfinite([x, y]).all() else None

    def label(img, t, org, sc, col, th=1):
        (tw, tht), bl = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        x, y = org
        sub = img[max(0, y - tht - 3):y + bl + 1, max(0, x - 3):x + tw + 3]
        sub[:] = (sub * 0.35).astype(np.uint8)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
    cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', VIDEO], stdin=subprocess.PIPE)
    for f in range(F0, F1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        for pl in TEAM:  # live speed under every player
            q = feet(pl, f)
            if q is not None and np.isfinite(SP[pl][f - F0]):
                label(im, f'{SP[pl][f - F0]:.1f} m/s', (q[0] - 24, q[1] + 40), 0.38, (225, 225, 225))
        # each pace change: white ring on the player who changed pace; his answerers get a coloured ring at their change
        for e in ev[(ev.frame <= f) & (f - ev.frame < HOLD)].itertuples():
            q = feet(e.player, f)
            if q is not None:
                cv2.ellipse(im, tuple(q), (20, 7), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
                label(im, e.kind, (q[0] - 34, q[1] + 22), 0.45, (255, 255, 255))
        # answers in the 0.2-0.4 s peak only (the delays that stand out above chance), one per player
        shown = set()
        for r in vid.itertuples():
            if 0 <= f - r.react_frame < HOLD and r.player not in shown:
                q = feet(r.player, f)
                if q is None:
                    continue
                shown.add(r.player)
                c = BGR[r.relation]
                rad = 14 + int(10 * (f - r.react_frame) / HOLD)
                cv2.ellipse(im, tuple(q), (rad, rad // 3), 0, 0, 360, c, 2, cv2.LINE_AA)
                label(im, f'+{r.reaction_s:.2f} s after {NAMES[r.leader]}', (q[0] - 60, q[1] - 40), 0.45, c)
        ov = im.copy(); cv2.rectangle(ov, (122, H - 64), (640, H - 26), (25, 25, 25), -1); im = cv2.addWeighted(ov, 0.6, im, 0.4, 0)
        x0 = 130
        for col, lab in (((255, 255, 255), 'changes pace'), (BGR['teammate'], 'teammate answers'), (BGR['opponent'], 'opponent answers')):
            cv2.circle(im, (x0 + 8, H - 49), 6, col, 2, cv2.LINE_AA)
            cv2.putText(im, lab, (x0 + 20, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            x0 += 20 + cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 24
        cv2.putText(im, f'frame {f} ({f / FPS:.1f} s)  {"in play" if inplay[f] else "dead time (not counted)"}   answers shown: 0.2-0.4 s after', (130, H - 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait(); print('video:', VIDEO)
