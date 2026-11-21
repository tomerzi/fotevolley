"""Reaction time: how long after one player's action (serve, reception, set, attack) each other player starts to move,
and how fast he then moves. Split by who reacts: the actor's teammate or an opponent.

Stimuli: the in-play player touches of data/contacts/ball_contacts_v3.csv (reception, set, attack, from
data/analysis/touch_actions_v3.csv), v3 frames mapped to v2. Serves are left out: their 'flight_start' is where the
ball tracking begins, often ~1 s after the kick, which happens out of the frame.
Movement: the players' ground positions (data/analysis/player_positions_v3.csv, 25 fps, m) -> velocity per frame
(tracking jumps faster than 9 m/s dropped).
Onset: the direction the player ends up going is from his spot at the action to the farthest point he reaches before
the next ball event (at least 0.5 s, at most 1.2 s later). If he gets less than 0.5 m away he did not move ('stayed').
His speed along that direction must reach 1.0 m/s on 2 frames in a row; the onset is the last frame before that with
less than 0.3 m/s along it. If he was already moving that way at 1 m/s 0.2 s before the touch, he read it in advance
('already moving', no reaction time). Reaction time = onset - action frame (negative = started just before the touch).
After the onset: time to reach 2 m/s, peak speed in the next 1 s and the distance covered in that second.
The frame grid is 40 ms, so every time is +- 0.04 s; touch frames are +- 1 frame.
Writes data/analysis/reaction_times_<start>_<end>.csv, a chart with --chart and an overlay clip with --video.
usage: python scripts/analysis/reaction_times.py [START END] [--chart OUT.png] [--video OUT.mp4]
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
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
NAMES = {'Franklin': 'Franklin', 'Hiltinho': 'Hiltinho', 'Maor_Haas': 'Maor Haas', 'Ron_Ben_Ishay': 'Ron Ben Ishay'}

fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
ev = pd.read_csv('data/contacts/ball_contacts_v3.csv')
act = pd.read_csv('data/analysis/touch_actions_v3.csv')
serve_team = act[act.action == 'serve'].groupby('rally').team.first()
act_of = act.dropna(subset=['touch']).set_index('frame').action
ev = ev[ev.in_play & (ev.event_type == 'touch')].copy()
ev['v2'] = ev.frame.map(fm)
ev = ev.dropna(subset=['v2']); ev['v2'] = ev.v2.astype(int)
ev = ev[(ev.v2 >= F0 + 10) & (ev.v2 <= F1)]
stim = []
for e in ev.itertuples():
    if e.event_kind == 'flight_start':
        team = serve_team.get(e.rally)
        if team is None:
            continue
        stim.append(dict(v2=e.v2, rally=e.rally, actor=None, team=team, action='serve'))
    else:
        stim.append(dict(v2=e.v2, rally=e.rally, actor=e.player, team=TEAM[e.player], action=act_of.get(e.frame, 'touch')))
stim = pd.DataFrame(stim)

p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y']).drop_duplicates(['v2_frame', 'player'])
P = {}
for pl, g in p.groupby('player'):
    g = g.set_index('v2_frame').reindex(range(F0 - 20, F1 + 60))
    xy = g[['x', 'y']].to_numpy()
    v = np.gradient(xy, axis=0) * FPS
    v[np.linalg.norm(v, axis=1) > 9] = np.nan
    P[pl] = (xy, v)
off = F0 - 20


nxt = np.sort(stim.v2.unique())


def onset(pl, t):
    """-> (onset frame index or None, status)"""
    xy, v = P[pl]
    i = t - off
    later = nxt[nxt > t]
    t_end = int(np.clip((later[0] if len(later) else t + 30) - t, 12, 30))
    seg = xy[i:i + t_end + 1]
    if not np.isfinite(seg[0]).all():
        return None, 'no data'
    dist = np.linalg.norm(seg - seg[0], axis=1)
    if np.nanmax(dist) < 0.5:
        return None, 'stayed'
    j = int(np.nanargmax(dist)); u = (seg[j] - seg[0]) / dist[j]
    s_ = v @ u
    lo = i - 8
    for c in range(i - 5, i + j + 1):
        if s_[c] >= 1.0 and s_[c + 1] >= 1.0:
            if c <= i - 5:
                return None, 'already moving'
            k = c
            while k > lo and not (s_[k - 1] < 0.3):
                k -= 1
            if k <= lo:
                return None, 'already moving'
            return k, 'reacted'
    return None, 'stayed'


rows = []
for s in stim.itertuples():
    for pl in TEAM:
        if pl == s.actor:
            continue
        rel = 'teammate' if TEAM[pl] == s.team else 'opponent'
        k, status = onset(pl, s.v2)
        r = dict(status=status, frame=s.v2, time_s=round(s.v2 / FPS, 2), rally=s.rally, action=s.action, actor=s.actor or f'{s.team} server',
                 actor_team=s.team, player=pl, relation=rel)
        if k is None:
            r['reaction_s'] = np.nan
        else:
            xy, v = P[pl]
            sp = np.linalg.norm(v[k:k + FPS], axis=1)
            fast = np.where(sp >= 2.0)[0]
            r.update(onset_frame=k + off, reaction_s=round((k + off - s.v2) / FPS, 2),
                     to_2ms_s=round(fast[0] / FPS, 2) if len(fast) else np.nan,
                     peak_speed_1s=round(np.nanmax(sp), 2),
                     dist_1s_m=round(float(np.nansum(np.linalg.norm(np.diff(xy[k:k + FPS + 1], axis=0), axis=1))), 2))
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(f'data/analysis/reaction_times_{F0}_{F1}.csv', index=False)
pd.set_option('display.width', 200)
print(d.to_string())
print()
print(d.groupby(['relation', 'action']).agg(n=('reaction_s', 'size'), reacted=('reaction_s', 'count'), median_s=('reaction_s', 'median'),
                                            to_2ms=('to_2ms_s', 'median'), peak=('peak_speed_1s', 'median')).round(2).to_string())
print()
print(d.groupby(['player', 'relation']).agg(n=('reaction_s', 'count'), median_s=('reaction_s', 'median'),
                                            to_2ms=('to_2ms_s', 'median'), peak=('peak_speed_1s', 'median')).round(2).to_string())

if CHART:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'teammate': '#2a78d6', 'opponent': '#eb6834'}
    ORDER = ['Maor_Haas', 'Ron_Ben_Ishay', 'Franklin', 'Hiltinho']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axs[0]
    for yi, pl in enumerate(ORDER):
        for rel, dy in (('teammate', -0.12), ('opponent', 0.12)):
            g = d[(d.player == pl) & (d.relation == rel)].dropna(subset=['reaction_s'])
            ax.scatter(g.reaction_s, np.full(len(g), yi + dy), s=46, color=COL[rel], edgecolor='#fcfcfb', linewidth=1.5, zorder=3,
                       label=('after a teammate\'s action' if rel == 'teammate' else 'after an opponent\'s action') if yi == 0 else None)
            if len(g):
                m = g.reaction_s.median()
                ax.plot([m, m], [yi + dy - 0.1, yi + dy + 0.1], color='#1d1d1b', lw=2, zorder=4)
    ax.axvline(0, color='#8a8980', lw=0.8)
    ax.set_yticks(range(4)); ax.set_yticklabels([NAMES[p_] for p_ in ORDER])
    ax.set_xlabel('reaction time (s): from the other player\'s touch to the start of the move', color='#52514c')
    ax.set_title('Reaction time (dot = one action, black tick = median)', loc='left', fontsize=10, color='#1d1d1b')
    ax.annotate('moved before\nthe touch', (0, 3.45), xytext=(-4, 0), textcoords='offset points', ha='right', fontsize=8, color='#52514c')
    ax2 = axs[1]
    for yi, pl in enumerate(ORDER):
        for rel, dy in (('teammate', -0.12), ('opponent', 0.12)):
            g = d[(d.player == pl) & (d.relation == rel)].dropna(subset=['peak_speed_1s'])
            ax2.scatter(g.peak_speed_1s, np.full(len(g), yi + dy), s=46, color=COL[rel], edgecolor='#fcfcfb', linewidth=1.5, zorder=3)
            if len(g):
                m = g.peak_speed_1s.median()
                ax2.plot([m, m], [yi + dy - 0.1, yi + dy + 0.1], color='#1d1d1b', lw=2, zorder=4)
    ax2.set_xlabel('peak speed in the first second (m/s)', color='#52514c')
    ax2.set_title('How fast the move is', loc='left', fontsize=10, color='#1d1d1b')
    for a in axs:
        a.grid(axis='x', color='#e5e4dd', lw=0.6)
        for s_ in ('top', 'right'):
            a.spines[s_].set_visible(False)
        a.spines['left'].set_color('#b9b8b0'); a.spines['bottom'].set_color('#b9b8b0'); a.tick_params(colors='#52514c', labelsize=8)
    axs[0].invert_yaxis()
    fig.legend(loc='upper right', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f'Reaction to another player\'s action, first {(F1 - F0 + 1) / FPS:.0f} s', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CHART, dpi=130, facecolor='#fcfcfb'); print('chart:', CHART)

if VIDEO:
    import cv2
    sys.path.insert(0, 'scripts/ball_height')
    from traj import Camera
    cam = Camera()
    W, H, S = 960, 540, 0.5
    BGR = {'teammate': (214, 120, 42), 'opponent': (52, 104, 235)}
    pos = p.set_index(['v2_frame', 'player'])
    cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', VIDEO], stdin=subprocess.PIPE)
    HOLD = 30  # frames a ring takes to grow

    def label(img, t, org, sc, col, th=1):
        (tw, tht), bl = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        x, y = org
        sub = img[max(0, y - tht - 3):y + bl + 1, max(0, x - 3):x + tw + 3]
        sub[:] = (sub * 0.35).astype(np.uint8)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    def feet(pl, f):
        try:
            x, y = pos.loc[(f, pl)].values.ravel()[:2]
        except KeyError:
            return None
        if not np.isfinite([x, y]).all():
            return None
        return (cam.project(np.array([[x, y, 0.0]]))[0] * S).astype(int)
    for f in range(F0, F1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        # the current action = the latest one whose window (from 8 frames before its touch) has begun
        cur = stim[stim.v2 - 8 <= f]
        if len(cur) and f - cur.v2.iloc[-1] < 1.6 * FPS:
            s = cur.iloc[-1]
            if f >= s.v2:
                q = feet(s.actor, f)
                if q is not None:
                    cv2.ellipse(im, tuple(q), (22, 8), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
                    label(im, s.action, (q[0] - 30, q[1] + 24), 0.5, (255, 255, 255), 1)
            el = (f - s.v2) / FPS
            txt = f'{s.action} by {NAMES[s.actor]}   {el:+.2f} s'
            label(im, txt, (330, 28), 0.6, (255, 255, 255), 1)
            # reactions to it: a coloured ring from the onset frame on, with the reaction time; grey note otherwise
            for r in d[d.frame == s.v2].itertuples():
                q = feet(r.player, f)
                if q is None:
                    continue
                if r.status == 'reacted' and f >= r.onset_frame:
                    c = BGR[r.relation]
                    rad = 14 + int(10 * min(1, (f - r.onset_frame) / HOLD))
                    cv2.ellipse(im, tuple(q), (rad, rad // 3), 0, 0, 360, c, 2, cv2.LINE_AA)
                    t = f'{r.reaction_s:+.2f} s'
                    label(im, t, (q[0] - 26, q[1] + 26), 0.55, c, 2)
                elif r.status in ('already moving', 'stayed') and f >= s.v2:
                    label(im, r.status, (q[0] - 34, q[1] + 22), 0.42, (200, 200, 200), 1)
        ov = im.copy(); cv2.rectangle(ov, (122, H - 64), (600, H - 26), (25, 25, 25), -1); im = cv2.addWeighted(ov, 0.6, im, 0.4, 0)
        x0 = 130
        for rel, lab in (('teammate', 'teammate reacts'), ('opponent', 'opponent reacts')):
            cv2.circle(im, (x0 + 8, H - 49), 6, BGR[rel], 2, cv2.LINE_AA)
            cv2.putText(im, lab, (x0 + 20, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            x0 += 20 + cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 24
        cv2.putText(im, f'frame {f} ({f / FPS:.1f} s)   ring = the moment the player starts to move, time after the touch',
                    (130, H - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait(); print('video:', VIDEO)
