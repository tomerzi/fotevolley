"""Distance of every player from the court lines (blue lines, 16 x 8 m) frame by frame.

From each player's ground position (data/analysis/player_positions_v3.csv, m) to the three lines of the player's own
half - the baseline (x = 0 or 16), the far sideline (y = 0) and the near sideline (y = 8) - and to the net (x = 8).
Distances are signed: positive inside the court, negative outside (beyond that line).
Writes data/analysis/line_distances_<start>_<end>.csv, a small-multiples chart (PNG, one panel per player) and, with
--video, an overlay clip with a thin line from each player's feet to each line and its length.
usage: python scripts/analysis/line_distances.py [START END] [--video OUT.mp4] [--chart OUT.png]
"""
import os, sys, subprocess
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

args = [a for a in sys.argv[1:] if not a.startswith('--')]
F0, F1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, 749)
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
VIDEO, CHART = opt('--video'), opt('--chart')
LINES = ['baseline', 'far sideline', 'near sideline', 'net']
COL_HEX = {'baseline': '#2a78d6', 'far sideline': '#eb6834', 'near sideline': '#1baf7a', 'net': '#eda100'}  # validated categorical slots 1-4
NAMES = {'Franklin': 'Franklin (BRA)', 'Hiltinho': 'Hiltinho (BRA)', 'Maor_Haas': 'Maor Haas (ISR)', 'Ron_Ben_Ishay': 'Ron Ben Ishay (ISR)'}
ORDER = ['Maor_Haas', 'Ron_Ben_Ishay', 'Franklin', 'Hiltinho']

p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y', 'occluded'])
p = p[(p.v2_frame >= F0) & (p.v2_frame <= F1)].drop_duplicates(['v2_frame', 'player']).rename(columns={'v2_frame': 'frame'})
left = p.x < 8
p['baseline'] = np.where(left, p.x, 16 - p.x)
p['far sideline'] = p.y
p['near sideline'] = 8 - p.y
p['net'] = np.abs(8 - p.x)
p['nearest_line'] = p[['baseline', 'far sideline', 'near sideline']].idxmin(axis=1)
p['nearest_m'] = p[['baseline', 'far sideline', 'near sideline']].min(axis=1)
p['time_s'] = (p.frame / 25).round(2)
p.round(3).to_csv(f'data/analysis/line_distances_{F0}_{F1}.csv', index=False)
summ = p.groupby('player')[LINES + ['nearest_m']].median().round(2).loc[ORDER]
print(summ.to_string())

if CHART:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, sharey=True)
    for ax, pl in zip(axs, ORDER):
        g = p[p.player == pl].sort_values('frame')
        for ln in LINES:
            ax.plot(g.time_s, g[ln], color=COL_HEX[ln], lw=2, solid_capstyle='round')
            last = g.dropna(subset=[ln]).iloc[-1]
            ax.annotate(ln, (last.time_s, last[ln]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color='#52514c')
        ax.axhline(0, color='#8a8980', lw=0.8)
        ax.set_title(NAMES[pl], loc='left', fontsize=10, color='#1d1d1b')
        ax.set_ylabel('m', color='#52514c'); ax.grid(axis='y', color='#e5e4dd', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_color('#b9b8b0'); ax.spines['bottom'].set_color('#b9b8b0'); ax.tick_params(colors='#52514c', labelsize=8)
    axs[-1].set_xlabel('time in the video (s)', color='#52514c')
    handles = [plt.Line2D([0], [0], color=COL_HEX[ln], lw=2) for ln in LINES]
    fig.legend(handles, LINES, loc='upper right', ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f'Distance from the court lines, first {(F1 - F0 + 1) / 25:.0f} s (below 0 = outside the line)', x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig.savefig(CHART, dpi=130, facecolor='#fcfcfb'); print('chart:', CHART)

if VIDEO:
    cam = Camera()
    W, H, S = 960, 540, 0.5
    BGR = {k: tuple(int(v.lstrip('#')[i:i + 2], 16) for i in (4, 2, 0)) for k, v in COL_HEX.items()}
    byf = {f: g for f, g in p.groupby('frame')}
    cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', VIDEO], stdin=subprocess.PIPE)
    for f in range(F0, F1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        glow = np.zeros_like(im, np.float32); core = np.zeros_like(im, np.float32); labels = []
        for r in (byf[f].itertuples() if f in byf else []):
            if not np.isfinite([r.x, r.y]).all():
                continue
            foot = np.array([r.x, r.y, 0.0])
            ends = {'baseline': [0.0 if r.x < 8 else 16.0, r.y, 0.0], 'far sideline': [r.x, 0.0, 0.0],
                    'near sideline': [r.x, 8.0, 0.0], 'net': [8.0, r.y, 0.0]}
            for ln, E in ends.items():
                P = (cam.project(foot + np.linspace(0, 1, 13)[:, None] * (np.array(E) - foot)) * S).astype(int)
                c = np.array(BGR[ln], np.float32)
                for i in range(12):
                    cv2.line(glow, tuple(P[i]), tuple(P[i + 1]), (c * 0.5).tolist(), 4, cv2.LINE_AA)
                    cv2.line(core, tuple(P[i]), tuple(P[i + 1]), c.tolist(), 1, cv2.LINE_AA)
                cv2.circle(core, tuple(P[-1]), 3, c.tolist(), -1, cv2.LINE_AA)
                labels.append((tuple(P[6]), f'{p.loc[r.Index, ln]:.1f}'))
        glow = cv2.GaussianBlur(glow, (0, 0), 2)
        im = np.clip(im.astype(np.float32) + glow * 0.9 + core, 0, 255).astype(np.uint8)
        for (x, y), t in labels:
            cv2.putText(im, t, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, t, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        ov = im.copy(); cv2.rectangle(ov, (122, H - 64), (560, H - 26), (25, 25, 25), -1); im = cv2.addWeighted(ov, 0.6, im, 0.4, 0)
        x0 = 130
        for ln in LINES:
            cv2.line(im, (x0, H - 49), (x0 + 18, H - 49), BGR[ln], 2, cv2.LINE_AA)
            cv2.putText(im, ln, (x0 + 24, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            x0 += 24 + cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 20
        cv2.putText(im, f'frame {f} ({f / 25:.1f} s)   distance in m, from the feet to each line of the own half and to the net', (130, H - 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait(); print('video:', VIDEO)
