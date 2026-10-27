"""Render the per-frame ball height on top of the main video.

Draws the ball, a dashed drop line to the point below it on the sand and
the height, plus a 0-6 m gauge, and each player's ground speed under
their feet (data/analysis/player_positions_v3.csv, in m with a 16 x 8 court). Writes 960x540 H.264 in 60 s segments
(each well under 15 MB) to the folder given as the first argument.
"""
import sys, os, subprocess
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj import Camera
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)  # all paths below are relative to the repo root


OUT = sys.argv[1]
W, H, SCALE = 960, 540, 0.5
SEG = 60
os.makedirs(OUT, exist_ok=True)

cam = Camera()
h = pd.read_csv('data/analysis/ball_height_v2.csv')
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
n = len(h)
XY = h[['court_x_m', 'court_y_m']].values
hm = h.height_m.values
has = ~np.isnan(hm)
# drop line: 12 points from the sand up to the ball
ts = np.linspace(0, 1, 13)
line = np.full((n, len(ts), 2), np.nan)
idx = np.flatnonzero(has)
pts = np.concatenate([np.c_[XY[idx], -hm[idx] * t] for t in ts])
proj = cam.project(pts).reshape(len(ts), len(idx), 2).transpose(1, 0, 2)
line[idx] = proj * SCALE
ball = b[['center_x', 'center_y']].values * SCALE
# frames inside a flight without a box: draw the ball where the fitted flight puts it
nobox = np.flatnonzero(has & np.isnan(ball[:, 0]))
ball[nobox] = line[nobox, -1]
ok_line = has & np.all(np.isfinite(line.reshape(n, -1)), 1) & \
    (np.abs(line[:, 0, 0] - W / 2) < W) & (np.abs(line[:, 0, 1] - H / 2) < H)

# player speed labels under the feet
pb = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'player', 'x1', 'x2', 'y2'])
ps = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'speed', 'airborne', 'posture'])
ps = ps.rename(columns={'v2_frame': 'frame'})
ps['speed_kmh'] = ps.speed * 8 / 9 * 3.6           # court units (18 x 9 lines) -> m for a 16 x 8 court
ps['reliable'] = ~ps.airborne & ~ps.posture.isin(['hidden', 'none'])
pl = pb.merge(ps, on=['frame', 'player'], how='left')
pl = {f: t[['player', 'x1', 'x2', 'y2', 'speed_kmh', 'reliable']].values for f, t in pl.groupby('frame')}
PCOL = {'ISR_A': (214, 120, 42), 'ISR_B': (52, 104, 235), 'BRA_A': (122, 175, 27), 'BRA_B': (0, 161, 237)}  # BGR, chart slots 1-4

COL = {'flight_fit': (230, 140, 40), 'size_only': (40, 170, 230)}  # BGR: blue / amber
FONT = cv2.FONT_HERSHEY_SIMPLEX
GX, GY0, GY1, GMAX = W - 46, 80, H - 62, 6.0


def text(im, s, org, size, color=(255, 255, 255), th=2):
    cv2.putText(im, s, org, FONT, size, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, FONT, size, color, th, cv2.LINE_AA)


def gauge(im, v, color):
    ov = im.copy()
    cv2.rectangle(ov, (GX - 30, GY0 - 30), (W - 6, GY1 + 22), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.45, im, 0.55, 0, im)
    cv2.line(im, (GX, GY0), (GX, GY1), (220, 220, 220), 2)
    for m in range(int(GMAX) + 1):
        y = int(GY1 - m / GMAX * (GY1 - GY0))
        cv2.line(im, (GX - 5, y), (GX + 5, y), (220, 220, 220), 1)
        text(im, f'{m}', (GX + 9, y + 5), 0.4, th=1)
    ynet = int(GY1 - 2.2 / GMAX * (GY1 - GY0))
    cv2.line(im, (GX - 12, ynet), (GX + 4, ynet), (60, 110, 240), 2)
    text(im, 'net', (GX - 30, ynet - 6), 0.35, (120, 160, 255), 1)
    text(im, 'm', (GX - 4, GY0 - 12), 0.45, th=1)
    if v is not None:
        y = int(GY1 - min(v, GMAX) / GMAX * (GY1 - GY0))
        cv2.circle(im, (GX, y), 7, color, -1)
        cv2.circle(im, (GX, y), 7, (255, 255, 255), 2)


cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
ff = subprocess.Popen(
    ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
     '-r', '25', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
     '-pix_fmt', 'yuv420p', '-g', '50', '-force_key_frames', f'expr:gte(t,n_forced*{SEG})',
     '-f', 'segment', '-segment_time', str(SEG), '-reset_timestamps', '1',
     '-segment_format_options', 'movflags=+faststart', f'{OUT}/part%02d.mp4'],
    stdin=subprocess.PIPE)
for f in range(n):
    ok, im = cap.read()
    if not ok:
        break
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    meth = h.method[f]
    v = hm[f] if has[f] else None
    color = COL.get(meth, (160, 160, 160))
    if v is not None and np.all(np.isfinite(ball[f])):
        u = tuple(np.round(ball[f]).astype(int))
        if ok_line[f]:
            L = line[f].astype(int)
            for k in range(0, len(ts) - 1, 2):
                cv2.line(im, tuple(L[k]), tuple(L[k + 1]), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.ellipse(im, tuple(L[0]), (10, 4), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(im, u, 12, color, 2, cv2.LINE_AA)
        text(im, f'{v:.2f} m', (u[0] + 16, u[1] + 6), 0.65)
    gauge(im, v, color)
    for p, x1, x2, y2, kmh, rel in pl.get(f, []):
        if not (kmh == kmh):  # NaN
            continue
        cx, cy = int((x1 + x2) / 2 * SCALE), int(min(y2 * SCALE + 16, H - 34))
        s = f'{kmh:.0f} km/h' if rel else '--'
        (tw, _), _ = cv2.getTextSize(s, FONT, 0.45, 1)
        cv2.rectangle(im, (cx - tw // 2 - 10, cy - 12), (cx + tw // 2 + 4, cy + 4), (25, 25, 25), -1)
        cv2.circle(im, (cx - tw // 2 - 4, cy - 4), 3, PCOL.get(p, (200, 200, 200)), -1)
        cv2.putText(im, s, (cx - tw // 2 + 2, cy), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    label = {'flight_fit': 'ballistic fit', 'size_only': 'size estimate'}.get(meth, 'ball not found')
    cv2.rectangle(im, (0, H - 26), (W, H), (25, 25, 25), -1)
    cv2.putText(im, f'{f // 1500}:{f / 25 % 60:04.1f}   frame {f}', (12, H - 8), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(im, f'ball height {v:.2f} m   ({label})' if v is not None else label, (260, H - 8), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ff.stdin.write(im.tobytes())
    if f % 1500 == 0:
        print(f, flush=True)
ff.stdin.close()
ff.wait()
