"""Where each player's shoulders face, relative to the net, with a ray drawn from each shoulder.

For every player and frame: the ViTPose shoulder points (data/pose/vitpose_body25.csv) are placed in 3-D by
intersecting their camera rays with the horizontal plane at shoulder height (0.82 x the player's measured stature,
data/analysis/player_statures.json). The shoulder line gives the facing direction (perpendicular to it, forwards by
the pose's left/right labels: right = R - L, facing = up x right). Angle to the net: 0 = square to the net,
90 = side-on, 180 = back to the net; its sign says which way the player is turned (+ towards the near sideline / the
camera side, - towards the far sideline). Lightly smoothed over 5 frames; frames where the shoulders are missing,
low-confidence or implausibly short/long (< 0.22 m or > 0.60 m apart on the sand plane) are left out.
Writes data/analysis/shoulder_orientation_<start>_<end>.csv and, with --video, an overlay clip with two rays per
player (one from each shoulder, 1.6 m long, along the facing direction) and a top-down court inset.
usage: python scripts/analysis/shoulder_rays.py [START_FRAME END_FRAME] [--video OUT.mp4]
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

args = [a for a in sys.argv[1:] if not a.startswith('--')]
F0, F1 = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, 749)
VIDEO = sys.argv[sys.argv.index('--video') + 1] if '--video' in sys.argv else None
cam = Camera()
STAT = {k: v['stature_m'] for k, v in json.load(open('data/analysis/player_statures.json')).items()}
NAMES = {'Franklin': 'Franklin', 'Hiltinho': 'Hiltinho', 'Maor_Haas': 'Maor Haas', 'Ron_Ben_Ishay': 'Ron Ben Ishay'}
COL = {'Maor_Haas': (214, 120, 42), 'Ron_Ben_Ishay': (52, 104, 235), 'Franklin': (122, 175, 27), 'Hiltinho': (0, 161, 237)}  # BGR
MIN_S = 0.3

pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player', 'r_shoulder_x', 'r_shoulder_y', 'r_shoulder_s',
                                                             'l_shoulder_x', 'l_shoulder_y', 'l_shoulder_s'])
pose = pose[(pose.frame >= F0) & (pose.frame <= F1)].drop_duplicates(['frame', 'player'])
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y'])
pos = pos[(pos.v2_frame >= F0) & (pos.v2_frame <= F1)].drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])


def on_plane(uv, z):
    d = cam.ray(np.asarray(uv, float).reshape(-1, 2))
    return cam.C + ((-z - cam.C[2]) / d[:, 2])[:, None] * d


rows = []
for r in pose.itertuples():
    ok = min(r.r_shoulder_s, r.l_shoulder_s) >= MIN_S and r.player in STAT
    if not ok:
        continue
    zh = 0.82 * STAT[r.player]
    R, L = on_plane([[r.r_shoulder_x, r.r_shoulder_y], [r.l_shoulder_x, r.l_shoulder_y]], zh)
    right = (R - L)[:2]
    w = float(np.linalg.norm(right))
    if not (0.22 <= w <= 0.60):
        continue
    right /= w
    face = np.array([right[1], -right[0]])  # up x right, world Z down
    try:
        px, py = pos.loc[(r.frame, r.player)].values.ravel()[:2]
    except KeyError:
        px, py = (R[0] + L[0]) / 2, (R[1] + L[1]) / 2
    to_net = np.array([1.0, 0.0]) if px < 8 else np.array([-1.0, 0.0])
    ang = np.degrees(np.arctan2(to_net[0] * face[1] - to_net[1] * face[0], to_net @ face))  # signed, from to_net to face
    # sign convention: + = turned towards the near sideline (camera side, +Y) whatever the court side
    ang_side = ang if px < 8 else -ang
    rows.append(dict(frame=r.frame, player=r.player, x=px, y=py, face_x=face[0], face_y=face[1], shoulder_w=w,
                     angle_to_net=ang_side, Rx=R[0], Ry=R[1], Lx=L[0], Ly=L[1], z=zh))
d = pd.DataFrame(rows).sort_values(['player', 'frame'])
# smooth the facing direction (vector median over 5 frames) per player
out = []
for p, g in d.groupby('player'):
    g = g.set_index('frame').reindex(range(F0, F1 + 1))
    fx = g.face_x.rolling(5, center=True, min_periods=2).median(); fy = g.face_y.rolling(5, center=True, min_periods=2).median()
    n = np.hypot(fx, fy)
    g['face_x_s'], g['face_y_s'] = fx / n, fy / n
    g['player'] = p
    out.append(g.reset_index().dropna(subset=['x']))
d = pd.concat(out).dropna(subset=['face_x_s', 'Rx'])
d['facing_deg'] = np.degrees(np.arctan2(d.face_y_s, d.face_x_s)).round(1)
tn = np.where(d.x < 8, 1.0, -1.0)
a = np.degrees(np.arctan2(tn * d.face_y_s, tn * d.face_x_s))
d['angle_to_net_deg'] = np.where(d.x < 8, a, -a).round(1)
d['abs_angle_to_net_deg'] = d.angle_to_net_deg.abs()
cols = ['frame', 'player', 'x', 'y', 'facing_deg', 'angle_to_net_deg', 'abs_angle_to_net_deg', 'shoulder_w', 'Rx', 'Ry', 'Lx', 'Ly', 'z']
d[cols].round(3).to_csv(f'data/analysis/shoulder_orientation_{F0}_{F1}.csv', index=False)
print(d.groupby('player').agg(frames=('frame', 'size'), median_abs_angle=('abs_angle_to_net_deg', 'median'),
                              square_to_net_pct=('abs_angle_to_net_deg', lambda v: (v < 30).mean() * 100),
                              side_on_pct=('abs_angle_to_net_deg', lambda v: ((v >= 60) & (v <= 120)).mean() * 100),
                              back_pct=('abs_angle_to_net_deg', lambda v: (v > 150).mean() * 100)).round(1).to_string())

if VIDEO:
    W, H, S = 960, 540, 0.5
    # inset: court seen from above, top-left, 16 x 8 m -> 240 x 120 px
    IX, IY, IS = 300, 12, 12.0  # over the crowd, clear of the scoreboard

    def ins(x, y):
        return int(IX + 10 + x * IS), int(IY + 10 + y * IS)
    byf = {f: g for f, g in d.groupby('frame')}
    cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', VIDEO], stdin=subprocess.PIPE)
    for f in range(F0, F1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        ov = im.copy()
        cv2.rectangle(ov, (IX, IY), (IX + 16 * int(IS) + 20, IY + 8 * int(IS) + 20), (30, 30, 30), -1)
        im = cv2.addWeighted(ov, 0.55, im, 0.45, 0)
        cv2.rectangle(im, ins(0, 0), ins(16, 8), (230, 200, 120), 1)
        cv2.line(im, ins(8, -0.4), ins(8, 8.4), (255, 255, 255), 2)
        for r in (byf[f].itertuples() if f in byf else []):
            if not np.isfinite([r.face_x_s, r.face_y_s, r.Rx, r.Lx, r.angle_to_net_deg]).all():
                continue
            c = COL.get(r.player, (200, 200, 200))
            fvec = np.array([r.face_x_s, r.face_y_s, 0.0])
            for sx, sy in ((r.Rx, r.Ry), (r.Lx, r.Ly)):
                A = np.array([sx, sy, -r.z]); B = A + 1.6 * fvec
                a2, b2 = (cam.project(np.array([A, B])) * S).astype(int)
                cv2.arrowedLine(im, tuple(a2), tuple(b2), (0, 0, 0), 5, cv2.LINE_AA, tipLength=0.15)
                cv2.arrowedLine(im, tuple(a2), tuple(b2), c, 2, cv2.LINE_AA, tipLength=0.15)
            mid = (cam.project(np.array([[(r.Rx + r.Lx) / 2, (r.Ry + r.Ly) / 2, -r.z - 0.45]]))[0] * S).astype(int)
            txt = f'{abs(r.angle_to_net_deg):.0f}'
            cv2.putText(im, txt, (mid[0] - 12, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, txt, (mid[0] - 12, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            p0 = ins(r.x, r.y); p1 = ins(r.x + 1.3 * r.face_x_s, r.y + 1.3 * r.face_y_s)
            cv2.circle(im, p0, 4, c, -1); cv2.arrowedLine(im, p0, p1, c, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(im, f'frame {f}  ({f / 25:.1f} s)   number = angle of the shoulders to the net (0 = square, 90 = side-on, 180 = back)',
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    print('video:', VIDEO)
