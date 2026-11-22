"""Where each player's eyes, shoulders, knees and feet point, relative to the net, drawn as thin laser-like rays.

Each body point (ViTPose BODY_25, data/pose/vitpose_body25.csv) is placed in 3-D by intersecting its camera ray with
the horizontal plane at that point's height (a share of the player's measured stature, player_statures.json):
  eyes      gaze = from the midpoint of the ears to the nose (eyes if an ear is missing), at 0.93 x stature;
            one ray from each eye
  shoulders perpendicular to the shoulder line (forwards by the pose's left/right labels), at 0.82 x stature;
            one ray from each shoulder
  knees     perpendicular to the line between the knees, at 0.28 x stature; one ray from each knee
  feet      each foot's own direction, heel -> big toe, on the sand; one ray from each foot
Heights assume an upright player, so crouched or airborne players are less exact. Keypoints below 0.3 confidence and
implausible spans are dropped; directions are smoothed over 5 frames.
Angle to the net: 0 = square to the net, 90 = side-on, 180 = back to it; + = turned towards the near (camera) side.
Writes data/analysis/body_direction_<full_match|first_30s|start_end>.csv and, with --video, an overlay clip (fixed-length rays,
1.4 m, fading out, one colour per body part, and a top-down court inset).
usage: python scripts/analysis/body_rays.py [START END] [--video OUT.mp4]
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
MIN_S = 0.3
PARTS = ['eyes', 'shoulders', 'knees', 'feet']
PCOL = {'eyes': (255, 235, 90), 'shoulders': (60, 215, 255), 'knees': (230, 110, 255), 'feet': (110, 255, 140)}  # BGR
HEB = {'eyes': 'eyes', 'shoulders': 'shoulders', 'knees': 'knees', 'feet': 'feet'}
KP = ['nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear', 'r_shoulder', 'l_shoulder', 'r_knee', 'l_knee', 'r_big_toe', 'l_big_toe', 'r_heel', 'l_heel']
pose = pd.read_csv('data/pose/vitpose_body25.csv', usecols=['frame', 'player'] + [f'{k}_{c}' for k in KP for c in 'xys'])
pose = pose[(pose.frame >= F0) & (pose.frame <= F1)].drop_duplicates(['frame', 'player'])
pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'x', 'y'])
pos = pos[(pos.v2_frame >= F0) & (pos.v2_frame <= F1)].drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])


def on_plane(uv, z):
    d = cam.ray(np.asarray(uv, float).reshape(-1, 2))
    return cam.C + ((-z - cam.C[2]) / d[:, 2])[:, None] * d


def kp(r, k):
    return (getattr(r, f'{k}_x'), getattr(r, f'{k}_y')) if getattr(r, f'{k}_s') >= MIN_S else None


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


rows = []
for r in pose.itertuples():
    if r.player not in STAT:
        continue
    st = STAT[r.player]
    try:
        px, py = pos.loc[(r.frame, r.player)].values.ravel()[:2]
    except KeyError:
        continue
    res = {}
    # eyes: ears' midpoint -> nose
    z = 0.93 * st
    nose = kp(r, 'nose'); re_, le_ = kp(r, 'r_ear'), kp(r, 'l_ear'); ry_, ly_ = kp(r, 'r_eye'), kp(r, 'l_eye')
    back = [re_, le_] if re_ and le_ else ([ry_, ly_] if ry_ and ly_ else None)
    if nose and back and ry_ and ly_:
        B = on_plane(back, z).mean(axis=0); N = on_plane([nose], z)[0]
        f = unit((N - B)[:2])
        if f is not None and 0.03 <= np.linalg.norm((N - B)[:2]) <= 0.35:
            res['eyes'] = (f, on_plane([ry_, ly_], z))
    for part, (a, b), h, span in (('shoulders', ('r_shoulder', 'l_shoulder'), 0.82, (0.22, 0.60)),
                                 ('knees', ('r_knee', 'l_knee'), 0.28, (0.08, 0.70))):
        A, Bk = kp(r, a), kp(r, b)
        if A and Bk:
            P = on_plane([A, Bk], h * st)
            right = (P[0] - P[1])[:2]; w = np.linalg.norm(right)
            if span[0] <= w <= span[1]:
                right /= w
                res[part] = (np.array([right[1], -right[0]]), P)
    feet_dirs, feet_pts = [], []
    for s in 'rl':
        T, Hh = kp(r, f'{s}_big_toe'), kp(r, f'{s}_heel')
        if T and Hh:
            P = on_plane([Hh, T], 0.03)
            v = (P[1] - P[0])[:2]; L = np.linalg.norm(v)
            if 0.10 <= L <= 0.45:
                feet_dirs.append(v / L); feet_pts.append(P[1])
    if feet_dirs:
        res['feet'] = (np.array(feet_dirs), np.array(feet_pts))
    for part, (f, P) in res.items():
        dirs = f if part == 'feet' else np.array([f, f])
        for k, (dv, Pk) in enumerate(zip(dirs, P)):
            rows.append(dict(frame=r.frame, player=r.player, part=part, k=k, x=px, y=py, fx=dv[0], fy=dv[1],
                             ox=Pk[0], oy=Pk[1], oz=-Pk[2]))
d = pd.DataFrame(rows)
# smooth each ray's direction over 5 frames
out = []
for key, g in d.groupby(['player', 'part', 'k']):
    g = g.set_index('frame').reindex(range(F0, F1 + 1))
    fx = g.fx.rolling(5, center=True, min_periods=2).median(); fy = g.fy.rolling(5, center=True, min_periods=2).median()
    n = np.hypot(fx, fy); g['fx'], g['fy'] = fx / n, fy / n
    g['player'], g['part'], g['k'] = key
    out.append(g.reset_index().dropna(subset=['ox', 'fx']))
d = pd.concat(out)
tn = np.where(d.x < 8, 1.0, -1.0)
a = np.degrees(np.arctan2(tn * d.fy, tn * d.fx))
d['angle_to_net_deg'] = np.where(d.x < 8, a, -a).round(1)
d['facing_deg'] = np.degrees(np.arctan2(d.fy, d.fx)).round(1)
NAME = 'full_match' if (F0, F1) == (0, 15585) else 'first_30s' if (F0, F1) == (0, 749) else f'{F0}_{F1}'
d.round(3).to_csv(f'data/analysis/body_direction_{NAME}.csv', index=False)
per = d.groupby(['player', 'part', 'frame']).angle_to_net_deg.mean().abs().reset_index()
print(per.pivot_table(index='player', columns='part', values='angle_to_net_deg', aggfunc='median').round(0)[PARTS].to_string())

if VIDEO:
    W, H, S = 960, 540, 0.5
    IX, IY, IS = 300, 12, 12.0
    LEN = 1.4
    LX = 130  # legend, clear of the bottom-left logo

    def ins(x, y):
        return int(IX + 10 + x * IS), int(IY + 10 + y * IS)
    byf = {f: g for f, g in d.groupby('frame')}
    cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', VIDEO], stdin=subprocess.PIPE)
    NSEG = 12
    for f in range(F0, F1 + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        ov = im.copy()
        cv2.rectangle(ov, (IX, IY), (IX + 16 * int(IS) + 20, IY + 8 * int(IS) + 20), (25, 25, 25), -1)
        cv2.rectangle(ov, (LX - 8, H - 64), (LX + 372, H - 26), (25, 25, 25), -1)
        im = cv2.addWeighted(ov, 0.55, im, 0.45, 0)
        cv2.rectangle(im, ins(0, 0), ins(16, 8), (230, 200, 120), 1)
        cv2.line(im, ins(8, -0.4), ins(8, 8.4), (255, 255, 255), 1)
        glow = np.zeros_like(im, dtype=np.float32); core = np.zeros_like(im, dtype=np.float32)
        for r in (byf[f].itertuples() if f in byf else []):
            col = np.array(PCOL[r.part], np.float32)
            A = np.array([r.ox, r.oy, -r.oz]); fv = np.array([r.fx, r.fy, 0.0])
            P = (cam.project(A + np.linspace(0, LEN, NSEG + 1)[:, None] * fv) * S)
            for i in range(NSEG):  # fading beam
                fade = 1.0 - i / NSEG
                p0, p1 = tuple(np.round(P[i]).astype(int)), tuple(np.round(P[i + 1]).astype(int))
                cv2.line(glow, p0, p1, (col * 0.55 * fade).tolist(), 5, cv2.LINE_AA)
                cv2.line(core, p0, p1, (col * fade).tolist(), 1, cv2.LINE_AA)
            o = tuple(np.round(P[0]).astype(int))
            cv2.circle(core, o, 2, col.tolist(), -1, cv2.LINE_AA)
            # inset: short faint lines per part from the player's spot
            q0 = ins(r.x, r.y); q1 = ins(r.x + 0.9 * r.fx, r.y + 0.9 * r.fy)
            cv2.line(core, q0, q1, (col * 0.9).tolist(), 1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 2.5)
        im = np.clip(im.astype(np.float32) + glow * 0.9 + core, 0, 255).astype(np.uint8)
        for r in (byf[f].drop_duplicates('player').itertuples() if f in byf else []):
            cv2.circle(im, ins(r.x, r.y), 3, (240, 240, 240), -1)
        x0 = LX
        for p in PARTS:
            cv2.line(im, (x0, H - 49), (x0 + 18, H - 49), PCOL[p], 2, cv2.LINE_AA)
            cv2.putText(im, HEB[p], (x0 + 24, H - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            x0 += 24 + cv2.getTextSize(HEB[p], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 22
        cv2.putText(im, f'frame {f}  ({f / 25:.1f} s)', (LX, H - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    print('video:', VIDEO)
