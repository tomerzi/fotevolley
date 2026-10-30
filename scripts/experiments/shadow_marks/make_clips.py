"""Shadow-marking page, version 2: a larger view and a few seconds of video around each frame.
For each item of OUT/shadows.json (make_tool.py): a 640 x 360 crop at full resolution around the ball and its
predicted shadow, frames t-40..t+40 (3.2 s), frame numbers burned in, 960 x 540 H.264.
Writes OUT/clips/c<frame>.mp4 and OUT/clips.json.
usage: python scripts/experiments/shadow_marks/make_clips.py HEIGHT_CSV OUT_DIR
"""
import os, sys, json, subprocess
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
sys.path.insert(0, 'scripts/ball_height')
from traj import Camera

csv, OUT = sys.argv[1], sys.argv[2]
os.makedirs(f'{OUT}/clips', exist_ok=True)
CW, CH, HALF, S = 640, 360, 40, 1.5
cam = Camera()
h = pd.read_csv(csv).set_index('frame')
J = json.load(open(f'{OUT}/shadows.json'))
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
items = []
for it in J['items']:
    f = it['id']
    su, sv = it['x0'] + it['pred'][0], it['y0'] + it['pred'][1]  # predicted shadow, full-frame px
    r = h.loc[f]
    bu, bv = cam.project(np.array([[r.court_x_m, r.court_y_m, -r.height_m]]))[0]
    cu, cv_ = ((su + bu) / 2, (sv + bv) / 2) if abs(su - bu) < CW * 0.8 and abs(sv - bv) < CH * 0.8 else (su, sv)
    x0 = int(np.clip(cu - CW / 2, 0, 1920 - CW)); y0 = int(np.clip(cv_ - CH / 2, 0, 1080 - CH))
    s0 = max(0, f - HALF)
    path = f'{OUT}/clips/c{f}.mp4'
    W, H = int(CW * S), int(CH * S)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-g', '5', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                          stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    n = 0
    for k in range(s0, f + HALF + 1):
        ok, im = cap.read()
        if not ok:
            break
        c = cv2.resize(im[y0:y0 + CH, x0:x0 + CW], (W, H), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(c, (W - 120, H - 30), (W, H), (0, 0, 0), -1)
        cv2.putText(c, str(k), (W - 112, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ff.stdin.write(c.tobytes()); n += 1
    ff.stdin.close(); ff.wait()
    items.append(dict(id=f, kind=it['kind'], h=it['h'], x0=x0, y0=y0, cw=CW, ch=CH, start=s0, n=n,
                      pred=[round(su - x0, 1), round(sv - y0, 1)]))
json.dump(dict(fps=25, items=items), open(f'{OUT}/clips.json', 'w'), separators=(',', ':'))
print(len(items), 'clips')
