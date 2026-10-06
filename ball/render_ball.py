"""Draw ball/ball_boxes.csv on the video.
usage: python ball/render_ball.py START END out.mp4          (frame range)
       python ball/render_ball.py --changed out.mp4           (only frames whose ball differs from ball_tracks_full.csv)"""
import os, sys
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {'detected': (0, 255, 0), 'interpolated': (0, 200, 255)}
OLD_COL = (255, 0, 255)

new = pd.read_csv(os.path.join(ROOT, 'ball', 'ball_boxes.csv'))
old = pd.read_csv(os.path.join(ROOT, 'ball_tracks_full.csv')).iloc[:len(new)]
if sys.argv[1] == '--changed':
    out_path = sys.argv[2]
    nd = new.source != 'missing'; od = old.source != 'missing'
    dist = np.hypot(new.center_x - old.center_x, new.center_y - old.center_y)
    frames = new.frame[(nd != od) | (nd & od & (dist > 20))].values
else:
    frames = np.arange(int(sys.argv[1]), int(sys.argv[2]))
    out_path = sys.argv[3]

cap = cv2.VideoCapture(os.path.join(ROOT, 'main_camera_clean_v2.mp4'))
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))
prev = -2
for f in frames:
    if f != prev + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read(); prev = f
    if not ok:
        break
    o = old.iloc[f]
    if o.source != 'missing':
        cv2.circle(im, (int(o.center_x), int(o.center_y)), 26, OLD_COL, 2)
    r = new.iloc[f]
    if r.source != 'missing':
        cv2.rectangle(im, (int(r.x1) - 4, int(r.y1) - 4), (int(r.x2) + 4, int(r.y2) + 4), COL[r.source], 3)
    label = f'{f} shot {r.shot} new:{r.source} {r.model if isinstance(r.model, str) else ""} {r.missing_reason if isinstance(r.missing_reason, str) else ""} | old:{o.source}'
    cv2.putText(im, label, (20, 1050), 0, 1.0, (0, 0, 0), 5); cv2.putText(im, label, (20, 1050), 0, 1.0, (255, 255, 255), 2)
    vw.write(cv2.resize(im, (1280, 720)))
vw.release()
print(len(frames), 'frames ->', out_path)
