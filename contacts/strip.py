"""Frame strip around one touch (crop around the ball, skeletons drawn) for checking by eye.
usage: python contacts/strip.py FRAME OUT.jpg [HALF_WIDTH_FRAMES] [STEP]"""
import os, sys
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'contacts'))
from contact_sheet_lib import J, E, COL
f0, out = int(sys.argv[1]), sys.argv[2]
hw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
frames = list(range(f0 - hw * step, f0 + hw * step + 1, step))
pose = pd.read_csv(os.path.join(ROOT, 'pose', 'vitpose_body25.csv'), skiprows=lambda i: i > 0 and False)
pose = pose[pose.frame.isin(frames)]
ball = pd.read_csv(os.path.join(ROOT, 'ball', 'ball_boxes.csv'))
b0 = ball.iloc[f0]
cx = b0.center_x if not np.isnan(b0.center_x) else ball.iloc[frames].center_x.mean()
cy = b0.center_y if not np.isnan(b0.center_y) else ball.iloc[frames].center_y.mean()
x0, y0 = int(np.clip(cx - 200, 0, 1920 - 400)), int(np.clip(cy - 200, 0, 1080 - 400))
cap = cv2.VideoCapture(os.path.join(ROOT, 'main_camera_clean_v2.mp4'))
tiles = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    raw = im.copy()
    for p in pose[pose.frame == f].itertuples():
        q = np.array([[getattr(p, j + '_x'), getattr(p, j + '_y'), getattr(p, j + '_s')] for j in J])
        for a, b in E:
            if min(q[a, 2], q[b, 2]) >= .3:
                cv2.line(im, tuple(q[a, :2].astype(int)), tuple(q[b, :2].astype(int)), COL[p.player], 1)
    b = ball.iloc[f]
    if b.source != 'missing':
        cv2.circle(im, (int(b.center_x), int(b.center_y)), 18, (0, 0, 255), 1)
    t = np.hstack([raw[y0:y0 + 400, x0:x0 + 400], im[y0:y0 + 400, x0:x0 + 400]])
    cv2.putText(t, str(f), (5, 30), 0, 1, (255, 255, 255), 2)
    tiles.append(t)
rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * max(0, 3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows))
