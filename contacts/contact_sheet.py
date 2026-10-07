"""Contact sheets for checking ball_contacts.csv by eye: a crop around the ball at each touch frame with all
skeletons, the ball (red) and the chosen player/part. usage: python contacts/contact_sheet.py OUT_PREFIX [START END]"""
import os, sys
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'contacts'))
from contact_sheet_lib import J, E, COL
prefix = sys.argv[1]
res = pd.read_csv(os.path.join(ROOT, 'contacts', 'ball_contacts.csv'))
if len(sys.argv) > 3:
    res = res[(res.frame >= int(sys.argv[2])) & (res.frame < int(sys.argv[3]))]
pose = pd.read_csv(os.path.join(ROOT, 'pose', 'vitpose_body25.csv'))
pose = pose[pose.frame.isin(set(res.frame))]
cap = cv2.VideoCapture(os.path.join(ROOT, 'main_camera_clean_v2.mp4'))
tiles = []
for r in res.itertuples():
    cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame); ok, im = cap.read()
    for p in pose[pose.frame == r.frame].itertuples():
        q = np.array([[getattr(p, j + '_x'), getattr(p, j + '_y'), getattr(p, j + '_s')] for j in J])
        for a, b in E:
            if min(q[a, 2], q[b, 2]) >= .3:
                cv2.line(im, tuple(q[a, :2].astype(int)), tuple(q[b, :2].astype(int)), COL[p.player], 2)
    cx, cy = int(r.ball_x), int(r.ball_y)
    cv2.circle(im, (cx, cy), 16, (0, 0, 255), 2)
    x0, y0 = int(np.clip(cx - 240, 0, 1920 - 480)), int(np.clip(cy - 180, 0, 1080 - 360))
    t = im[y0:y0 + 360, x0:x0 + 480].copy()
    lab = f'{r.frame} {r.player} {r.body_part} {r.confidence:.2f}'
    cv2.putText(t, lab, (5, 25), 0, 0.75, (0, 0, 0), 5); cv2.putText(t, lab, (5, 25), 0, 0.75, (255, 255, 255), 2)
    tiles.append(t)
per = 12
for k in range(0, len(tiles), per):
    t = tiles[k:k + per]; t += [np.zeros_like(tiles[0])] * (-len(t) % 4)
    cv2.imwrite(f'{prefix}_{k // per}.jpg', np.vstack([np.hstack(t[i:i + 4]) for i in range(0, len(t), 4)]))
print(len(tiles), 'tiles')
