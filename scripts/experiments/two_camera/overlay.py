"""Draw the calibrated replay camera's court/net projection and the ball positions on the screenshot,
next to the main-camera frame of the same moment."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_cam2 import project, D, ROOT
import pandas as pd

res = json.load(open(os.path.join(D, 'result.json')))
p = np.array(res['cam2'])
im = cv2.imread(os.path.join(D, 'replay_cam2.png'))
S = 3
big = cv2.resize(im, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)

def seg(A, B, col, n=40):
    P = np.linspace(A, B, n)
    uv = project(p, P) * S
    cv2.polylines(big, [uv.astype(np.int32)], False, col, 2, cv2.LINE_AA)

for y in (0, 8):
    seg([0, y, 0], [16, y, 0], (255, 255, 0))
for x in (0, 16):
    seg([x, 0, 0], [x, 8, 0], (255, 255, 0))
seg([8, -0.5, -2.2], [8, 8.5, -2.2], (0, 255, 255))
h = pd.read_csv(os.path.join(ROOT, 'data', 'analysis', 'ball_height_v2.csv')).set_index('frame')
r = h.loc[9638]
site = np.array([r.court_x_m, r.court_y_m, -r.height_m])
tri = np.array(res['9638'])
for P, col, lab in ((site, (0, 0, 255), 'site (main camera only)'), (tri, (0, 200, 0), 'two cameras')):
    uv = project(p, P)[0] * S
    cv2.circle(big, tuple(uv.astype(int)), 22, col, 3, cv2.LINE_AA)
    print(lab, 'projects at', (uv / S).round(1), 'observed ball', res.get('ball', None))
cv2.imwrite(os.path.join(D, 'replay_overlay.jpg'), big, [cv2.IMWRITE_JPEG_QUALITY, 90])
cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4'))
cap.set(1, 9638); ok, fr = cap.read()
cv2.imwrite(os.path.join(D, 'main_9638.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
