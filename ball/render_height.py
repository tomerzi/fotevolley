"""Draw the ball height on the video: box, height label, the vertical from the ball's ground point up to the ball,
and a height-over-time strip. Computes heights for the range with ball/ball_height.py.
usage: python ball/render_height.py START END out.mp4"""
import os, sys, subprocess
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib import Camera, load
from ball_height import compute

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COL = {'flight': (0, 255, 0), 'size': (0, 200, 255)}
start, end, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

cam = Camera(*load())
h = compute(start, end).set_index('frame')
b = pd.read_csv(os.path.join(ROOT, 'correct_bbox', 'ball_boxes_corrected3.csv')).set_index('frame')

SW, SH, HMAX = 1920, 220, 8.0       # strip size, metres at the top of the strip
fr = np.arange(start, end)
xs = ((fr - start) / max(1, len(fr) - 1) * (SW - 80) + 60).astype(int)
ys = lambda m: int(SH - 20 - m / HMAX * (SH - 40))

cap = cv2.VideoCapture(os.path.join(ROOT, 'main_camera_clean_v2.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
tmp = out_path + '.tmp.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720 + 147))
for k, f in enumerate(fr):
    ok, im = cap.read()
    if not ok:
        break
    r = h.loc[f]
    if r.method != 'missing':
        bb = b.loc[f]
        col = COL[r.method]
        g = tuple(int(v) for v in cam.project([r.X, r.Y, 0])[0])
        top = tuple(int(v) for v in cam.project([r.X, r.Y, -r.height_m])[0])
        cv2.line(im, g, top, (255, 255, 255), 2)
        cv2.ellipse(im, g, (14, 5), 0, 0, 360, (255, 255, 255), 2)
        cv2.rectangle(im, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), col, 2)
        cv2.putText(im, f'{r.height_m:.2f} m', (int(bb.x2) + 8, int(bb.y1) + 10), 0, 1.0, (0, 0, 0), 5)
        cv2.putText(im, f'{r.height_m:.2f} m', (int(bb.x2) + 8, int(bb.y1) + 10), 0, 1.0, col, 2)
    cv2.putText(im, f'frame {f}', (1650, 1060), 0, 0.9, (255, 255, 255), 2)
    # strip
    s = np.full((SH, SW, 3), 30, np.uint8)
    for m in range(0, int(HMAX) + 1, 2):
        cv2.line(s, (60, ys(m)), (SW - 20, ys(m)), (80, 80, 80), 1)
        cv2.putText(s, f'{m}m', (5, ys(m) + 5), 0, 0.6, (200, 200, 200), 1)
    cv2.line(s, (60, ys(2.2)), (SW - 20, ys(2.2)), (0, 0, 200), 1)
    cv2.putText(s, 'net', (SW - 60, ys(2.2) - 5), 0, 0.5, (0, 0, 220), 1)
    for kk in range(len(fr)):
        rr = h.loc[fr[kk]]
        if rr.method != 'missing':
            cv2.circle(s, (xs[kk], ys(min(rr.height_m, HMAX))), 3, COL[rr.method], -1)
    cv2.line(s, (xs[k], 0), (xs[k], SH), (255, 255, 255), 1)
    full = np.vstack([cv2.resize(im, (1280, 720)), cv2.resize(s, (1280, 147))])
    vw.write(full)
vw.release()
import imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '23', out_path], check=True)
os.remove(tmp)
print('->', out_path)
