"""Draw BODY_25 skeletons (pose/vitpose_body25.csv), player labels and the ball (ball/ball_boxes.csv if present).
Frames listed in pose/suspects.csv get a red warning line (skeleton weak / player missing / ball missing).
usage: python scripts/pose/render_pose.py START END out.mp4 [MIN_SCORE]
       python scripts/pose/render_pose.py --suspects out.mp4      (only flagged frames, edge cases excluded)"""
import os, sys
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sp = os.path.join(ROOT, 'data', 'pose', 'suspects.csv')
sus = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame(columns=['frame', 'kind', 'who', 'detail'])
LABEL = {'pose_weak': 'weak skeleton', 'pose_edge': 'partly off-screen', 'player_none': 'no box',
         'ball_hidden': 'BALL MISSING'}
warn = {f: '  |  '.join(f"{r.who} {LABEL[r.kind]}" if r.kind != 'ball_hidden' else LABEL[r.kind]
                        for r in g.itertuples()) for f, g in sus.groupby('frame')}
if sys.argv[1] == '--suspects':
    out_path = sys.argv[2]; MIN_S = 0.3
    frame_list = sorted(sus[~sus.kind.isin(['pose_edge'])].frame.unique())
else:
    start, end, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    MIN_S = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    frame_list = range(start, end)
J = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist', 'mid_hip', 'r_hip',
     'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'l_eye', 'r_ear', 'l_ear', 'l_big_toe',
     'l_small_toe', 'l_heel', 'r_big_toe', 'r_small_toe', 'r_heel']
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12),
         (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22),
         (22, 23), (11, 24)]
COL = {'BRA_A': (0, 255, 255), 'BRA_B': (0, 200, 0), 'ISR_A': (255, 160, 0), 'ISR_B': (255, 0, 160)}

pose = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'vitpose_body25.csv'))
fs = set(frame_list)
pose = {f: g for f, g in pose[pose.frame.isin(fs)].groupby('frame')}
bp = os.path.join(ROOT, 'data', 'ball_pipeline', 'ball_boxes.csv')
ball = pd.read_csv(bp).set_index('frame') if os.path.exists(bp) else None

cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4'))
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (1920, 1080))
prev = -2
for f in frame_list:
    if f != prev + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    prev = f
    ok, im = cap.read()
    if not ok:
        break
    for _, r in pose.get(f, pd.DataFrame()).iterrows():
        c = COL.get(r.player, (255, 255, 255))
        p = np.array([[r[f'{j}_x'], r[f'{j}_y'], r[f'{j}_s']] for j in J])
        for a, b in EDGES:
            if min(p[a, 2], p[b, 2]) >= MIN_S:
                cv2.line(im, tuple(p[a, :2].astype(int)), tuple(p[b, :2].astype(int)), c, 2, cv2.LINE_AA)
        for q in p:
            if q[2] >= MIN_S:
                cv2.circle(im, tuple(q[:2].astype(int)), 3, (0, 0, 255), -1)
        top = p[p[:, 2] >= MIN_S]
        if len(top):
            cv2.putText(im, r.player, (int(top[:, 0].min()), int(top[:, 1].min()) - 8), 0, 0.7, c, 2)
    if ball is not None and f in ball.index and ball.loc[f, 'source'] != 'missing':
        b = ball.loc[f]
        cv2.rectangle(im, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                      (0, 255, 0) if b.source == 'detected' else (0, 165, 255), 2)
    cv2.putText(im, str(f), (20, 1060), 0, 1, (255, 255, 255), 2)
    if f in warn:
        cv2.rectangle(im, (0, 1000), (1920, 1080), (0, 0, 160), -1)
        cv2.putText(im, f'{f}  CHECK: {warn[f]}', (20, 1055), 0, 1.3, (255, 255, 255), 3)
    vw.write(im)
vw.release()
print('wrote', out_path)
