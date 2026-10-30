"""Contact sheets of the middle frame of each suspect range, with skeletons + ball drawn.
usage: python scripts/pose/suspect_sheet.py KIND OUT_PREFIX [PER_SHEET]"""
import os, sys, cv2, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv += [] ; kind, prefix = sys.argv[1], sys.argv[2]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 12
J = ['nose','neck','r_shoulder','r_elbow','r_wrist','l_shoulder','l_elbow','l_wrist','mid_hip','r_hip','r_knee','r_ankle','l_hip','l_knee','l_ankle','r_eye','l_eye','r_ear','l_ear','l_big_toe','l_small_toe','l_heel','r_big_toe','r_small_toe','r_heel']
E = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),(0,15),(0,16),(15,17),(16,18),(14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]
COL = {'BRA_A': (0,255,255), 'BRA_B': (0,200,0), 'ISR_A': (255,160,0), 'ISR_B': (255,0,160)}
pose = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'vitpose_body25.csv'))
boxes = pd.read_csv(os.path.join(ROOT, 'data', 'tracks', 'player_boxes.csv'))
ball = pd.read_csv(os.path.join(ROOT, 'data', 'ball_pipeline', 'ball_boxes.csv'))
rng = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'suspect_ranges.csv')); rng = rng[rng.kind == kind]
cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4')); tiles = []
for r in rng.itertuples():
    f = (r.start + r.end) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    for p in pose[pose.frame == f].itertuples():
        c = COL[p.player]; q = np.array([[getattr(p, j+'_x'), getattr(p, j+'_y'), getattr(p, j+'_s')] for j in J])
        for a, b in E:
            if min(q[a,2], q[b,2]) >= .3: cv2.line(im, tuple(q[a,:2].astype(int)), tuple(q[b,:2].astype(int)), c, 3)
    for bx in boxes[boxes.frame == f].itertuples():
        thick = 4 if bx.player == r.who else 1
        cv2.rectangle(im, (bx.x1, bx.y1), (bx.x2, bx.y2), COL[bx.player], thick); cv2.putText(im, bx.player, (bx.x1, bx.y1-6), 0, 1, COL[bx.player], 2)
    b = ball.iloc[f]
    if b.source != 'missing': cv2.rectangle(im, (int(b.x1)-6, int(b.y1)-6), (int(b.x2)+6, int(b.y2)+6), (0,0,255), 3)
    t = f'{f} [{r.start}-{r.end}] {r.who}'
    cv2.putText(im, t, (20, 1050), 0, 1.6, (0,0,0), 8); cv2.putText(im, t, (20, 1050), 0, 1.6, (255,255,255), 3)
    tiles.append(cv2.resize(im, (640, 360)))
for k in range(0, len(tiles), per):
    t = tiles[k:k+per]; t += [np.zeros_like(t[0])] * (-len(t) % 3)
    cv2.imwrite(f'{prefix}_{k//per}.jpg', np.vstack([np.hstack(t[i:i+3]) for i in range(0, len(t), 3)]))
print(len(tiles), 'tiles')
