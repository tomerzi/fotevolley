"""Contact sheet of ball crops. usage: python scripts/ball/sheet.py MODE out.jpg [N]   MODE: newdet | changed | interp"""
import os, sys, cv2, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mode, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
new = pd.read_csv(os.path.join(ROOT, 'data', 'ball_pipeline', 'ball_boxes.csv'))
old = pd.read_csv(os.path.join(ROOT, 'data', 'tracks', 'ball_tracks_full.csv')).iloc[:len(new)]
dist = np.hypot(new.center_x - old.center_x, new.center_y - old.center_y)
sel = {'newdet': (new.source == 'detected') & (old.source == 'missing'),
       'changed': (new.source == 'detected') & (old.source == 'yolo') & (dist > 20),
       'interp': (new.source == 'interpolated'),
       'tracked': (new.source == 'tracked'), 'rescued': (new.source == 'rescued'),
       'lost': (new.source == 'missing') & (old.source == 'yolo')}[mode]
fr = new.frame[sel].values
fr = fr[np.linspace(0, len(fr) - 1, min(N, len(fr))).astype(int)] if len(fr) else fr
cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4')); tiles = []
for f in fr:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    r = new.iloc[f]; o = old.iloc[f]
    if r.source == 'missing': r = o.copy(); r['confidence'] = o.confidence
    im = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
    if o.source != 'missing': cv2.circle(im, (int(o.center_x) + 100, int(o.center_y) + 100), 24, (255, 0, 255), 2)
    cx, cy = int(r.center_x) + 100, int(r.center_y) + 100
    t = im[cy - 100:cy + 100, cx - 100:cx + 100].copy()
    cv2.rectangle(t, (100 - 22, 100 - 22), (100 + 22, 100 + 22), (0, 255, 0), 1)
    cv2.putText(t, f'{f} {r.confidence if r.confidence == r.confidence else 0:.2f}', (3, 15), 0, 0.5, (0, 255, 255), 1)
    tiles.append(t)
while len(tiles) % 6: tiles.append(np.zeros((200, 200, 3), np.uint8))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
print(sel.sum(), 'frames;', len(fr), 'shown')
