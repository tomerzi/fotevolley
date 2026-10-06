"""Low-conf ball re-detection on every frame with both ball models -> ball/candidates.csv (resumable).
usage: python ball/detect_ball.py [LIMIT]"""
import os, sys, csv, time
import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'main_camera_clean_v2.mp4')
OUT = os.path.join(ROOT, 'ball', 'candidates.csv')
MODELS = {'exp2': 'exp-2.pt', 'bd': 'ballDedect.pt'}
IMGSZ, CONF, BATCH = 1280, 0.05, 16
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

done = -1
if os.path.exists(OUT):
    with open(OUT) as f:
        lines = f.read().splitlines()
    good = [l for l in lines[1:] if l.count(',') == 7]
    if good:
        done = int(good[-1].split(',')[0])
        good = [l for l in good if int(l.split(',')[0]) < done]  # redo last frame fully
        done -= 1
    with open(OUT, 'w') as f:
        f.write('frame,model,x1,y1,x2,y2,conf,cy\n')
        f.writelines(l + '\n' for l in good)
else:
    with open(OUT, 'w') as f:
        f.write('frame,model,x1,y1,x2,y2,conf,cy\n')

models = {k: YOLO(os.path.join(ROOT, 'ball', v)) for k, v in MODELS.items()}
cap = cv2.VideoCapture(VIDEO)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end = min(n, limit) if limit else n
start = done + 1
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
f = open(OUT, 'a', newline='')
w = csv.writer(f)
t0 = time.time(); fi = start
while fi < end:
    frames, idx = [], []
    while len(frames) < BATCH and fi < end:
        ok, im = cap.read()
        if not ok:
            fi = end; break
        frames.append(im); idx.append(fi); fi += 1
    if not frames:
        break
    for name, m in models.items():
        res = m.predict(frames, imgsz=IMGSZ, conf=CONF, half=True, verbose=False, max_det=10)
        for i, r in zip(idx, res):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                w.writerow([i, name, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(c, 4), round((y1 + y2) / 2, 1)])
    f.flush()
    if idx[0] % 800 < BATCH:
        el = time.time() - t0
        print(f'{idx[-1]}/{end} {(idx[-1]-start+1)/el:.1f} fps', flush=True)
f.close()
print('done', time.time() - t0)
