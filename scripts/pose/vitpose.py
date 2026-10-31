"""ViTPose-L COCO-WholeBody (HF transformers, converted by pose/convert_wholebody.py) on every identified player box
in data/tracks/player_boxes.csv, reported as the 25 OpenPose BODY_25 joints:
17 COCO body + 6 foot points from the whole-body head, Neck = mid-shoulders, MidHip = mid-hips (score = min of the pair).
-> pose/vitpose_body25.csv : frame,player,team,box_source + <joint>_x,<joint>_y,<joint>_s (pixels; resumable)
usage: python scripts/pose/vitpose.py [MODEL] [LIMIT_FRAMES]"""
import os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO = os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4')
MODEL = (sys.argv[1] if len(sys.argv) > 1 else '') or os.path.join(ROOT, 'models', 'vitpose_weights', 'vitpose-l-wholebody-hf')
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = os.path.join(ROOT, 'data', 'pose', os.environ.get('POSE_OUT', 'vitpose_body25.csv'))
FRAMES_PER_BATCH = 16
# BODY_25 joint -> COCO-WholeBody index (or a pair to average)
BODY25 = [('nose', 0), ('neck', (5, 6)), ('r_shoulder', 6), ('r_elbow', 8), ('r_wrist', 10),
          ('l_shoulder', 5), ('l_elbow', 7), ('l_wrist', 9), ('mid_hip', (11, 12)), ('r_hip', 12),
          ('r_knee', 14), ('r_ankle', 16), ('l_hip', 11), ('l_knee', 13), ('l_ankle', 15),
          ('r_eye', 2), ('l_eye', 1), ('r_ear', 4), ('l_ear', 3), ('l_big_toe', 17), ('l_small_toe', 18),
          ('l_heel', 19), ('r_big_toe', 20), ('r_small_toe', 21), ('r_heel', 22)]
KP = [k for k, _ in BODY25]


def to_body25(kp, sc):
    out = []
    for _, src in BODY25:
        if isinstance(src, tuple):
            a, b = src
            out.append([*((kp[a] + kp[b]) / 2), min(sc[a], sc[b])])
        else:
            out.append([*kp[src], sc[src]])
    return np.round(np.array(out), 3)


COLS = ['frame', 'player', 'team', 'box_source'] + [f'{k}_{a}' for k in KP for a in 'xys']

dev = 'cuda'
proc = AutoProcessor.from_pretrained(MODEL)
model = VitPoseForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
plus = 'plus' in MODEL   # ViTPose+ is multi-dataset: expert 0 = COCO

boxes = pd.read_csv(os.path.join(ROOT, 'data', 'tracks', 'player_boxes.csv'))
by_frame = {f: g for f, g in boxes.groupby('frame')}
n = int(cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FRAME_COUNT))
end = min(n, LIMIT) if LIMIT else n

start = 0
if os.path.exists(OUT):
    old = pd.read_csv(OUT)
    if len(old):
        last = old.frame.max()
        old[old.frame < last].to_csv(OUT, index=False)   # redo last (maybe partial) frame
        start = int(last)
if start == 0:
    pd.DataFrame(columns=COLS).to_csv(OUT, index=False)

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
t0 = time.time(); fi = start; ncrops = 0
while fi < end:
    imgs, metas, bxs = [], [], []
    while len(imgs) < FRAMES_PER_BATCH and fi < end:
        ok, im = cap.read()
        if not ok:
            fi = end; break
        g = by_frame.get(fi)
        if g is not None and len(g):
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            b = g[['x1', 'y1', 'x2', 'y2']].values.astype(float)
            bxs.append(np.c_[b[:, :2], b[:, 2:] - b[:, :2]])          # COCO xywh
            metas.append(g[['frame', 'player', 'team', 'source']].values)
        fi += 1
    if not imgs:
        continue
    inputs = proc(imgs, boxes=bxs, return_tensors='pt').to(dev)
    inputs['pixel_values'] = inputs['pixel_values'].half()
    nb = inputs['pixel_values'].shape[0]
    kw = {'dataset_index': torch.zeros(nb, dtype=torch.long, device=dev)} if plus else {}
    with torch.no_grad():
        out = model(**inputs, **kw)
    out.heatmaps = out.heatmaps.float()
    res = proc.post_process_pose_estimation(out, boxes=bxs)
    rows = []
    for meta, per_img in zip(metas, res):
        for m, p in zip(meta, per_img):
            kp = p['keypoints'].cpu().numpy(); sc = p['scores'].cpu().numpy()
            rows.append(list(m) + list(to_body25(kp, sc).ravel()))
    ncrops += len(rows)
    pd.DataFrame(rows, columns=COLS).to_csv(OUT, mode='a', header=False, index=False)
    if (fi // FRAMES_PER_BATCH) % 50 == 0:
        el = time.time() - t0
        print(f'{fi}/{end}  {(fi - start) / el:.1f} frames/s  {ncrops / el:.0f} crops/s', flush=True)
print('done', round(time.time() - t0), 's', ncrops, 'crops')
