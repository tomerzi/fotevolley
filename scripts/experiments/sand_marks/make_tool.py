"""Build the sand-landing marking page (artifact with a shared database).

Candidates: ball events with no player and the ball on the sand (data/contacts/ball_contacts_v3.csv) plus the
'sand' touch labels, v3 frames -> v2. For each: a strip of 13 frames (t-6 .. t+6), 320 x 320 px at full
resolution around the ball at the event frame, and the automatic ball position per frame.
Writes OUT/strips/s<v2 frame>.jpg and OUT/landings.json.
usage: python scripts/experiments/sand_marks/make_tool.py OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
OUT = sys.argv[1]
os.makedirs(f'{OUT}/strips', exist_ok=True)
CROP, HALF = 320, 6

fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
bc = pd.read_csv('data/contacts/ball_contacts_v3.csv')
ev = bc[(bc.event_type != 'touch') & (bc.ball_on_sand == True)][['frame', 'rally', 'in_play']]
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
ev = pd.concat([ev, tl[tl.kind == 'sand'][['frame', 'rally', 'in_play']]]).drop_duplicates('frame').sort_values('frame')
bb = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
out = []
for r in ev.itertuples():
    if r.frame not in fm.index:
        continue
    fr = int(fm[int(r.frame)])
    near = bb.iloc[max(0, fr - 3): fr + 4].dropna(subset=['center_x'])
    if not len(near):
        continue
    cx, cy = near.center_x.median(), near.center_y.median()
    x0 = int(np.clip(cx - CROP / 2, 0, 1920 - CROP)); y0 = int(np.clip(cy - CROP / 2, 0, 1080 - CROP))
    tiles, balls = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr - HALF)
    for k in range(-HALF, HALF + 1):
        ok, im = cap.read()
        f = fr + k
        tiles.append(im[y0:y0 + CROP, x0:x0 + CROP] if ok else np.zeros((CROP, CROP, 3), np.uint8))
        bx, by = (bb.center_x.iloc[f], bb.center_y.iloc[f]) if 0 <= f < len(bb) else (np.nan, np.nan)
        balls.append(None if np.isnan(bx) else [round(float(bx - x0), 1), round(float(by - y0), 1)])
    cv2.imwrite(f'{OUT}/strips/s{fr}.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    out.append(dict(id=fr, v2=fr, rally=int(r.rally), inplay=bool(r.in_play), x0=x0, y0=y0, balls=balls))
json.dump(dict(crop=CROP, half=HALF, landings=out), open(f'{OUT}/landings.json', 'w'), separators=(',', ':'))
print(len(out), 'landings')
