"""Build the chest-touch marking page (an artifact with a shared database).

For each labelled chest touch (data/contacts/touch_labels_final.csv, v3 frames -> v2): a strip of 9 frames
(t-4 .. t+4) cropped 320 x 320 px around the player's chest at full resolution, plus the crop origin so a click
maps back to full-image pixels. Writes OUT/strips/t<id>.jpg and OUT/touches.json.
usage: python scripts/experiments/chest_marks/make_tool.py OUT_DIR
"""
import os, sys, json
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
OUT = sys.argv[1]
os.makedirs(f'{OUT}/strips', exist_ok=True)
CROP, HALF = 320, 4

fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
tl = tl[(tl.kind == 'player') & (tl.body_part == 'chest')]
pb = pd.read_csv('data/boxes/player_boxes_v9.csv', usecols=['frame', 'player', 'x1', 'y1', 'x2', 'y2']).set_index(['frame', 'player'])
bb = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
out = []
for r in tl.itertuples():
    fr = int(fm[int(r.frame)])
    try:
        x1, y1, x2, y2 = pb.loc[(fr, r.player)].values.ravel()[:4]
        cx, cy = (x1 + x2) / 2, y1 + 0.35 * (y2 - y1)
    except KeyError:
        cx, cy = bb.center_x[fr], bb.center_y[fr]
    x0 = int(np.clip(cx - CROP / 2, 0, 1920 - CROP)); y0 = int(np.clip(cy - CROP / 2, 0, 1080 - CROP))
    tiles, balls = [], []
    for k in range(-HALF, HALF + 1):
        f = fr + k
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        tiles.append(im[y0:y0 + CROP, x0:x0 + CROP] if ok else np.zeros((CROP, CROP, 3), np.uint8))
        bx, by = bb.center_x.get(f, np.nan), bb.center_y.get(f, np.nan)
        balls.append(None if np.isnan(bx) else [round(float(bx - x0), 1), round(float(by - y0), 1)])
    cv2.imwrite(f'{OUT}/strips/t{int(r.touch)}.jpg', np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
    out.append(dict(id=int(r.touch), v2=fr, player=r.player, rally=int(r.rally), inplay=bool(r.in_play),
                    x0=x0, y0=y0, balls=balls))
json.dump(dict(crop=CROP, half=HALF, touches=out), open(f'{OUT}/touches.json', 'w'), separators=(',', ':'))
print(len(out), 'touches')
