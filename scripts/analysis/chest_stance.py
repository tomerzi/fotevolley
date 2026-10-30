"""Distance between the feet at every chest touch (v3 video).

For each chest touch in data/contacts/touch_labels_final.csv: each foot's ground point = mean of heel, big toe and
small toe (Sapiens2, score >= S_MIN; ankle if none), median over frames t-1..t+1, pixels -> court units with the
homography. Reported: stance (distance between the feet), its parts along the net (y) and towards the net (x), and
its parts across / along the body (hip line on the court: side-to-side vs one foot in front of the other).
Each foot gets a status from frames t-PLANT_DT..t+PLANT_DT: 'planted' (moves less than PLANT_MOVE box heights),
'raised' (moves and is more than LIFT_FRAC leg lengths above the other foot in the image) or 'moving' (a step or a
slide). One camera cannot tell a raised foot from a foot set further back, hence the motion test. 'jump' copies
data/analysis/player_positions_v3.csv at the touch frame. reliable = no raised foot and no jump.

Lengths are court units (court lines = 18 x 9); x 8/9 for metres if the court is 16 x 8 (docs/ANALYSIS_PLAN.md).

inputs : data/contacts/touch_labels_final.csv, data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv,
         data/analysis/player_positions_v3.csv, data/mappings/homography_px_to_m.npy
output : data/analysis/chest_stance_v3.csv
usage: python scripts/analysis/chest_stance.py [SHEET.jpg]   (optional contact sheet of 12 touches to check by eye)"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
S_MIN = 0.3
LIFT_FRAC = 0.5      # as in player_positions.py
DT = 1               # frames on each side of the touch for the stance
PLANT_DT = 2         # frames on each side of the touch for the foot status
PLANT_MOVE = 0.06    # box heights: a foot moving less than this around the touch stands on the sand
FOOT = ('heel', 'big_toe', 'small_toe')
H = np.load(os.path.join(ROOT, 'data', 'mappings', 'homography_px_to_m.npy'))


def to_court(p):
    return cv2.perspectiveTransform(np.asarray(p, float).reshape(1, -1, 2), H)[0]


def foot_px(r, side):
    """ground point of one foot in pixels (NaN if not seen) and the ankle height used for the lift test."""
    pts = [(r[f'{side}_{k}_x'], r[f'{side}_{k}_y']) for k in FOOT if r[f'{side}_{k}_s'] >= S_MIN]
    if not pts and r[f'{side}_ankle_s'] >= S_MIN:
        pts = [(r[f'{side}_ankle_x'], r[f'{side}_ankle_y'])]
    return np.mean(pts, 0) if pts else np.full(2, np.nan)


def main():
    t = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'touch_labels_final.csv'))
    t = t[(t.kind == 'player') & (t.body_part == 'chest')]
    kp = [f'{s}_{k}' for s in ('left', 'right') for k in FOOT + ('ankle', 'hip')]
    cols = ['frame', 'player', 'y1', 'y2'] + [f'{k}_{a}' for k in kp for a in 'xys']
    frames = set(f for x in t.frame for f in range(x - PLANT_DT, x + PLANT_DT + 1))
    pose = pd.concat(c[c.frame.isin(frames)] for c in pd.read_csv(
        os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'), usecols=cols,
        chunksize=100000)).set_index(['frame', 'player'])
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'), usecols=['frame', 'player', 'jump', 'posture'])
    pos = pos.set_index(['frame', 'player'])
    rows = []
    for r in t.itertuples():
        fr = [f for f in range(r.frame - PLANT_DT, r.frame + PLANT_DT + 1) if (f, r.player) in pose.index]
        P = [pose.loc[(f, r.player)] for f in fr]
        feet = {s: np.array([foot_px(p, s) for p in P]).reshape(-1, 2) for s in ('left', 'right')}
        near = np.array([abs(f - r.frame) <= DT for f in fr], bool)
        row = dict(touch=r.touch, frame=r.frame, time_sec=r.time_sec, rally=r.rally, in_play=r.in_play,
                   player=r.player, frames_used=0)
        L, R = feet['left'][near], feet['right'][near]
        ok = np.isfinite(L).all(1) & np.isfinite(R).all(1)
        if ok.any():
            lc, rc = to_court(L[ok]), to_court(R[ok])
            lm, rm = np.median(lc, 0), np.median(rc, 0)
            d = rm - lm
            row.update(frames_used=int(ok.sum()), left_x=lm[0], left_y=lm[1], right_x=rm[0], right_y=rm[1],
                       stance=np.hypot(*d), towards_net=abs(d[0]), along_net=abs(d[1]))
            hips = [[[p.left_hip_x, p.left_hip_y], [p.right_hip_x, p.right_hip_y]] for p, n in zip(P, near)
                    if n and min(p.left_hip_s, p.right_hip_s) >= S_MIN]
            if hips:
                hc = to_court(np.median(np.array(hips), 0))            # hip line, as if on the sand: a direction only
                ax = (hc[1] - hc[0]) / np.linalg.norm(hc[1] - hc[0])
                row.update(side_to_side=abs(d @ ax), front_back=abs(d[0] * ax[1] - d[1] * ax[0]))
        boxh = np.median([p.y2 - p.y1 for p in P]) if P else np.nan
        legs = [np.hypot(p[f'{s}_hip_x'] - p[f'{s}_ankle_x'], p[f'{s}_hip_y'] - p[f'{s}_ankle_y'])
                for p in P for s in ('left', 'right') if min(p[f'{s}_hip_s'], p[f'{s}_ankle_s']) >= S_MIN]
        leg = np.median(legs) if legs else 0.45 * boxh
        ymed = {s: np.nanmedian(feet[s][near, 1]) if np.isfinite(feet[s][near, 1]).any() else np.nan
                for s in feet}
        for s, o in (('left', 'right'), ('right', 'left')):
            a = feet[s][np.isfinite(feet[s]).all(1)]
            move = np.max(np.hypot(*(a - np.median(a, 0)).T)) / boxh if len(a) >= 3 else np.nan
            raised = ymed[o] - ymed[s] > LIFT_FRAC * leg                  # higher in the image than the other foot
            row[f'{s}_move'] = move
            row[f'{s}_foot'] = 'unseen' if not np.isfinite(move) else 'planted' if move < PLANT_MOVE else \
                'raised' if raised else 'moving'
        st = pos.loc[(r.frame, r.player)] if (r.frame, r.player) in pos.index else None
        row.update(jump=bool(st.jump) if st is not None else False, posture=st.posture if st is not None else 'none')
        row['reliable'] = row['frames_used'] > 0 and 'raised' not in (row['left_foot'], row['right_foot']) \
            and not row['jump']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(OUT, 'chest_stance_v3.csv'), index=False)
    good = df[df.in_play & df.reliable]
    print(f'{len(df)} chest touches, {len(good)} in play and reliable;', 'feet:',
          pd.concat([df.left_foot, df.right_foot]).value_counts().to_dict())
    print(good[['stance', 'along_net', 'towards_net', 'side_to_side', 'front_back']].describe().round(2).to_string())
    print(good.groupby('player').stance.agg(['count', 'median', 'mean']).round(2).to_string())
    if len(sys.argv) > 1:
        sheet(df, sys.argv[1])


def sheet(df, path, n=12):
    """crops around the player at the touch frame, feet points drawn, stance written."""
    Hi = np.linalg.inv(H)
    boxes = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'player_boxes_v3video.csv')).set_index(['frame', 'player'])
    cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v3.mp4'))
    tiles = []
    for r in df[df.frames_used > 0].sample(min(n, len(df)), random_state=1).itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
        ok, im = cap.read()
        b = boxes.loc[(r.frame, r.player)]
        for (x, y), col in (((r.left_x, r.left_y), (0, 255, 0)), ((r.right_x, r.right_y), (0, 0, 255))):
            p = cv2.perspectiveTransform(np.array([[[x, y]]], float), Hi)[0][0]
            cv2.circle(im, (int(p[0]), int(p[1])), 5, col, -1)
        cx, cy, hh = int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2), int(max(b.y2 - b.y1, 120))
        y0, x0 = max(cy - hh, 0), max(cx - hh, 0)
        c = cv2.resize(im[y0:cy + hh, x0:cx + hh], (300, 300))
        cv2.putText(c, f'{r.frame} {r.player} {r.stance:.2f} {r.left_foot[0]}{r.right_foot[0]}', (4, 18), 0, 0.55,
                    (255, 255, 255), 2)
        tiles.append(c)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == '__main__':
    main()
