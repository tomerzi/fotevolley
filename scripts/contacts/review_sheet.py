"""Contact sheets to check ball events by eye: for each event, crops of frames t-4, t, t+4 around the ball,
zoomed, with every player's Sapiens2 skeleton (chosen player thick), the ball path and the decision.
usage: python scripts/contacts/review_sheet.py OUT_PREFIX [--frames F1,F2,...] [--filter EXPR] [--per 12]
       EXPR is a pandas query on data/contacts/ball_contacts_v3.csv, e.g. "in_play and confidence < 0.6" """
import argparse
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COL = {'BRA_A': (0, 220, 255), 'BRA_B': (0, 140, 0), 'ISR_A': (255, 120, 0), 'ISR_B': (255, 0, 200)}
BONES = [('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
         ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
         ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
         ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
         ('left_ankle', 'left_big_toe'), ('right_ankle', 'right_big_toe'), ('left_ankle', 'left_heel'),
         ('right_ankle', 'right_heel'), ('neck', 'nose')]
JS = sorted({j for b in BONES for j in b})
CROP, Z = 260, 2          # crop half-size px, zoom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--frames', default='')
    ap.add_argument('--filter', default='')
    ap.add_argument('--per', type=int, default=12)
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'ball_contacts_v3.csv'))
    ev['event'] = range(len(ev))
    if a.frames:
        ev = ev[ev.frame.isin([int(x) for x in a.frames.split(',')])]
    if a.filter:
        ev = ev.query(a.filter)
    pose = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'),
                       usecols=['frame', 'player'] + [f'{j}_{c}' for j in JS for c in 'xys'])
    need = set()
    for f in ev.frame:
        need |= {f - 4, f, f + 4}
    pose = pose[pose.frame.isin(need)]
    pf = {f: g for f, g in pose.groupby('frame')}
    ball = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_v3video.csv')).set_index('frame')
    cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v3.mp4'))
    tiles = []
    for e in ev.itertuples():
        cx, cy = int(e.ball_x), int(e.ball_y)
        row = []
        for df in (-4, 0, 4):
            f = e.frame + df
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                im = np.zeros((1080, 1920, 3), np.uint8)
            for r in pf.get(f, pd.DataFrame()).itertuples():
                th = 3 if r.player == e.player else 1
                for j1, j2 in BONES:
                    if getattr(r, f'{j1}_s') > .3 and getattr(r, f'{j2}_s') > .3:
                        cv2.line(im, (int(getattr(r, f'{j1}_x')), int(getattr(r, f'{j1}_y'))),
                                 (int(getattr(r, f'{j2}_x')), int(getattr(r, f'{j2}_y'))), COL[r.player], th)
            for g in range(e.frame - 12, e.frame + 13):          # ball path, red before, cyan after
                if g in ball.index and not np.isnan(ball.at[g, 'center_x']):
                    cv2.circle(im, (int(ball.at[g, 'center_x']), int(ball.at[g, 'center_y'])), 2,
                               (0, 0, 255) if g < e.frame else (255, 255, 0), -1)
            if df == 0:
                cv2.circle(im, (cx, cy), int((e.ball_d if e.ball_d == e.ball_d else 27) / 2) + 3, (255, 255, 255), 1)
            x0, y0 = int(np.clip(cx - CROP, 0, 1920 - 2 * CROP)), int(np.clip(cy - CROP, 0, 1080 - 2 * CROP))
            c = cv2.resize(im[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP], None, fx=Z * 0.5, fy=Z * 0.5)
            cv2.putText(c, f'{f}', (5, 20), 0, 0.6, (255, 255, 255), 2)
            row.append(c)
        t = np.hstack(row)
        bar = np.zeros((44, t.shape[1], 3), np.uint8)
        lab = (f'#{e.event} f{e.frame} r{e.rally} {"PLAY" if e.in_play else "dead"} '
               f'{e.player if isinstance(e.player, str) else "-"} {e.body_part if isinstance(e.body_part, str) else ""} '
               f'{e.limb_side if isinstance(e.limb_side, str) else ""} c={e.confidence}')
        lab2 = (f'2nd {e.second_player if isinstance(e.second_player, str) else ""} '
                f'{e.second_part if isinstance(e.second_part, str) else ""} dv={e.velocity_change} '
                f'd={e.ball_d} depth={e.depth_ratio}')
        cv2.putText(bar, lab, (5, 18), 0, 0.55, (255, 255, 255), 1)
        cv2.putText(bar, lab2, (5, 38), 0, 0.5, (180, 180, 180), 1)
        tiles.append(np.vstack([bar, t]))
    for k in range(0, len(tiles), a.per):
        ch = tiles[k:k + a.per]
        cols = 2
        while len(ch) % cols:
            ch.append(np.zeros_like(ch[0]))
        g = np.vstack([np.hstack(ch[i:i + cols]) for i in range(0, len(ch), cols)])
        cv2.imwrite(f'{a.out}_{k // a.per:02d}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(len(tiles), 'tiles')


if __name__ == '__main__':
    main()
