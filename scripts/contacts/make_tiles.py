"""One labelled image per ball event (data/contacts/review/tiles/e<event>_f<frame>.jpg): a zoomed crop around the
ball at the touch frame, every skeleton thin, the chosen player's thick, the ball circled, its path drawn
(red = before, cyan = after), and the label burned in at the bottom.
usage: python scripts/contacts/make_tiles.py"""
import os
import cv2
import numpy as np
import pandas as pd
from review_sheet import BONES, COL, JS, ROOT

OUT = os.path.join(ROOT, 'data', 'contacts', 'review', 'tiles')
HALF, SIZE = 210, 420


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.read_csv(os.path.join(ROOT, 'data', 'contacts', 'ball_contacts_v3.csv'))
    ev['event'] = range(len(ev))
    pose = pd.read_csv(os.path.join(ROOT, 'data', 'pose', 'sapiens2', 'sapiens2_pose_v3_00000-13494.csv'),
                       usecols=['frame', 'player'] + [f'{j}_{c}' for j in JS for c in 'xys'])
    pf = {f: g for f, g in pose[pose.frame.isin(set(ev.frame))].groupby('frame')}
    ball = pd.read_csv(os.path.join(ROOT, 'data', 'boxes', 'ball_boxes_v3video.csv')).set_index('frame')
    cap = cv2.VideoCapture(os.path.join(ROOT, 'video', 'main_camera_clean_v3.mp4'))
    for e in ev.sort_values('frame').itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, e.frame)
        ok, im = cap.read()
        if not ok:
            continue
        who = e.player if isinstance(e.player, str) else ''
        for r in pf.get(e.frame, pd.DataFrame()).itertuples():
            th = 4 if r.player == who else 1
            for j1, j2 in BONES:
                if getattr(r, f'{j1}_s') > .3 and getattr(r, f'{j2}_s') > .3:
                    cv2.line(im, (int(getattr(r, f'{j1}_x')), int(getattr(r, f'{j1}_y'))),
                             (int(getattr(r, f'{j2}_x')), int(getattr(r, f'{j2}_y'))), COL[r.player], th)
        for g in range(e.frame - 10, e.frame + 11):
            if g in ball.index and not np.isnan(ball.at[g, 'center_x']):
                cv2.circle(im, (int(ball.at[g, 'center_x']), int(ball.at[g, 'center_y'])), 3,
                           (0, 0, 255) if g < e.frame else (255, 255, 0), -1)
        cx, cy = int(e.ball_x), int(e.ball_y)
        cv2.circle(im, (cx, cy), int((e.ball_d if e.ball_d == e.ball_d else 27) / 2) + 4, (255, 255, 255), 2)
        x0, y0 = int(np.clip(cx - HALF, 0, 1920 - 2 * HALF)), int(np.clip(cy - HALF, 0, 1080 - 2 * HALF))
        t = cv2.resize(im[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], (SIZE, SIZE))
        lab = f'f{e.frame}  ' + (f'{who} {e.body_part} {e.limb_side if isinstance(e.limb_side, str) else ""}'
                                 if who else 'no player')
        cv2.rectangle(t, (0, SIZE - 30), (SIZE, SIZE), (0, 0, 0), -1)
        cv2.putText(t, lab, (8, SIZE - 9), 0, 0.65, COL.get(who, (255, 255, 255)), 2)
        cv2.imwrite(os.path.join(OUT, f'e{e.event:03d}_f{e.frame:05d}.jpg'), t, [cv2.IMWRITE_JPEG_QUALITY, 72])
    print(len(os.listdir(OUT)), 'tiles in', OUT)


if __name__ == '__main__':
    main()
