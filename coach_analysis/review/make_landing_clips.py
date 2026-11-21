"""Review clips: for each point, 3 s before to 1.5 s after the frame used as the ball landing
(data/analysis/point_landings.csv). Red circle = ball detection in that frame (data/boxes/ball_boxes_corrected3.csv),
yellow cross = computed landing spot on the sand (court x, y -> pixels with the homography), shown all the time.
Writes coach_analysis/review/clips/point_XX.mp4 (960x540 H.264) and clips_index.csv (pixel gap at the landing frame)."""
import subprocess, numpy as np, pandas as pd, cv2
BEFORE, AFTER = 75, 38
d = pd.read_csv('data/analysis/point_landings.csv')
b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv').set_index('frame')
Hi = np.linalg.inv(np.load('data/mappings/homography_px_to_m.npy'))
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4'); N = int(cap.get(7))
SRC = {'hand-marked landing': 'landing frame: HAND-CONFIRMED', 'ball track': 'landing frame: AUTOMATIC (not checked)',
       'rally end': 'NO LANDING FOUND (rally end)'}
rows = []
for q in d.itertuples():
    f0, f1 = max(0, q.frame - BEFORE), min(N - 1, q.frame + AFTER)
    cross = None
    if not pd.isna(q.x):
        g = cv2.perspectiveTransform(np.array([[[q.x, q.y]]], float), Hi)[0, 0]; cross = (int(g[0]), int(g[1]))
    out = f'coach_analysis/review/clips/point_{q.rally:02d}.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '960x540', '-r', '25',
                           '-i', '-', '-c:v', 'libx264', '-crf', '28', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    cap.set(1, f0)
    gap = None
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        ball = None
        if f in b.index and not pd.isna(b.at[f, 'center_x']):
            ball = (int(b.at[f, 'center_x']), int(b.at[f, 'center_y']))
            cv2.circle(img, ball, 24, (0, 0, 255), 3)
        if cross:
            cv2.drawMarker(img, cross, (0, 255, 255), cv2.MARKER_CROSS, 34, 4)
        land = f == q.frame
        if land:
            cv2.rectangle(img, (0, 0), (1919, 1079), (0, 255, 255), 14)
            if ball and cross:
                gap = int(np.hypot(ball[0] - cross[0], ball[1] - cross[1]))
        t = (f - q.frame) / 25
        txt = f'point {q.rally}  frame {f}  t={t:+.2f}s  ' + ('<<< LANDING FRAME >>>' if land else '')
        cv2.rectangle(img, (0, 1000), (1920, 1080), (0, 0, 0), -1)
        cv2.putText(img, txt, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255) if land else (255, 255, 255), 3)
        cv2.putText(img, SRC[q.source], (1100, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        frame = cv2.resize(img, (960, 540))
        reps = 12 if land else 1          # hold the landing frame ~0.5 s
        for _ in range(reps):
            ff.stdin.write(frame.tobytes())
    ff.stdin.close(); ff.wait()
    rows.append(dict(rally=q.rally, clip=out.split('/')[-1], gap_px=gap))
pd.DataFrame(rows).to_csv('coach_analysis/review/clips_index.csv', index=False)
print(pd.DataFrame(rows).gap_px.describe())
