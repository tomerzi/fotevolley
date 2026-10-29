"""Build the point-moment marking page: one short clip per rally (the last seconds before the point ended),
frame number burned in, plus the automatic moment from scripts/analysis/conceded_distance.py for reference.
usage: python scripts/experiments/point_marks/make_tool.py OUT_DIR
"""
import os, sys, json, subprocess
import pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
OUT = sys.argv[1]
os.makedirs(f'{OUT}/clips', exist_ok=True)
r = pd.read_csv('data/points/rallies_v2.csv').set_index('rally')
auto = pd.read_csv('data/analysis/conceded_partner_distance.csv').set_index('rally')
cap = cv2.VideoCapture('video/main_camera_clean_v2.mp4')
W, H = 854, 480
pts = []
for R, q in r.iterrows():
    a = auto.loc[R]
    s = int(max(q.start_frame, min(q.end_frame - 125, a.point_frame - 75)))
    e = int(min(q.end_frame + 25, s + 300))
    path = f'{OUT}/clips/p{R:02d}.mp4'
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '25', '-i', '-',
                           '-c:v', 'libx264', '-preset', 'medium', '-crf', '29', '-g', '5', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                          stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    for f in range(s, e + 1):
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (W - 140, H - 32), (W, H), (0, 0, 0), -1)
        cv2.putText(im, f'{f}', (W - 132, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        ff.stdin.write(im.tobytes())
    ff.stdin.close(); ff.wait()
    pts.append(dict(rally=int(R), set=int(q.set), start=s, end=e, loser='ISR' if q.winner == 'BRA' else 'BRA',
                    score=f'{q.ISR_score_after}-{q.BRA_score_after}', auto=int(a.point_frame)))
json.dump(dict(fps=25, points=pts), open(f'{OUT}/points.json', 'w'), separators=(',', ':'))
print(len(pts), 'clips')
