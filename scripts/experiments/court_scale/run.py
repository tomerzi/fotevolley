"""EXPERIMENT: which size of the blue lines makes the video physically consistent?

For each COURT_SCALE s the blue lines are taken as (16 x 8) * s m. The camera is calibrated (lines, net top
2.20 m, Israeli players 1.86 m), the ball heights are re-estimated with touch / sand anchors (gravity fixed at
9.81 m/s^2) and scored:
  * g_free: gravity fitted freely on clean flights with an apex (should be 9.81),
  * anchor dh / dxy: how far the fitted ball is from the touching body part (should be ~0),
  * chest / head: median ball height at the labelled chest / head touches,
  * net and neck errors of the calibration (px).
Writes data/experiments/court_scale/scan.csv (one row per scale) and keeps every run in data/experiments/court_scale/s<scale>/.
usage: python scripts/experiments/court_scale/run.py [scale ...]
"""
import os, sys, re, json, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path.insert(0, HERE)
BASE = 'data/experiments/court_scale'
scales = [float(a) for a in sys.argv[1:]] or [1.0, 1.0625, 1.125, 1.1875]


def free_g(cam_path):
    import importlib, traj
    importlib.reload(traj)
    cam = traj.Camera(cam_path)
    b = pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
    ok = b.source.isin(['detected', 'rescued', 'manual']).values
    u, T = b[['center_x', 'center_y']].values, b.time_sec.values
    gs = []
    for s in range(0, len(b) - 20, 3):
        idx = np.arange(s, s + 20)
        if not ok[idx].all():
            continue
        vy = np.diff(u[idx, 1])
        if not (vy[:6].mean() < -3 and vy[-6:].mean() > 3):  # apex inside the window
            continue
        x, r = traj.fit(cam, T[idx], u[idx], free_g=True)
        if np.percentile(r, 90) < 2.0:
            gs.append(x[6])
    return float(np.median(gs)), len(gs)


fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
tl['v2'] = tl.frame.map(fm)
rows = []
for sc in scales:
    out = f'{BASE}/s{sc:.4f}'
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, COURT_SCALE=str(sc), OUT=out)
    cal = subprocess.run([sys.executable, f'{HERE}/calibrate.py'], env=env, capture_output=True, text=True).stdout
    open(f'{out}/calibrate.log', 'w').write(cal)
    subprocess.run([sys.executable, '-W', 'ignore', f'{HERE}/estimate.py'], env=env, capture_output=True, text=True)
    net = re.search(r'model y=(\d+)px', cal)
    necks = [float(v) for v in re.findall(r'ISR_\w: \d+ poses, neck dy median (-?[\d.]+)px', cal)]
    cam_h = float(re.search(r'height=([\d.]+) m', cal).group(1))
    pp = re.search(r'pp=\((\d+),(\d+)\)', cal).groups()
    g, ng = free_g(f'{out}/camera.npz')
    a = pd.read_csv(f'{out}/anchors.csv')
    h = pd.read_csv(f'{out}/ball_height_v2.csv')
    m = tl.merge(h, left_on='v2', right_on='frame')
    m = m[m.method == 'flight_fit']
    st = json.load(open(f'{out}/player_statures.json'))
    rows.append(dict(scale=sc, lines_m=f'{16 * sc:.2f} x {8 * sc:.2f}', g_free=round(g, 2), g_windows=ng,
                     anchor_dh=round(float((a.fit_h - a.target_h).median()), 2),
                     anchor_dxy=round(float(a.dxy.median()), 2),
                     chest_h=round(float(m[m.body_part == 'chest'].height_m.median()), 2),
                     head_h=round(float(m[m.body_part == 'head'].height_m.median()), 2),
                     net_err_px=int(net.group(1)) - 622 if net else None,
                     isr_neck_px=round(float(np.mean(necks)), 1), camera_h=cam_h, pp_y=int(pp[1]),
                     bra_a=st['BRA_A']['stature_m'], bra_b=st['BRA_B']['stature_m']))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{BASE}/scan.csv', index=False)
