"""Hold-out test of the marked-anchor estimate: anchors from the even touch ids, scored on the odd ones (and
the reverse). Score = flight-fit height at the marked frame minus the marked geometric height (result.csv).
usage: python scripts/experiments/chest_marks/holdout.py NAME CAMERA STATURES"""
import os, sys, json, glob, subprocess
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
name, cam, st = sys.argv[1:4]
geo = pd.read_csv('data/experiments/chest_marks/result.csv')
ms = {m['touch']: m for m in (json.load(open(f)) for f in glob.glob('data/experiments/chest_marks/marks/marks/*.json'))}
res = []
for use, test in [('even', 1), ('odd', 0)]:
    out = f'data/experiments/chest_marks/run_{name}_{use}'
    env = dict(os.environ, OUT=out, CAM=cam, STATURES=st, MARK_USE=use)
    subprocess.run([sys.executable, '-W', 'ignore', f'{HERE}/estimate_marked.py'], env=env, check=True, capture_output=True)
    h = pd.read_csv(f'{out}/ball_height_v2.csv').set_index('frame')
    for r in geo[(geo.feet == 'both') & (geo.touch % 2 == test)].itertuples():
        f = ms[r.touch]['frame']
        if h.method.get(f) == 'flight_fit' and r.calibration.startswith(sys.argv[4] if len(sys.argv) > 4 else ''):
            res.append(dict(touch=r.touch, err=h.height_m[f] - r.ball_h))
d = pd.DataFrame(res)
print(f'{name}: held-out marked touches n={len(d)}, fit - marked height: median {d.err.median():+.2f} m, MAE {d.err.abs().median():.2f} m')
