"""Held-out errors split by whether the flight had >= 3 ball shadows: odd sand landings (height - 0.11 m) and odd
chest marks (fit height - marked height, from score_csv.py's score.csv in the same folder).
usage: python scripts/experiments/shadow/score_by_shadow.py RUN_DIR [SHADOWS_CSV]"""
import os, sys, json, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
run = sys.argv[1]
sh = sys.argv[2] if len(sys.argv) > 2 else 'data/experiments/shadow/shadows_p3_hand.csv'
h = pd.read_csv(f'{run}/chain.csv').set_index('frame')
nsh = pd.Series(pd.read_csv(sh).frame.values).map(h.flight_id).value_counts()
ms = {m['touch']: m for m in (json.load(open(f)) for f in glob.glob('data/experiments/chest_marks/marks/marks/*.json'))}
rows = []
for r in pd.read_csv(f'{run}/score.csv').itertuples():
    f = int(ms[r.touch]['frame'])
    rows.append(('chest', r.err, nsh.get(h.flight_id[f], 0)))
for fn in glob.glob('data/experiments/sand_marks/marks/landings/*.json'):
    m = json.load(open(fn))
    if m.get('landing') is True and m['frame'] % 2 == 1 and h.method.get(m['frame']) == 'flight_fit':
        rows.append(('sand', h.height_m[m['frame']] - 0.11, nsh.get(h.flight_id[m['frame']], 0)))
d = pd.DataFrame(rows, columns=['kind', 'err', 'nsh'])
d['flight'] = np.where(d.nsh >= 3, 'with shadows', 'few shadows')
print(d.groupby(['kind', 'flight']).err.agg(n='size', median='median', mae=lambda x: x.abs().median(), rms=lambda x: np.sqrt((x ** 2).mean())).round(3).to_string())
