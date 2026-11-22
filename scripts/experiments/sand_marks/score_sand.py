"""Score a per-frame ball height file on the hand-marked sand landings (ball centre should be 0.11 m).
usage: python scripts/experiments/sand_marks/score_sand.py BALL_HEIGHT_CSV [even|odd|all]   (parity of the frame)"""
import os, sys, json, glob
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
csv, parity = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'all')
ms = [json.load(open(f)) for f in glob.glob('data/experiments/sand_marks/marks/landings/*.json')]
fr = [m['frame'] for m in ms if m.get('landing') is True and (parity == 'all' or (m['frame'] % 2 == 0) == (parity == 'even'))]
h = pd.read_csv(csv).set_index('frame')
s = h.loc[fr]; s = s[s.method == 'flight_fit']
print(f'sand landings ({parity}) n={len(s)}: height median {s.height_m.median():.2f} m (0.11 expected), MAE {(s.height_m - 0.11).abs().median():.2f} m')
