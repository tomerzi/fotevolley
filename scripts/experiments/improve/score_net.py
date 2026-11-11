"""Score a per-frame ball height file on the net: where a flight crosses the net plane (X = 8 m) the ball centre must
be at least 2.31 m up (net top 2.20 m + one radius). Crossings within 6 frames of a labelled net touch are skipped.
usage: python scripts/experiments/improve/score_net.py BALL_HEIGHT_CSV [even|odd|all]   (parity of the frame)"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(ROOT)
csv, parity = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'all')
h = pd.read_csv(csv)
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv'); fm = fm[~fm.deleted].set_index('v3_frame').v2_frame
tl = pd.read_csv('data/contacts/touch_labels_final.csv')
nt = [int(fm[f]) for f in tl[tl.kind == 'net'].frame if f in fm.index]
hs = []
for fid, g in h[(h.flight_id >= 0) & (h.method == 'flight_fit')].groupby('flight_id'):
    x = g.court_x_m.values - 8.0
    for k in np.flatnonzero(np.sign(x[:-1]) != np.sign(x[1:])):
        w = abs(x[k]) / (abs(x[k]) + abs(x[k + 1]) + 1e-9)
        f = int(g.frame.values[k] if w < 0.5 else g.frame.values[k + 1])
        if (parity != 'all' and (f % 2 == 0) != (parity == 'even')) or any(abs(f - t) <= 6 for t in nt):
            continue
        hs.append(g.height_m.values[k] * (1 - w) + g.height_m.values[k + 1] * w)
hs = np.array(hs)
low = hs < 2.31
print(f'net crossings ({parity}) n={len(hs)}: height median {np.median(hs):.2f} m, below the net top {low.mean() * 100:.0f}% '
      f'(by {np.median(2.31 - hs[low]) if low.any() else 0:.2f} m median)')
