"""Per-frame height uncertainty (1 sigma, m) by how the frame was measured, from the held-out checks
(data/experiments/shadow/README.md, pass-3 run): flights with >= 3 shadows: sand landings MAE 0.056 / rms 0.073 m ->
0.08; flights with fewer shadows: chest MAE 0.25, landings rms 0.32 -> 0.25; at a touch outside a flight (player's
depth): 0.10 (estimate: the player's position error); single-frame shadow triangulation: 0.15 (one check only);
ball size only: 0.40 (chest touches MAE 0.28).
usage: python scripts/experiments/shadow/uncertainty.py HEIGHT_CSV SHADOWS_CSV OUT_CSV"""
import sys
import numpy as np, pandas as pd
csv, sh, out = sys.argv[1:4]
h = pd.read_csv(csv)
nsh = pd.Series(pd.read_csv(sh).frame.values).map(h.set_index('frame').flight_id).value_counts()
n = h.flight_id.map(nsh).fillna(0)
sig = np.select([(h.method == 'flight_fit') & (n >= 3), h.method == 'flight_fit', h.method == 'touch_depth', h.method == 'shadow_tri', h.method == 'size_only'],
                [0.08, 0.25, 0.10, 0.15, 0.40], np.nan)
h['height_sigma_m'] = sig
h['flight_shadows'] = n.astype(int)
h.drop(columns=[c for c in ['height_sigma_raw'] if c in h], inplace=True)
h.to_csv(out, index=False)
print(h.groupby('method').height_sigma_m.agg(['count', 'median']).round(2).to_string())
