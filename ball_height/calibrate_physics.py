"""Learn the flight physics of this match from the flights whose 3-D path is pinned by the shadow:
the mean unmodelled acceleration (wind / spin / residual scale).  It becomes the prior for flights without shadow.
usage: python calibrate_physics.py   -> physics.json"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain as ch
import reconstruct as rc
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    ch.AE_MEAN = np.zeros(3)
    res, fl, ct, outs = rc.main(write=False, verbose=False)
    r = fl[(fl.rich) & (fl.n_obs >= 14) & (fl.state == 'flight')]
    ae = r[['ae_x', 'ae_y', 'ae_z']]
    med = ae.median().values
    sd = 1.48 * (ae - med).abs().median().values
    dur = (r.frame_end - r.frame_start) / 25.0
    vx = (r.x1 - r.x0) / dur
    print('rich flights:', len(r), ' median extra acceleration', np.round(med, 3), ' robust sd', np.round(sd, 3))
    print('ae_x by vx:', r.groupby(pd.cut(vx, [-20, -8, -3, 0, 3, 8, 20])).ae_x.median().round(2).to_dict())
    json.dump(dict(kd=ch.KD_DEFAULT, ae_mean=[float(v) for v in med], ae_sd=[float(v) for v in sd], n_flights=int(len(r)),
                   g_effective=float(9.81 - med[2])), open(os.path.join(HERE, 'physics.json'), 'w'), indent=1)
