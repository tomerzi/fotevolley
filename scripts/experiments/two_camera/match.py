"""Find which main-camera frame a replay screenshot shows: for each candidate frame, calibrate the
replay camera with that frame's player ground positions and compare the fit residuals."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_multi as cm
from calib_cam2 import main

pos = pd.read_csv(os.path.join(cm.ROOT, 'data', 'analysis', 'player_positions_v3.csv'))


def ground(frame, player):
    r = pos[(pos.v2_frame == frame) & (pos.player == player)].iloc[0]
    if np.isnan(r.u):
        return [r.x * 16 / 18, r.y * 8 / 9]
    return main.ground(np.array([[r.u, r.v]]))[0][:2].tolist()


def build(base, frame, roles):
    s = dict(base); s['players'] = []
    for p in base['players_px']:
        q = dict(p); q['player'] = roles[p['role']]; q['ground'] = ground(frame, q['player'])
        s['players'].append(q)
    s['frames'] = [frame]
    return s


if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    cm.SHARED = False
    out = []
    for frame, roles in spec['candidates']:
        s = build(spec['shot'], frame, roles)
        sol, cams = cm.fit([s])
        R, C, f = cams[0]
        r = np.array(cm.shot_res(R, C, f, s))
        P, e = cm.tri(R, C, f, s, frame)
        out.append((frame, np.sqrt(np.mean(r ** 2)), C.round(2).tolist(), round(f), round(e, 1)))
        print('frame %d  rms %.2f  C=%s f=%d  ball reproj %.1f px' % out[-1])
