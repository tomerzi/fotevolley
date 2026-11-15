"""Independent check of the vertical scale with the net (known height 2.20 m).

The top tape of the net throws a straight shadow line on the sand.  With the sun slope a (shadow shift along the court
per metre of height, estimated ONLY from ball/shadow pairs) the tape height is  z = (x_shadow - x_net) / a.
Nothing in the pipeline was tuned to the net height, so recovering ~2.20 m validates camera + sun + scale together.
usage: python validate_net.py"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera
from data import load_ball
from shadow_candidates import read_gray
HERE = os.path.dirname(os.path.abspath(__file__))


def net_shadow_lines(cam, bg):
    """ground x of the dark lines left of the net in the median image, for several image rows"""
    out = []
    for v in range(440, 761, 20):
        prof = bg[v - 2:v + 3, 820:1000].mean(0)
        base = np.median(prof)
        idx = [i for i in range(3, len(prof) - 3) if prof[i] < base - 25 and prof[i] == prof[i - 3:i + 4].min()]
        xs = []
        for i in idx:
            # sub-pixel minimum
            w = np.clip(base - prof[i - 3:i + 4], 0, None); u = 820 + i + (w * np.arange(-3, 4)).sum() / w.sum()
            xs.append(cam.to_plane(np.array([[u, v]], float), 0.0)[0][0])
        if len(xs) >= 2:
            out.append((v, min(xs), max(xs)))
    return np.array(out)


if __name__ == '__main__':
    cam = Camera.from_json(os.path.join(HERE, 'camera.json'))
    b = load_ball()
    sun = pd.read_csv(os.path.join(HERE, 'sun_per_shot.csv')).set_index('shot')
    rows = []
    for shot, g in b.groupby('shot'):
        if len(g) < 60 or shot not in sun.index or sun.loc[shot, 'n'] < 40:
            continue
        f0 = int(g.frame.min()); n = len(g)
        fr = read_gray(f0 + n // 2 - 20, 40)
        bg = np.median(fr[::2], axis=0)
        L = net_shadow_lines(cam, bg)
        if len(L) < 6:
            continue
        xt, xb = np.median(L[:, 1]), np.median(L[:, 2])
        rows.append(dict(shot=shot, a=sun.loc[shot, 'a_fit'], x_top=xt, x_bottom=xb, straight_cm=100 * L[:, 1].std(), n_rows=len(L)))
    d = pd.DataFrame(rows)
    # x_shadow = x_net + a * z  -> regression over the shots (the sun moves during the match)
    A = np.c_[np.ones(len(d)), d.a]
    (xnet, ztop), *_ = np.linalg.lstsq(A, d.x_top.values, rcond=None)
    (xnet_b, zbot), *_ = np.linalg.lstsq(A, d.x_bottom.values, rcond=None)
    # with the net plane taken from the image of the edge-on net (x = 7.90)
    d['z_top_m'] = (d.x_top - 7.90) / d.a
    d['z_bottom_m'] = (d.x_bottom - 7.90) / d.a
    pd.set_option('display.width', 200)
    print(d.round(3).to_string())
    print('\nnet top tape height from its shadow (net plane x=7.90 from the image): median %.3f m, min %.3f, max %.3f  (true: 2.20 m)' % (
        d.z_top_m.median(), d.z_top_m.min(), d.z_top_m.max()))
    print('net bottom edge: median %.3f m  -> net is %.2f m tall' % (d.z_bottom_m.median(), d.z_top_m.median() - d.z_bottom_m.median()))
    print('free regression over shots (no assumption on the net plane): x_net = %.3f, top = %.3f m ; x_net = %.3f, bottom = %.3f m' % (xnet, ztop, xnet_b, zbot))
    print('shadow line straightness (std of ground x along the line): median %.1f cm' % d.straight_cm.median())
    d.round(4).to_csv(os.path.join(HERE, 'validation_net_height.csv'), index=False)
