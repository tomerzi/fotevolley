"""Diagnostic plots: ball height versus frame for every shot.  usage: python plots.py [out_dir]"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miniplot import Plot, COLORS
HERE = os.path.dirname(os.path.abspath(__file__))


def plot_shot(d, ct, path, title=''):
    f0, f1 = d.frame.min(), d.frame.max()
    zmax = max(5.0, np.nanmax(d.z_m.values) + 0.5 if np.isfinite(d.z_m).any() else 5.0)
    p = Plot(w=max(900, min(2600, 4 * (f1 - f0) + 200)), h=460, xlim=(f0 - 1, f1 + 1), ylim=(0, zmax), title=title,
             xlabel='frame', ylabel='ball centre height above sand [m]')
    # state bands
    col = dict(ground=(235, 225, 200), held=(225, 235, 250), unknown=(245, 245, 245))
    st = d.state.values; fr = d.frame.values
    a = 0
    for i in range(1, len(d) + 1):
        if i == len(d) or st[i] != st[a]:
            if st[a] in col:
                p.band(fr[a] - 0.5, fr[i - 1] + 0.5, col[st[a]])
            a = i
    p.hline(2.20, 'gray')
    p.text(f0, 2.22, 'net 2.20 m', 'gray', 'lb')
    # uncertainty band
    z = d.z_m.values; s = d.z_sigma_m.values
    fl = d.method.isin(['flight+shadow']).values
    p.line(fr, np.where(fl, z, np.nan), 'blue', 2, 'flight, camera + shadow')
    p.line(fr, np.where(d.method.eq('flight').values, z, np.nan), 'orange', 2, 'flight, camera only (cloud / no shadow)')
    p.line(fr, np.where(d.state.eq('ground').values, z, np.nan), 'brown', 2, 'on the sand')
    p.line(fr, np.where(d.state.eq('held').values, z, np.nan), 'purple', 2, 'carried / at a player')
    p.line(fr, np.where(d.method.eq('interp').values, z, np.nan), 'gray', 1, 'interpolated')
    p.scatter(fr, d.z_shadow_m.values, 'red', 2, 'direct shadow triangulation (single frame)')
    if ct is not None and len(ct):
        for _, c in ct.iterrows():
            p.vline(c.frame, dict(player='green', sand='brown', net='black').get(c.kind, 'lightgray'), 1, dash=True)
            if c.kind == 'player':
                p.text(c.frame, min(zmax - 0.3, c.z_m + 0.25), '%s %s' % (c.player, c.body_part), 'green', 'mb')
    p.legend()
    p.save(path)


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else HERE
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, 'plots')
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(os.path.join(src, 'ball_height.csv'))
    ct = pd.read_csv(os.path.join(src, 'ball_contacts.csv'))
    for shot, g in d.groupby('shot'):
        if not np.isfinite(g.z_m).any():
            continue
        plot_shot(g, ct[ct.shot == shot], os.path.join(out, 'shot_%02d_frames_%05d_%05d.png' % (shot, g.frame.min(), g.frame.max())),
                  'shot %d  (frames %d-%d)' % (shot, g.frame.min(), g.frame.max()))
