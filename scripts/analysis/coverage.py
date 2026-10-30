"""Area each player covers (v3 video), from data/analysis/player_positions_v3.csv.

Per player and rally (rallies_v3video.csv start..end): convex hull of the smoothed positions (area, units^2), the
largest distance from the player's mean position, and the share of time in each 3x3 square of the half - by the
court squares L1-R9 (scripts/court/court_squares.py) and as the team sees them (share_<lane>_<col>: lanes left /
mid / right for a player facing the net, columns net / mid / base from the net back; same for both halves).
Per player over all rallies in the team's view, so that the side switches do not matter: xr = distance from the
net into the own half, yr = across the court from the team's left sideline (0) to its right (9) when facing the net
(the half is the one holding most of the player's positions in that rally; seen from the camera, the left-half
team's right is the near sideline, the right-half team's right is the far one). Convex hull, and the areas of the
smallest regions holding 50% / 90% of the time (positions binned on a CELL grid and smoothed with a BW Gaussian),
plus the share of time per square.

Lengths are court units (court lines = 18 x 9); x 8/9 for metres, x 64/81 for areas, if the court is 16 x 8.

inputs : data/analysis/player_positions_v3.csv, data/points/rallies_v3video.csv
outputs: data/analysis/coverage_rallies_v3.csv, data/analysis/coverage_match_v3.csv
usage: python scripts/analysis/coverage.py [HEATMAP.png]   (optional: one own-half heat map per player)"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, QhullError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data', 'analysis')
CELL = 0.1           # units, density grid
BW = 0.5             # units, Gaussian smoothing of the density
EXTENT = (-4, 13, -5, 14)   # xr0, xr1, yr0, yr1 of the density grid in team-view coordinates (players run outside too)
SQUARES = [f'{s}{i}' for s in 'LR' for i in range(1, 10)]
LANES, COLS = ('left', 'mid', 'right'), ('net', 'mid', 'base')


def hull_area(xy):
    try:
        return ConvexHull(xy).volume if len(xy) >= 3 else 0.0
    except QhullError:          # all points on a line
        return 0.0


def density(xy):
    """share of time per unit^2 on the CELL grid over EXTENT (rows = x, columns = y)."""
    x0, x1, y0, y1 = EXTENT
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[np.arange(x0, x1 + CELL, CELL), np.arange(y0, y1 + CELL, CELL)])
    return gaussian_filter(h, BW / CELL) / len(xy) / CELL ** 2


def density_areas(xy, levels=(0.5, 0.9)):
    d = np.sort(density(xy).ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return [float((np.searchsorted(c, q) + 1) * CELL ** 2) for q in levels]


def rel_square(sq):
    """court square -> (lane as the team sees it, column from the net); None for 'out'. Squares are numbered from
    the far sideline (row 0) and from x = 0 (column 0): on the left half row 0 is the team's left and column 0 the
    baseline, on the right half row 0 is the team's right and column 0 at the net."""
    if sq == 'out':
        return None
    i = int(sq[1:]) - 1
    row, col = i // 3, i % 3
    if sq[0] == 'L':
        return f'{LANES[row]}_{COLS[2 - col]}'
    return f'{LANES[2 - row]}_{COLS[col]}'


def square_shares(sq):
    v = sq.value_counts(normalize=True)
    rel = sq.map(rel_square).value_counts(normalize=False) / len(sq)
    out = {f'share_{s}': float(v.get(s, 0.0)) for s in SQUARES + ['out']}
    out.update({f'share_{r}_{c}': float(rel.get(f'{r}_{c}', 0.0)) for r in LANES for c in COLS})
    return out


def main():
    pos = pd.read_csv(os.path.join(OUT, 'player_positions_v3.csv'), usecols=['frame', 'player', 'x', 'y', 'square'])
    rallies = pd.read_csv(os.path.join(ROOT, 'data', 'points', 'rallies_v3video.csv'))
    rows, allr = [], []
    for r in rallies.itertuples():
        seg = pos[pos.frame.between(r.start_frame, r.end_frame)]
        for pl, g in seg.groupby('player'):
            xy = g[['x', 'y']].values
            left = (g.x < 9).mean() > 0.5
            allr.append(g.assign(xr=9 - g.x if left else g.x - 9, yr=g.y if left else 9 - g.y))
            rows.append(dict(rally=r.rally, set=r.set, player=pl, team=pl[:3], winner=r.winner, frames=len(g),
                             hull_area=hull_area(xy), max_from_mean=float(np.hypot(*(xy - xy.mean(0)).T).max()),
                             **square_shares(g.square)))
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'coverage_rallies_v3.csv'), index=False)
    allr = pd.concat(allr)
    match = []
    for pl, g in allr.groupby('player'):
        xy = g[['xr', 'yr']].values
        a50, a90 = density_areas(xy)
        match.append(dict(player=pl, team=pl[:3], frames=len(g), hull_area=hull_area(xy), area50=a50, area90=a90,
                          **square_shares(g.square)))
    m = pd.DataFrame(match)
    m.round(4).to_csv(os.path.join(OUT, 'coverage_match_v3.csv'), index=False)
    print(m[['player', 'frames', 'hull_area', 'area50', 'area90']].round(2).to_string(index=False))
    rel = m.set_index('player')[[f'share_{r}_{c}' for r in LANES for c in COLS]]
    print((rel * 100).round(0).to_string())
    if len(sys.argv) > 1:
        heatmaps(allr, m, sys.argv[1])


BLUE = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # sequential ramp
INK, MUTED, GRID = '#1f1f1d', '#6b6b66', '#9a9a94'


def heatmaps(allr, m, path):
    """one panel per player in the team's view: net at the top, the team's left sideline on the left."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', BLUE)
    dens = {pl: density(g[['xr', 'yr']].values) for pl, g in allr.groupby('player')}
    vmax = np.percentile(np.concatenate([d.ravel() for d in dens.values()]), 99.8)
    x0, x1, y0, y1 = EXTENT
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.9), constrained_layout=True)
    for ax, (pl, d) in zip(axes, sorted(dens.items())):
        im = ax.imshow(d, origin='upper', extent=(y0, y1, x1, x0), cmap=cmap, vmin=0, vmax=vmax,
                       interpolation='bilinear')
        for k in (3, 6):
            ax.plot([k, k], [0, 9], color=GRID, lw=0.8)
            ax.plot([0, 9], [k, k], color=GRID, lw=0.8)
        ax.plot([0, 9, 9, 0], [0, 0, 9, 9], color=MUTED, lw=1.2)
        ax.plot([-0.6, 9.6], [0, 0], color=INK, lw=2.5)
        ax.text(9.6, -0.15, 'net', color=INK, fontsize=9, ha='right', va='bottom')
        r = m.set_index('player').loc[pl]
        ax.set_title(f'{pl}   90% area {r.area90:.0f} u\u00b2', color=INK, fontsize=11, loc='left')
        ax.set_xlim(-1.5, 10.5); ax.set_ylim(10.5, -1.5)
        ax.set_xticks([0, 3, 6, 9]); ax.set_yticks([0, 3, 6, 9])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("team's left \u2192 right, facing the net (court units)", color=MUTED, fontsize=8)
    axes[0].set_ylabel('distance from the net (court units)', color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label('share of rally time per unit\u00b2', color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.outline.set_visible(False)
    fig.suptitle("Where each player spends rally time (own half in the team's view, all 60 rallies)", color=INK,
                 fontsize=13, x=0.01, ha='left')
    fig.savefig(path, dpi=130, facecolor='white')


if __name__ == '__main__':
    main()
