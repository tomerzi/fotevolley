"""Draw data/analysis/phase_heatmaps.json as data/analysis/phase_heatmaps.png with Pillow only (no matplotlib):
rows = players, columns = all / serve / reception / build / attack / defence; own view, net at the top (cyan),
the player's left on the left, white = court lines of the own half. Colour scale per map (99.5th percentile)."""
import os, json
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
d = json.load(open('data/analysis/phase_heatmaps.json'))
summ = pd.read_csv('data/analysis/phase_heatmaps_summary.csv')
PH = ['all', 'serve', 'reception', 'build', 'attack', 'defence']
HE = {'all': 'all rally', 'serve': 'serve', 'reception': 'reception', 'build': 'build attack', 'attack': 'after own attack', 'defence': 'defence'}
S = 24  # px per metre
cell, x0, y0 = d['cell'], d['x0'], d['y0']
stops = np.array([[0, 0, 4], [60, 15, 110], [180, 50, 120], [250, 130, 60], [252, 250, 190]], float)
def cmap(v):
    v = np.clip(v, 0, 1) * (len(stops) - 1); i = np.minimum(v.astype(int), len(stops) - 2); f = (v - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
tiles = []
for p, m in d['maps'].items():
    row = []
    for ph in PH:
        H = np.array(m[ph])                         # [x (baseline->net), y (left->right)]
        q = np.quantile(H[H > 0], 0.995) if (H > 0).any() else 1
        img = cmap(H / q)[::-1]                      # net at the top
        im = Image.fromarray(img).resize((int(H.shape[1] * cell * S), int(H.shape[0] * cell * S)), Image.BILINEAR)
        dr = ImageDraw.Draw(im)
        X = lambda xo: (d['nx'] * cell - (xo - x0)) * S   # image row of own-view x
        Y = lambda yo: (yo - y0) * S
        dr.rectangle([Y(0), X(8), Y(8), X(0)], outline=(255, 255, 255), width=2)
        dr.line([Y(-1.5), X(8), Y(9.5), X(8)], fill=(0, 230, 255), width=4)
        s = summ[(summ.player == p) & (summ.phase == ph)].iloc[0]
        dr.rectangle([0, im.height - 34, im.width, im.height], fill=(0, 0, 0))
        dr.text((4, im.height - 32), f'{p} | {HE[ph]}', fill=(255, 255, 0))
        dr.text((4, im.height - 17), f'{s.seconds:.0f}s  net {s.from_net_m}m  left {s.left_half:.0%}', fill=(255, 255, 255))
        row.append(im)
    tiles.append(row)
w, h = tiles[0][0].size
out = Image.new('RGB', (w * len(PH) + 5 * (len(PH) - 1), h * len(tiles) + 5 * (len(tiles) - 1)), (40, 40, 40))
for i, row in enumerate(tiles):
    for j, im in enumerate(row):
        out.paste(im, (j * (w + 5), i * (h + 5)))
out.save('data/analysis/phase_heatmaps.png'); print(out.size)
