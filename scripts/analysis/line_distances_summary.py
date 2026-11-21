"""Summary of data/analysis/line_distances_<start>_<end>.csv for play time only (the 60 rallies in
data/points/rallies_v2.csv; dead time between points is left out).
Writes data/analysis/line_distances_summary_players.csv (per player) and line_distances_by_rally.csv (per rally and
player: median distance from each line, closest approach to the net, share of frames outside a line).
usage: python scripts/analysis/line_distances_summary.py [START END]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
F0, F1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (0, 15585)
LINES = ['baseline', 'far sideline', 'near sideline', 'net']
ORDER = ['Maor_Haas', 'Ron_Ben_Ishay', 'Franklin', 'Hiltinho']
d = pd.read_csv(f'data/analysis/line_distances_{F0}_{F1}.csv')
ral = pd.read_csv('data/points/rallies_v2.csv')
rid = pd.Series(np.nan, index=range(F0, F1 + 1))
for r in ral.itertuples():
    rid.loc[r.start_frame:r.end_frame] = r.rally
d['rally'] = d.frame.map(rid)
play = d.dropna(subset=['rally']).copy()
play['outside'] = (play[['baseline', 'far sideline', 'near sideline']] < 0).any(axis=1)
pl = play.groupby('player').agg(frames=('frame', 'size'), **{f'{ln}_median_m': (ln, 'median') for ln in LINES},
                                nearest_line_median_m=('nearest_m', 'median'), net_min_m=('net', 'min'),
                                outside_pct=('outside', lambda v: v.mean() * 100)).loc[ORDER].round(2)
pl['nearest_line_most_often'] = play.groupby('player').nearest_line.agg(lambda v: v.value_counts().idxmax()).loc[ORDER]
pl.to_csv('data/analysis/line_distances_summary_players.csv')
print(pl.to_string())
br = play.groupby(['rally', 'player']).agg(**{f'{ln}_median_m': (ln, 'median') for ln in LINES}, net_min_m=('net', 'min'),
                                           outside_pct=('outside', lambda v: v.mean() * 100)).round(2).reset_index()
br.to_csv('data/analysis/line_distances_by_rally.csv', index=False)
print('rallies:', br.rally.nunique())

# chart: one panel per player, median distance from each line in every rally (play time only)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL_HEX = {'baseline': '#2a78d6', 'far sideline': '#eb6834', 'near sideline': '#1baf7a', 'net': '#eda100'}
NAMES = {'Franklin': 'Franklin (BRA)', 'Hiltinho': 'Hiltinho (BRA)', 'Maor_Haas': 'Maor Haas (ISR)', 'Ron_Ben_Ishay': 'Ron Ben Ishay (ISR)'}
fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, sharey=True)
for ax, p_ in zip(axs, ORDER):
    g = br[br.player == p_].sort_values('rally')
    for ln in LINES:
        ax.plot(g.rally, g[f'{ln}_median_m'], color=COL_HEX[ln], lw=2, marker='o', ms=3.5)
        last = g.iloc[-1]
        ax.annotate(ln, (last.rally, last[f'{ln}_median_m']), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color='#52514c')
    ax.set_title(NAMES[p_], loc='left', fontsize=10, color='#1d1d1b')
    ax.set_ylabel('m', color='#52514c'); ax.grid(axis='y', color='#e5e4dd', lw=0.6); ax.set_ylim(0, None)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#b9b8b0'); ax.spines['bottom'].set_color('#b9b8b0'); ax.tick_params(colors='#52514c', labelsize=8)
axs[-1].set_xlabel('rally', color='#52514c')
fig.legend([plt.Line2D([0], [0], color=COL_HEX[ln], lw=2) for ln in LINES], LINES, loc='upper right', ncol=4, frameon=False, fontsize=9)
fig.suptitle('Median distance from the court lines in each rally, whole video (play time only)', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0, 0.94, 0.96))
fig.savefig('video/analysis/line_distances_by_rally.png', dpi=130, facecolor='#fcfcfb')
