#!/usr/bin/env python3
"""Robustness checks for the separation -> failed-defence candidate (H2).

Varies: separation threshold, anchor offset (frames before the ball crosses the net),
inclusion of interpolated boxes, and per-phase split. Also lists the failing events for
manual video review. Run from repo root after build_tables.py and analyze.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'footvolley-analysis-v1' / 'derived'
inc = pd.read_csv(D / 'incoming_possessions.csv')
pl = pd.read_csv(D / 'players_norm.csv')
snap = pd.read_csv(D / 'snapshots_incoming.csv')
anchor = snap.groupby('event_id').agg(anchor_frame=('anchor_frame', 'first'), phase=('phase', 'first')).reset_index()
inc = inc.merge(anchor, on='event_id', how='left')
lab = inc[inc.fail.notna()].copy()
out = {}


def sep_at(frame, team, allow_interp=False):
    g = pl[(pl.frame == frame) & (pl.team == team)]
    if not allow_interp:
        g = g[g.observed]
    if g.player.nunique() != 2:
        return np.nan
    a, b = g.iloc[0], g.iloc[1]
    return float(np.hypot(a.lateral_m - b.lateral_m, a.depth_m - b.depth_m))


rows = []
for off in (0, 5, 10, 15):  # 0.0, 0.2, 0.4, 0.6 s before the crossing anchor
    s = lab.apply(lambda r: sep_at(int(r.anchor_frame) - off, r.team), axis=1)
    for thr in (3.5, 4.0, 4.5):
        for team in ('ISR', 'BRA'):
            m = lab.team == team
            w, n_ = m & (s >= thr), m & (s < thr) & s.notna()
            rows.append({'offset_frames': off, 'thr_m': thr, 'team': team,
                         'wide_fail': f'{int(lab.fail[w].sum())}/{int(w.sum())}',
                         'narrow_fail': f'{int(lab.fail[n_].sum())}/{int(n_.sum())}'})
out['grid'] = rows

isr = lab[lab.team == 'ISR']
wide = isr[isr.sep_m >= 4.0]
out['isr_wide_events'] = wide[['event_id', 'rally_id', 'phase', 'anchor_frame', 'sep_m', 'width_m', 'stagger_m',
                               'front_depth_m', 'back_depth_m', 'outcome', 'n_candidate_contacts']].round(2).to_dict('records')
out['isr_wide_rally_clusters'] = int(wide.rally_id.nunique())
out['isr_narrow_fail_events'] = isr[(isr.sep_m < 4.0) & (isr.fail == 1)][['event_id', 'anchor_frame', 'sep_m', 'phase']].round(2).to_dict('records')
out['phase_split'] = {f'{t}/{ph}': {'wide': f"{int(g[g.sep_m >= 4].fail.sum())}/{int((g.sep_m >= 4).sum())}",
                                    'narrow': f"{int(g[g.sep_m < 4].fail.sum())}/{int((g.sep_m < 4).sum())}"}
                      for (t, ph), g in lab.groupby(['team', 'phase'])}
# Rally-level (one value per rally) to avoid within-rally dependence: did the rally contain an ISR wide incoming?
r = isr.groupby('rally_id').agg(any_wide=('sep_m', lambda x: bool((x >= 4).any())), any_fail=('fail', 'max'))
out['isr_rally_level'] = {'rallies_with_wide': f"{int(r[r.any_wide].any_fail.sum())}/{int(r.any_wide.sum())} ended in ISR failed defence",
                          'rallies_without_wide': f"{int(r[~r.any_wide].any_fail.sum())}/{int((~r.any_wide).sum())}"}
# Fisher exact (two-sided) on event-level ISR table, descriptive only
from math import comb


def fisher(a, b, c, d):
    n = a + b + c + d; r1 = a + b; c1 = a + c
    p0 = comb(r1, a) * comb(n - r1, c1 - a) / comb(n, c1)
    p = 0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        px = comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
        if px <= p0 + 1e-12:
            p += px
    return p


a = int(wide.fail.sum()); b = len(wide) - a
nar = isr[isr.sep_m < 4]; c = int(nar.fail.sum()); d = len(nar) - c
out['isr_fisher_two_sided_p_event_level'] = round(fisher(a, b, c, d), 4)
out['tests_run_in_H2'] = 'three predeclared shape splits x 2 teams (6 tests); Bonferroni alpha 0.05/6=0.0083'
(D / 'sensitivity_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
