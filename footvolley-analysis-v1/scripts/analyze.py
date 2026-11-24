#!/usr/bin/env python3
"""Pre-declared hypothesis tests on the derived tables (see coaching_report.md / insights.json).

Run after build_tables.py, from the repository root:
    python3 footvolley-analysis-v1/scripts/analyze.py
Writes footvolley-analysis-v1/derived/analysis_results.json. Descriptive statistics with
rally-cluster bootstrap intervals; one match => exploratory, no between-match generalisation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'footvolley-analysis-v1' / 'derived'
RNG = np.random.default_rng(20261124)
FPS = 25.0
res = {}

rallies = pd.read_csv(D / 'rallies.csv')
poss = pd.read_csv(D / 'possessions.csv')
cands = pd.read_csv(D / 'contact_candidates.csv')
snaps = pd.read_csv(D / 'snapshots_incoming.csv')
players = pd.read_csv(D / 'players_norm.csv')
points = pd.read_csv(ROOT / 'point_events_FINAL_VERIFIED.csv').sort_values('point_number')


def boot_ci(groups, stat, n=4000):
    """Cluster bootstrap: groups = list of per-cluster DataFrames."""
    vals = []
    k = len(groups)
    for _ in range(n):
        pick = [groups[i] for i in RNG.integers(0, k, k)]
        v = stat(pd.concat(pick))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if vals else None


# ------------------------------------------------------------------ H1: scoreboard sequence
seq = []
for s_id, grp in [(1, points[points.point_number <= 31]), (2, points[points.point_number >= 32])]:
    prev = None
    for p in grp.itertuples():
        batch2 = p.timestamp_precision == 'BATCH_ONLY'
        seq.append({'set': s_id, 'point': p.point_number, 'winner': p.country,
                    'prev_winner': prev, 'order_known': not batch2})
        prev = p.country
seq = pd.DataFrame(seq)
# for the two batch updates the two points were won by the same team (ISR), so the
# order is irrelevant for "previous winner" of the second one and of the next point.
elig = seq[seq.prev_winner.notna()]
h1 = {}
for team in ('ISR', 'BRA'):
    hold = elig[elig.prev_winner == team]
    side = elig[elig.prev_winner != team]
    h1[team] = {
        'won_after_winning_prev (serve if winner-serves rule)': f'{(hold.winner == team).sum()}/{len(hold)}',
        'won_after_losing_prev (receive if winner-serves rule)': f'{(side.winner == team).sum()}/{len(side)}',
    }
h1['total_points'] = {'ISR': int((seq.winner == 'ISR').sum()), 'BRA': int((seq.winner == 'BRA').sum())}
h1['set_scores'] = 'Set1 BRA 18-13, Set2 BRA 18-11'
# streaks
runs, cur = [], None
for p in seq.itertuples():
    if cur and cur['team'] == p.winner and cur['set'] == p.set:
        cur['len'] += 1; cur['end'] = p.point
    else:
        cur = {'team': p.winner, 'set': p.set, 'start': p.point, 'end': p.point, 'len': 1}; runs.append(cur)
h1['streaks_ge3'] = [r for r in runs if r['len'] >= 3]
# bootstrap the ISR side-out rate over points (points are serially dependent -> wide CI, descriptive)
isr_so = elig[elig.prev_winner == 'BRA']
h1['ISR_sideout_rate'] = round(float((isr_so.winner == 'ISR').mean()), 3)
h1['BRA_sideout_rate'] = round(float((elig[elig.prev_winner == 'ISR'].winner == 'BRA').mean()), 3)
by_set = {}
for s_id in (1, 2):
    e = elig[elig.set == s_id]
    by_set[s_id] = {t: f"{((e.prev_winner != t) & (e.winner == t)).sum()}/{(e.prev_winner != t).sum()}" for t in ('ISR', 'BRA')}
h1['sideouts_by_set'] = by_set
res['H1_scoreboard'] = h1

# ------------------------------------------------------------------ incoming possessions
inc = poss[(poss.poss_idx >= 1)].copy()
prev = poss[['rally_id', 'poss_idx', 'next_cross_clean']].rename(columns={'poss_idx': 'pi', 'next_cross_clean': 'arrived_clean'})
prev['poss_idx'] = prev.pi + 1
inc = inc.merge(prev[['rally_id', 'poss_idx', 'arrived_clean']], on=['rally_id', 'poss_idx'])
inc = inc[inc.arrived_clean & inc.eligible]
inc['event_id'] = inc.rally_id + '_in' + inc.poss_idx.astype(str)
inc['fail'] = np.where(inc.outcome == 'returned', 0, np.where(inc.outcome == 'terminal_lost', 1, np.nan))
pairs = snaps[(snaps.observed) & (snaps.valid_ground)].groupby('event_id').filter(lambda g: g.player_id.nunique() == 2)
shape = pairs.groupby('event_id').apply(lambda g: pd.Series({
    'team': g.team_id.iloc[0],
    'sep_m': float(np.hypot(*(g[['x_m', 'depth_m']].iloc[0] - g[['x_m', 'depth_m']].iloc[1]))),
    'width_m': abs(g.x_m.iloc[0] - g.x_m.iloc[1]),
    'stagger_m': abs(g.depth_m.iloc[0] - g.depth_m.iloc[1]),
    'mean_depth_m': g.depth_m.mean(), 'front_depth_m': g.depth_m.min(), 'back_depth_m': g.depth_m.max(),
    'mean_lateral_m': g.x_m.mean(),
}), include_groups=False).reset_index()
inc = inc.merge(shape, on='event_id', how='left', suffixes=('', '_s'))
inc.to_csv(D / 'incoming_possessions.csv', index=False)

h2 = {'eligible_incoming_possessions': int(len(inc)), 'with_complete_observed_pair': int(inc.sep_m.notna().sum()),
      'outcome_counts': inc.outcome.value_counts().to_dict()}
labeled = inc[inc.fail.notna() & inc.sep_m.notna()]
h2['labeled_returned_or_lost'] = int(len(labeled))
h2['rally_clusters'] = int(labeled.rally_id.nunique())
desc = {}
for team in ('ISR', 'BRA'):
    t = labeled[labeled.team == team]
    desc[team] = {'n': int(len(t)), 'fail_rate': round(float(t.fail.mean()), 3),
                  **{c: {'median': round(float(t[c].median()), 2), 'iqr': [round(float(t[c].quantile(.25)), 2), round(float(t[c].quantile(.75)), 2)]}
                     for c in ('sep_m', 'width_m', 'stagger_m', 'mean_depth_m', 'front_depth_m', 'back_depth_m')}}
h2['shape_by_team'] = desc
# predeclared: stagger (front/back split) >= 2.0 m vs < 2.0 m ; separation tertiles
for team in ('ISR', 'BRA', 'ALL'):
    t = labeled if team == 'ALL' else labeled[labeled.team == team]
    out = {}
    for name, mask in [('stagger>=2m', t.stagger_m >= 2.0), ('stagger<2m', t.stagger_m < 2.0),
                       ('sep>=4m', t.sep_m >= 4.0), ('sep<4m', t.sep_m < 4.0),
                       ('back_depth>=6m', t.back_depth_m >= 6.0), ('back_depth<6m', t.back_depth_m < 6.0)]:
        sub = t[mask]
        out[name] = {'fails': int(sub.fail.sum()), 'n': int(len(sub)), 'rate': round(float(sub.fail.mean()), 3) if len(sub) else None}
    h2[f'fail_by_shape_{team}'] = out
groups = [g for _, g in labeled.groupby('rally_id')]
for c, thr in [('stagger_m', 2.0), ('sep_m', 4.0), ('back_depth_m', 6.0)]:
    def diff(df, c=c, thr=thr):
        a, b = df[df[c] >= thr], df[df[c] < thr]
        return (a.fail.mean() - b.fail.mean()) if len(a) and len(b) else None
    h2[f'diff_fail_{c}_ge{thr}_minus_lt'] = {'point': round(float(diff(labeled)), 3), 'rally_bootstrap_95': boot_ci(groups, diff)}
res['H2_incoming_shape_vs_outcome'] = h2

# ------------------------------------------------------------------ H3: player occupancy & roles
active = []
for p in poss.itertuples():
    active.append(players[(players.frame >= p.first_frame) & (players.frame <= p.last_frame)].assign(rally_id=p.rally_id))
act = pd.concat(active).drop_duplicates(['frame', 'player'])
act = act[act.observed & act.valid_ground]
act['lat_zone'] = pd.cut(act.lateral_m, [-99, 3, 6, 99], labels=['left', 'centre', 'right'])
act['dep_zone'] = pd.cut(act.depth_m, [-99, 3, 6, 99], labels=['front', 'mid', 'back'])
occ = (act.groupby(['player', 'dep_zone', 'lat_zone'], observed=False).size() /
       act.groupby('player').size()).round(3)
h3 = {'active_player_frames': int(len(act)),
      'occupancy_fraction': {pl: {f'{d}-{l}': float(occ[pl][d][l]) for d in ['front', 'mid', 'back'] for l in ['left', 'centre', 'right']}
                             for pl in sorted(act.player.unique())},
      'median_lateral_m': act.groupby('player').lateral_m.median().round(2).to_dict(),
      'median_depth_m': act.groupby('player').depth_m.median().round(2).to_dict()}
# who is on the left (lower lateral) at incoming anchors, per team, per rally
roles = {}
for team in ('ISR', 'BRA'):
    s = pairs[pairs.team_id == team]
    left = s.loc[s.groupby('event_id').x_m.idxmin()].player_id.value_counts().to_dict()
    deeper = s.loc[s.groupby('event_id').depth_m.idxmax()].player_id.value_counts().to_dict()
    roles[team] = {'player_on_left_at_incoming': left, 'deeper_player_at_incoming': deeper}
h3['roles_at_incoming'] = roles
miss = players.groupby('player').apply(lambda g: round(float((~g.observed).mean()), 4), include_groups=False).to_dict()
h3['interpolated_fraction_by_player'] = miss
res['H3_occupancy'] = h3

# ------------------------------------------------------------------ H4: contact candidates
c = cands[cands.poss_idx >= 0]
h4 = {'candidates_total': int(len(cands)), 'assigned_to_possession': int(len(c)),
      'ambiguous_player': int(cands.ambiguous_player.sum()),
      'region_by_player': cands.groupby('player').coarse_region.value_counts().unstack(fill_value=0).to_dict('index'),
      'candidates_per_possession': poss.n_candidate_contacts.value_counts().sort_index().to_dict()}
first = c[c.order_in_poss == 1].merge(inc[['rally_id', 'poss_idx', 'fail', 'team']], on=['rally_id', 'poss_idx', 'team'])
first = first[first.fail.notna()]
h4['first_touch_region_vs_fail'] = {reg: {'fails': int(g.fail.sum()), 'n': int(len(g))} for reg, g in first.groupby('coarse_region')}
res['H4_contact_candidates'] = h4

# ------------------------------------------------------------------ H5 (linked): touches used vs outcome
lab = inc[inc.fail.notna()]
h5 = {}
for k, g in lab.groupby(lab.n_candidate_contacts.clip(upper=3)):
    h5[f'{k}{"+" if k == 3 else ""}_candidates'] = {'fails': int(g.fail.sum()), 'n': int(len(g))}
h5['note'] = '0 candidates often means tracking/occlusion, not zero touches'
# time spent in own half before return (s), returned only, by team
ret = inc[inc.outcome == 'returned']
h5['possession_duration_s_returned'] = {t: {'median': round(float(((g.last_frame - g.first_frame) / FPS).median()), 2), 'n': int(len(g))}
                                        for t, g in ret.groupby('team')}
res['H5_linked_touches'] = h5

(D / 'analysis_results.json').write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps(res, indent=1, default=str))
