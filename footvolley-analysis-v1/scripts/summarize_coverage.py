#!/usr/bin/env python3
"""Summarise coverage.py scenario outputs by team and ISR outcome (modeled, not measured)."""
import json
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parents[1] / 'derived'
inc = pd.read_csv(D / 'incoming_possessions.csv')[['event_id', 'fail']]
out = {}
for f in sorted((D / 'coverage').glob('cov_*.json')):
    d = json.loads(f.read_text())
    s = pd.DataFrame([{'event_id': x['event_id'], 'team': x['team_id'], 'union': x['union_fraction'],
                       'uncovered': x['uncovered_fraction'], 'overlap': x['overlap_fraction'],
                       'back_row_uncovered': sum(z['uncovered_fraction'] for z in x['zones'] if z['depth_zone'] == 3) / 3}
                      for x in d['snapshots']]).merge(inc, on='event_id', how='left')
    key = f.stem
    out[key] = {'assumptions': d['assumptions'], 'eligible': d['eligible_snapshots'], 'excluded': d['excluded_snapshots'],
                'by_team_median': s.groupby('team')[['union', 'uncovered', 'overlap', 'back_row_uncovered']].median().round(3).to_dict('index'),
                'by_team_fail_median_uncovered': {f'{k[0]}_fail{int(k[1])}': v for k, v in s[s.fail.notna()].groupby(['team', 'fail']).uncovered.median().round(3).to_dict().items()}}
(D / 'coverage_summary.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
