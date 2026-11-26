"""Coach metrics recomputed with the owner's labelling-game answers (2026-11-24 evening), without touching data/analysis.

Reads data/contacts/touches_master.csv, data/contacts/owner_decisions.csv (all rows, including the labelling game),
data/experiments/labelling_game/answers.csv (stance), data/analysis/touch_body.csv (earlier stance labels),
data/analysis/set_quality.csv (air time / apex of every pass to the partner), data/points/point_end_verified.csv.
Writes coach_analysis/recomputed_2026-11-24/*.csv.

Action of a touch = the owner's decision when there is one, else the possession rule in touches_master.csv.
A rally whose first filmed touch is a set has an unfilmed serve reception ('reception_unfilmed'); there the set is
touch 2 of its possession. Possessions are rebuilt from the final actions: a new possession starts when the team
changes or after an attack.
usage: python scripts/analysis/round2_recompute.py"""
import os
from math import sqrt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = 'coach_analysis/recomputed_2026-11-24'
os.makedirs(OUT, exist_ok=True)

def wilson(k, n, z=1.96):
    if n == 0: return ''
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{round(100 * (c - h))}-{round(100 * (c + h))}'

t = pd.read_csv('data/contacts/touches_master.csv')
dec = pd.read_csv('data/contacts/owner_decisions.csv')
for d in dec[dec.kind == 'touch'].itertuples():
    sel = t.touch == int(d.touch)
    if d.field == 'in_play': t.loc[sel, 'in_play'] = d.value == 'True'
    elif d.field == 'action': t.loc[sel, 'action'] = d.value
    elif d.field == 'body_part': t.loc[sel, 'body_part'] = d.value
    elif d.field == 'limb_side': t.loc[sel, 'limb_side'] = d.value
    elif d.field == 'kind': t.loc[sel, 'kind'] = d.value
for d in dec[dec.kind == 'add'].itertuples():
    part = dec[(dec.kind == 'add_detail') & (dec.rally == d.rally) & (dec.field == 'body_part')].value
    t = pd.concat([t, pd.DataFrame([dict(touch=t.touch.max() + 1, frame=int(d.frame), rally=int(d.rally), set=int(t[t.rally == d.rally].set.iloc[0]),
                                         in_play=True, kind='player', player=d.player, team='ISR', body_part=part.iloc[0] if len(part) else 'unknown',
                                         action='attack', action_subtype='added_by_owner')])], ignore_index=True)
t = t[(t.kind == 'player') & t.in_play.astype(bool) & t.player.notna()].sort_values(['rally', 'frame']).reset_index(drop=True)
t['body_part'] = t.body_part.fillna('unknown')

# possessions from the final actions
poss, tip, unf = [], [], []
for r, g in t.groupby('rally', sort=False):
    k, prev_team, prev_att, n, first = 0, None, False, 0, True
    for x in g.itertuples():
        if prev_team != x.team or prev_att:
            k += 1; n = 0
            start_unfilmed = first and x.action == 'set'
        n += 1
        poss.append(k); tip.append(n + (1 if start_unfilmed else 0)); unf.append(start_unfilmed)
        prev_team, prev_att, first = x.team, x.action == 'attack', False
t['poss'] = poss; t['tip'] = tip; t['reception_unfilmed'] = unf

e = pd.read_csv('data/points/point_end_verified.csv').set_index('point')
t['last'] = t.groupby('rally').touch.transform('last') == t.touch
t['shown'] = t.rally.map(e.end_type) != 'excluded'
t['winner'] = t.rally.map(e.winner)
t['lost'] = t['last'] & t.shown & (t.winner != t.team)
t['won'] = t['last'] & t.shown & (t.winner == t.team)
t['success'] = np.where(t['last'] & ~t.shown, np.nan, (~t.lost).astype(float))
t['pkey'] = t.rally.astype(str) + '_' + t.poss.astype(str)
att = set(t[t.action == 'attack'].pkey)

# stance: earlier owner labels + labelling game
tb = pd.read_csv('data/analysis/touch_body.csv').set_index('touch')
ga = pd.read_csv('data/experiments/labelling_game/answers.csv')
ga = ga[ga.kind == 'sample_touch'].set_index('touch')
t['stance'] = t.touch.map(ga.stance).fillna(t.touch.map(tb.stance_label))
t.to_csv(f'{OUT}/touches.csv', index=False)

# 1. touches by player x action x body part
pd.crosstab([t.team, t.player, t.action], t.body_part, margins=True).to_csv(f'{OUT}/touches_by_action_part.csv')

# 2. touch success
rows = []
for (p, a), g in list(t.groupby(['player', 'action'])) + [((p, 'all'), g) for p, g in t.groupby('player')]:
    el = int(g.success.notna().sum()); k = int(g.success.sum())
    rows.append(dict(player=p, action=a, success=k, eligible=el, unknown=int(g.success.isna().sum()), pct=round(100 * k / el) if el else None, ci95=wilson(k, el)))
pd.DataFrame(rows).to_csv(f'{OUT}/touch_success.csv', index=False)

# 3. reception stability: filmed receptions; serve = first possession of the rally
rec = t[(t.action == 'reception') & (t.tip == 1)].copy()
rec['situation'] = np.where(rec.poss == 1, 'serve', 'attack')
rec['to_attack'] = rec.pkey.isin(att)
rows = []
for keys, g in list(rec.groupby(['team', 'situation'])) + list(rec.groupby(['player', 'situation'])) + list(rec.groupby(['body_part', 'situation'])):
    k, n = int(g.to_attack.sum()), len(g)
    rows.append(dict(who=keys[0], situation=keys[1], n=n, to_attack=k, pct=round(100 * k / n), ci95=wilson(k, n), lost_on_reception=int(g.lost.sum())))
pd.DataFrame(rows).to_csv(f'{OUT}/reception_stability.csv', index=False)
uf = t[t.reception_unfilmed & (t.tip == 2)]
u = uf.assign(to_attack=uf.pkey.isin(att)).groupby('team').agg(rallies=('rally', 'size'), to_attack=('to_attack', 'sum'))
u.to_csv(f'{OUT}/unfilmed_reception_rallies.csv')

# 4. set quality (sets only), from the pass table
sq = pd.read_csv('data/analysis/set_quality.csv')
sq['action'] = sq.touch.map(t.set_index('touch').action)
s = t[t.action == 'set'].copy()
nx = t.groupby('rally')[['team', 'action', 'player']].shift(-1)
s['q'] = np.where(s.lost, 'failed', np.where(nx.loc[s.index, 'team'].isna(), 'unknown',
          np.where((nx.loc[s.index, 'team'] == s.team) & (nx.loc[s.index, 'action'] == 'attack'), 'partner_attack', 'other')))
rows = []
for p, g in s.groupby('player'):
    q = sq[(sq.action == 'set') & (sq.player == p)]
    rows.append(dict(player=p, sets=len(g), partner_attack=int((g.q == 'partner_attack').sum()), failed=int((g.q == 'failed').sum()),
                     other=int((g.q == 'other').sum()), after_unfilmed_reception=int(g.reception_unfilmed.sum()),
                     with_flight=len(q), air_s_median=q.air_time_s.median(), air_s_q1=q.air_time_s.quantile(.25), air_s_q3=q.air_time_s.quantile(.75),
                     apex_m_median=q.apex_m.median(), partner_move_m_median=q.partner_move_m.median()))
pd.DataFrame(rows).round(2).to_csv(f'{OUT}/set_quality_sets.csv', index=False)

# 5. attacks: outcome and touch number in possession
a = t[t.action == 'attack']
rows = []
for p, g in a.groupby('player'):
    rows.append(dict(player=p, attacks=len(g), point_won=int(g.won.sum()), point_lost=int(g.lost.sum()), unknown=int(g.success.isna().sum()),
                     on_touch_1=int((g.tip == 1).sum()), on_touch_2=int((g.tip == 2).sum()), on_touch_3=int((g.tip == 3).sum())))
pd.DataFrame(rows).to_csv(f'{OUT}/attacks.csv', index=False)

# 6. how each point ended, with the net-fault author
fault = dec[(dec.kind == 'point') & (dec.field == 'fault_by')].set_index('rally').value
rows = []
for p, E in e.iterrows():
    g = t[t.rally == p]; l = g.iloc[-1] if len(g) else None
    if E.end_type == 'excluded': cat = 'not_shown'
    elif E.end_type == 'fault': cat = 'fault'
    elif l is not None and l.team == E.winner: cat = 'winning_touch'
    elif E.end_type == 'landed_in': cat = 'landed_on_own_side'
    elif E.end_type in ('out', 'landed_out'): cat = 'out'
    else: cat = 'net'
    who_lost = (fault.get(p) or (E.last_touch if isinstance(E.last_touch, str) else None)) if cat == 'fault' else (l.player if l is not None and cat != 'winning_touch' and cat != 'not_shown' else None)
    rows.append(dict(point=p, set=E.set, winner=E.winner, loser=E.loser, category=cat, last_player=None if l is None else l.player,
                     last_action=None if l is None else l.action, last_part=None if l is None else l.body_part, last_tip=None if l is None else l.tip,
                     point_lost_by=who_lost))
P = pd.DataFrame(rows)
P.to_csv(f'{OUT}/points.csv', index=False)

# 7. stance per player x action (owner labels only)
st = t[t.stance.isin(['set', 'moving', 'jump'])]
pd.crosstab([st.player, st.action], st.stance, margins=True).to_csv(f'{OUT}/stance.csv')
print('touches', len(t), 'unique', t.touch.is_unique, '| unfilmed-reception rallies', int((t.reception_unfilmed & (t.tip == 2)).sum()), '| stance labels', len(st))
