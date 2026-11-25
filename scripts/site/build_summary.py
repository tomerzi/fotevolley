"""Hebrew coach summary of the Graz 2022 final (one HTML page, all numbers from the repo).

Reads data/analysis/*, data/points/point_end_verified.csv and coach_analysis/recomputed_2026-11-24/*.
Writes coach_analysis/match_summary_he.html.
usage: python scripts/site/build_summary.py [OUT.html]
"""
import json
import sys

import numpy as np
import pandas as pd

A = 'data/analysis/'
R = 'coach_analysis/recomputed_2026-11-24/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'coach_analysis/match_summary_he.html'
TEMPLATE = 'scripts/site/summary_template.html'

HE = {'Franklin': 'פרנקלין', 'Hiltinho': 'הילטיניו', 'Maor_Haas': 'מאור חאס', 'Ron_Ben_Ishay': 'רון בן ישי',
      'ISR': 'ישראל', 'BRA': 'ברזיל'}
PLAYERS = ['Ron_Ben_Ishay', 'Maor_Haas', 'Franklin', 'Hiltinho']
PART = {'chest': 'חזה', 'foot': 'רגל', 'head': 'ראש', 'knee': 'ברך', 'shin': 'שוק', 'shoulder': 'כתף', 'unknown': 'לא ידוע'}
ACT = {'reception': 'קבלה', 'set': 'הגבהה', 'attack': 'התקפה', 'serve': 'הגשה'}
PHASE = {'all': 'כל המשחק', 'serve': 'הגשה', 'reception': 'קבלה', 'build': 'בניית מתקפה', 'attack': 'אחרי התקפה שלנו',
         'defence': 'הגנה'}
LANE = {'left': 'שמאל', 'mid': 'מרכז', 'right': 'ימין'}
DEPTH = {'net': 'רשת', 'mid': 'אמצע', 'base': 'בסיס'}


def n(p):
    return HE.get(p, p)


def f(x, d=2):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


S = {}  # html snippets for the template

# ---------- coverage, speed
cov = pd.read_csv(A + 'coverage_match_v3.csv').set_index('player')
spd = pd.read_csv(A + 'player_speed_rallies_v3.csv').groupby('player').agg(
    dist=('distance', 'sum'), mean=('mean_speed', 'mean'), top=('top_speed', 'max'), sprint=('sprint_sec', 'sum'))
ph = pd.read_csv(A + 'phase_heatmaps_summary.csv')
allp = ph[ph.phase == 'all'].set_index('player')
S['coverage'] = table(['שחקן', 'שטח ליבה 50% (מ״ר)', 'שטח 90% (מ״ר)', 'מעטפת כל המיקומים (מ״ר)', 'זמן בחצי השמאלי שלו',
                       'מרחק ממוצע מהרשת (מ׳)', 'מרחק כולל בראלים (מ׳)', 'מהירות ממוצעת (מ׳/ש׳)', 'מהירות שיא (מ׳/ש׳)',
                       'שניות ספרינט'],
                      [[n(p), f(cov.area50[p], 1), f(cov.area90[p], 1), f(cov.hull_area[p], 1),
                        f'{allp.left_half[p] * 100:.0f}%', f(allp.from_net_m[p]), f(spd.dist[p], 0), f(spd['mean'][p]),
                        f(spd.top[p]), f(spd.sprint[p], 0)] for p in PLAYERS])

# phase heat maps (downsampled 2x for the page)
hm = json.load(open(A + 'phase_heatmaps.json'))
maps = {}
for p in PLAYERS:
    maps[p] = {}
    for k, g in hm['maps'][p].items():
        a = np.array(g, float)
        a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]
        a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))
        maps[p][k] = np.round(a / a.max() * 99).astype(int).tolist() if a.max() > 0 else a.astype(int).tolist()
HM = dict(cell=hm['cell'] * 2, x0=hm['x0'], y0=hm['y0'], maps=maps, axes=hm['axes'])
print('heat map axes:', hm['axes'], np.array(hm['maps']['Franklin']['all']).shape)
pairs = pd.read_csv(A + 'phase_heatmaps_pairs.csv')
S['phase_table'] = table(['שלב'] + [f'{n(p)}<br><span class="mut">זמן בחצי השמאלי</span>' for p in PLAYERS] +
                         ['חפיפה ישראל', 'חפיפה ברזיל'],
                         [[PHASE[k]] + [f"{ph[(ph.player == p) & (ph.phase == k)].left_half.iloc[0] * 100:.0f}%" for p in PLAYERS] +
                          [f"{pairs[(pairs.team == t) & (pairs.phase == k)].overlap.iloc[0]:.2f}" for t in ('ISR', 'BRA')]
                          for k in PHASE])

# ---------- distances between partners
ff = pd.read_csv(A + 'formation_frames.csv')
fp = pd.read_csv(A + 'formation_points.csv')
cd = pd.read_csv(A + 'conceded_partner_distance.csv')
ts = pd.read_csv(A + 'team_spacing_v3.csv')
FORM = {'both_net': 'שניהם ברשת', 'front_back': 'אחד קדימה אחד אחורה', 'side_side': 'זה לצד זה'}
rows = []
for t in ('ISR', 'BRA'):
    g = ff[ff.team == t]
    fr = g.formation.value_counts(normalize=True)
    c = cd[cd.loser == t].distance_m
    tt = ts[ts.defending_team == t].groupby('outcome').distance.median()
    rows.append([n(t), f(g.distance.median()), f(g.distance.quantile(.25)) + '–' + f(g.distance.quantile(.75)),
                 ' · '.join(f'{FORM[k]} {fr.get(k, 0) * 100:.0f}%' for k in FORM),
                 f(c.median()) + f' <span class="mut">(n={len(c)})</span>', f(tt.get('returned')),
                 f(tt.get('kill')), f(tt.get('failed_defence'))])
S['spacing'] = table(['קבוצה', 'מרחק חציוני (מ׳)', 'טווח 25–75%', 'מערך', 'מרחק ברגע שהפסידו נקודה',
                      'בהגנה: החזירו התקפה', 'בהגנה: ספגו ווינר', 'בהגנה: נגעו ולא החזירו'], rows)

# ---------- chest
cp = pd.read_csv(A + 'chest_profile_by_player.csv').set_index('player')
tb = pd.read_csv(A + 'touch_body.csv')
c = tb[(tb.body_part == 'chest') & tb.in_play].copy()
for col in ['knee_l', 'knee_r', 'hip_l', 'hip_r', 'trunk_lean']:
    c.loc[~c[col + '_ok'].astype(bool), col] = np.nan
cm = c.groupby('player')[['knee_l', 'knee_r', 'hip_l', 'hip_r', 'trunk_lean']].median()
back = c.groupby('player').trunk_lean_dir.apply(lambda s: (s == 'back').mean())
S['chest'] = table(['שחקן', 'נגיעות חזה', 'מרחק בין הרגליים (מ׳)', 'מזה: לכיוון הרשת', 'מזה: מצד לצד',
                    'ברך שמאל°', 'ברך ימין°', 'ירך שמאל°', 'ירך ימין°', 'הטיית גו°', 'נשען אחורה', 'גובה הכדור (מ׳)',
                    'מהירות הכדור הנכנס (מ׳/ש׳)', 'מרחק מהרשת (מ׳)', 'שתי רגליים נטועות'],
                   [[n(p), int(cp.chest_touches[p]), f(cp.stance_m[p]), f(cp.towards_net_m[p]), f(cp.side_to_side_m[p]),
                     f(cm.knee_l[p], 0), f(cm.knee_r[p], 0), f(cm.hip_l[p], 0), f(cm.hip_r[p], 0), f(cm.trunk_lean[p], 0),
                     f'{back[p] * 100:.0f}%', f(cp.ball_height_m[p]), f(cp.ball_speed_in[p], 1), f(cp.net_dist_m[p]),
                     f'{cp.both_feet_planted_pct[p]:.0f}%'] for p in PLAYERS])

# ---------- points: where, which part, error chain
pe = pd.read_csv('data/points/point_end_verified.csv')
pts = pd.read_csv(R + 'points.csv')
land = pe[pe.x.notna() & pe.end_type.isin(['landed_in', 'out', 'landed_out'])]
def isr_left(r):
    # mirror so Israel's half is always the left one (x < 8)
    flip = (r.half_team == 'ISR') == (r.x > 8) if isinstance(r.half_team, str) else False
    return (16 - r.x, 8 - r.y) if flip else (r.x, r.y)


LAND = [dict(p=int(r.point), x=round(isr_left(r)[0], 2), y=round(isr_left(r)[1], 2), w=r.winner, e=r.end_type, sq=r.square if isinstance(r.square, str) else '',
             lt=n(r.last_touch) if isinstance(r.last_touch, str) else '') for r in land.itertuples()]
grid = {}
v = pe[(pe.end_type == 'landed_in') & pe.lane.notna()]
for t in ('ISR', 'BRA'):
    g = v[v.half_team == t]
    grid[t] = {f'{d}_{l}': int(((g.depth == d) & (g.lane == l)).sum()) for d in DEPTH for l in LANE}
S['end_types'] = table(['איך הסתיימה הנקודה', 'ישראל הפסידה', 'ברזיל הפסידה'],
                       [[lbl, int(((pe.loser == 'ISR') & (pe.end_type == k)).sum()), int(((pe.loser == 'BRA') & (pe.end_type == k)).sum())]
                        for k, lbl in [('landed_in', 'הכדור נחת בתוך המגרש שלה'), ('out', 'הכדור שלה יצא החוצה'),
                                       ('landed_out', 'נחת בחוץ'), ('net', 'רשת'), ('fault', 'עבירה'), ('excluded', 'לא צולם')]])
win = pts[pts.category == 'winning_touch']
S['win_parts'] = table(['שחקן', 'ווינרים'] + [PART[k] for k in ('foot', 'head', 'shoulder')],
                       [[n(p), int((win.last_player == p).sum())] +
                        [int(((win.last_player == p) & (win.last_part == k)).sum()) for k in ('foot', 'head', 'shoulder')]
                        for p in PLAYERS])
err = pts[pts.point_lost_by.notna()].copy()
err['tip'] = err.last_tip.fillna(0).astype(int)
S['err_rows'] = table(['נקודה', 'מערכה', 'מי נגע אחרון', 'איבר', 'פעולה', 'נגיעה מתוך 3', 'איך'],
                      [[int(r.point), int(r.set), n(r.point_lost_by), PART.get(r.last_part, r.last_part or '–'),
                        ACT.get(r.last_action, r.last_action), r.tip or '–',
                        {'out': 'אאוט', 'net': 'רשת', 'fault': 'עבירה', 'landed_on_own_side': 'נפל אצלנו'}.get(r.category, r.category)]
                       for r in err[err.loser == 'ISR'].itertuples()])
ec = []
for t in ('ISR', 'BRA'):
    e = err[err.loser == t]
    lost_total = int((pts.loser == t).sum())
    ec.append([n(t), lost_total, len(e), int((pts[(pts.loser == t)].category == 'winning_touch').sum()),
               int((e.tip == 1).sum()), int((e.tip == 2).sum()), int((e.tip == 3).sum()),
               ' · '.join(f'{PART[k]} {c}' for k, c in e.last_part.value_counts().items())])
S['err_summary'] = table(['קבוצה', 'נקודות שהפסידה', 'מטעות שלה', 'מווינר של היריב', 'בנגיעה 1', 'בנגיעה 2', 'בנגיעה 3',
                          'איבר בטעות'], ec)

# player position at point moments (lane x depth as the player sees it)
pp = pd.read_csv(A + 'point_positions.csv')
S['pp'] = table(['שחקן', 'תוצאה', 'ברשת', 'באמצע', 'בבסיס'],
                [[n(p), 'ניצחנו' if res == 'won' else 'הפסדנו'] +
                 [int(((pp.player == p) & (pp.result == res) & (pp.depth == d)).sum()) for d in ('net', 'mid', 'base')]
                 for p in PLAYERS for res in ('won', 'lost')])

# ---------- touches by part, success
tp = pd.read_csv(R + 'touches_by_action_part.csv')
parts = ['chest', 'foot', 'head', 'shoulder', 'knee', 'shin']
S['touch_parts'] = table(['שחקן', 'פעולה'] + [PART[k] for k in parts] + ['סה״כ'],
                         [[n(r.player), ACT[r.action]] + [int(getattr(r, k)) or '·' for k in parts] + [int(r.All)]
                          for r in tp[tp.team != 'All'].itertuples()] +
                         [['<b>הכל</b>', ''] + [f'<b>{int(tp[tp.team == "All"][k].iloc[0])}</b>' for k in parts] +
                          [f'<b>{int(tp[tp.team == "All"].All.iloc[0])}</b>']])
su = pd.read_csv(R + 'touch_success.csv')
S['success'] = table(['שחקן'] + [ACT[a] for a in ('reception', 'set', 'attack')] + ['הכל'],
                     [[n(p)] + [f"<b>{r.pct}%</b> <span class=\"mut\">{r.success}/{r.eligible} · {r.ci95}</span>"
                                for a in ('reception', 'set', 'attack', 'all')
                                for r in su[(su.player == p) & (su.action == a)].itertuples()] for p in PLAYERS])
t = pd.read_csv(R + 'touches.csv')
rec = t[t.action == 'reception']
ba = pd.read_csv(A + 'ball_after_touch.csv')
rows = []
for p in PLAYERS:
    for k in ('chest', 'foot', 'head'):
        g = rec[(rec.player == p) & (rec.body_part == k)]
        b = ba[(ba.player == p) & (ba.action == 'reception') & (ba.body_part == k)]
        if len(g):
            rows.append([n(p), PART[k], len(g), f'{g.success.mean() * 100:.0f}%', f(b.apex_m.median()),
                         f(b.air_time_s.median())])
S['reception'] = table(['שחקן', 'טכניקה', 'קבלות', 'לא איבד את הכדור', 'גובה שיא אחרי הקבלה (מ׳)',
                        'זמן עד הנגיעה הבאה (ש׳)'], rows)
rs = pd.read_csv(R + 'reception_stability.csv').set_index(['who', 'situation'])
S['rec_team'] = table(['', 'קבלת הגשה (מצולמת)', 'קבלת התקפה (הגנה)'],
                      [[n(w)] + [f"{rs.loc[(w, s)].pct}% <span class=\"mut\">{rs.loc[(w, s)].to_attack}/{rs.loc[(w, s)].n}</span>"
                                 if (w, s) in rs.index else '–' for s in ('serve', 'attack')]
                       for w in ['ISR', 'BRA'] + PLAYERS])

# ---------- set quality, ball after touch
sets = ba[ba.action == 'set']
sq = pd.read_csv(R + 'set_quality_sets.csv').set_index('player')
S['sets'] = table(['מגביה', 'הגבהות', 'הגיעו להתקפה', 'זמן באוויר עד ההתקפה (ש׳)', 'פיזור הזמן (סטיית תקן)',
                   'גובה שיא (מ׳)', 'טווח גובה 25–75%', 'מהן בחזה'],
                  [[n(p), int(sq.sets[p]), int(sq.partner_attack[p]), f(sq.air_s_median[p]),
                    f(sets[sets.player == p].air_time_s.std()), f(sets[sets.player == p].apex_m.median()),
                    f(sets[sets.player == p].apex_m.quantile(.25)) + '–' + f(sets[sets.player == p].apex_m.quantile(.75)),
                    int(((sets.player == p) & (sets.body_part == 'chest')).sum())] for p in PLAYERS])
S['apex_parts'] = table(['פעולה', 'איבר', 'נגיעות', 'גובה הכדור בנגיעה (מ׳)', 'גובה שיא אחרי (מ׳)', 'זמן עד הנגיעה הבאה (ש׳)'],
                        [[ACT[a], PART[k], len(g), f(g.height_at_touch_m.median()), f(g.apex_m.median()), f(g.air_time_s.median())]
                         for (a, k), g in ba.groupby(['action', 'body_part']) if len(g) >= 3])

# ---------- partner at the set, foot side at foot sets
sp = pd.read_csv(A + 'set_partner.csv')
S['partner'] = table(['מגביה → תוקף', 'הגבהות', 'מרחק (מ׳)', 'התוקף רחוק מהרשת יותר ב- (מ׳)', 'הפרש לאורך הרשת (מ׳)',
                      'התוקף בתנועה', 'מהירות התוקף (מ׳/ש׳)', 'זווית התוקף לרשת°', 'סטיית המבט מהכדור°'],
                     [[f'{n(s)} → {n(p)}', len(g), f(g.distance_m.median()), f(-g.partner_towards_net_m.median()),
                       f(g.partner_along_net_m.median()), f'{g.partner_moving.mean() * 100:.0f}%', f(g.partner_speed_ms.median()),
                       f(g.partner_angle_to_net_deg.median(), 0), f(g.partner_facing_vs_ball_deg.median(), 0)]
                      for (s, p), g in sp.groupby(['setter', 'partner'])])
fs = pd.read_csv(A + 'set_foot_side.csv')
S['footside'] = table(['שחקן', 'רגל', 'הגבהות ברגל', 'שמאל לגוף', 'מרכז', 'ימין לגוף', 'חציון: ימינה מהגוף (מ׳)',
                       'חציון: לפני הגוף (מ׳)'],
                      [[n(p), {'left': 'שמאל', 'right': 'ימין'}[ft], len(g), int((g.side == 'left').sum()),
                        int((g.side == 'middle').sum()), int((g.side == 'right').sum()), f(g.right_of_body_m.median()),
                        f(g.in_front_m.median())] for (p, ft), g in fs.groupby(['player', 'foot'])])

# ---------- stance (owner labels)
st = pd.read_csv(R + 'stance.csv')
S['stance'] = table(['שחקן', 'פעולה', 'עמד יציב', 'הגיע בתנועה', 'קפץ', 'תויגו'],
                    [[n(r.player), ACT[r.action], r.set, r.moving, r.jump, r.All] for r in st[st.player.isin(PLAYERS)].itertuples()])

# ---------- return to base
rb = pd.read_csv(A + 'return_to_base_summary.csv')
S['base'] = table(['שחקן', 'אחרי', 'פעמים', 'התרחק מהבסיס (מ׳)', 'זמן חזרה חציוני (ש׳)', 'חזר לפני סוף הראלי'],
                  [[n(r.player), ACT[r.action], r.n, f(r.dist_from_base), f(r.return_s_median), f'{r.returned_share * 100:.0f}%']
                   for r in rb.itertuples()])

# ---------- body direction
bd = pd.read_csv(A + 'body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'angle_to_net_deg'])
s = bd[bd.part == 'shoulders'].groupby(['player', 'frame']).angle_to_net_deg.mean().abs()
bins = [(0, 30, 'פנים לרשת (0–30°)'), (30, 60, 'באלכסון (30–60°)'), (60, 120, 'צידי (60–120°)'), (120, 181, 'גב לרשת (120–180°)')]
DIR = {p: [round(float(((s[p] >= a) & (s[p] < b)).mean()) * 100, 1) for a, b, _ in bins] for p in PLAYERS}
tbo = tb[tb.in_play]
S['facing_touch'] = table(['פעולה', 'פנים למצלמה', 'צידי', 'גב למצלמה'],
                          [[ACT[a]] + [f"{(g.facing == k).mean() * 100:.0f}%" for k in ('camera', 'side', 'away')]
                           for a, g in tbo.groupby('action') if a in ACT])

data = dict(HM=HM, LAND=LAND, GRID=grid, DIR=DIR, DIRBINS=[b[2] for b in bins], NAMES={p: n(p) for p in PLAYERS},
            PHASES=PHASE)
html = open(TEMPLATE, encoding='utf-8').read().replace('/*DATA*/', 'const D=' + json.dumps(data, separators=(',', ':')) + ';')
for k, val in S.items():
    html = html.replace('{{' + k + '}}', val)
assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html) // 1024, 'KB')
