"""Hebrew coach summary of the Graz 2022 final, every number backed by video.

Every number in the page is a button: it opens the touches (contact frame + skeleton + measured values) or points
(decisive moment + landing) it was computed from, each with its annotated rally clip at that moment. Numbers measured
on every frame (coverage, speed, partner distance, body direction) open the rally clips, which draw the same
measurement live. Media: coach_analysis/evidence/ (scripts/evidence/render_evidence.py).

Reads data/analysis/*, data/points/point_end_verified.csv and coach_analysis/recomputed_2026-11-24/*.
Writes coach_analysis/match_summary_he.html (the page expects evidence/ next to it when published).
usage: python scripts/site/build_summary.py [OUT.html]
"""
import json
import sys

import numpy as np
import pandas as pd

A = 'data/analysis/'
R = 'coach_analysis/recomputed_2026-11-24/'
EV = 'coach_analysis/evidence/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'coach_analysis/match_summary_he.html'
TEMPLATE = 'scripts/site/summary_template.html'

HE = {'Franklin': 'פרנקלין', 'Hiltinho': 'הילטיניו', 'Maor_Haas': 'מאור האס', 'Ron_Ben_Ishay': 'רון בן ישי',
      'ISR': 'ישראל', 'BRA': 'ברזיל'}
PLAYERS = ['Ron_Ben_Ishay', 'Maor_Haas', 'Franklin', 'Hiltinho']
PART = {'chest': 'חזה', 'foot': 'רגל', 'head': 'ראש', 'knee': 'ברך', 'shin': 'שוק', 'shoulder': 'כתף', 'unknown': 'לא ידוע'}
ACT = {'reception': 'קבלה', 'set': 'הגבהה', 'attack': 'התקפה', 'serve': 'הגשה'}
PHASE = {'all': 'כל המשחק', 'serve': 'הגשה', 'reception': 'קבלה', 'build': 'בניית מתקפה', 'attack': 'אחרי התקפה שלנו',
         'defence': 'הגנה'}
LANE = {'left': 'שמאל', 'mid': 'מרכז', 'right': 'ימין'}
DEPTH = {'net': 'רשת', 'mid': 'אמצע', 'base': 'בסיס'}
END = {'landed_in': 'נחת בפנים', 'out': 'אאוט', 'landed_out': 'נחת בחוץ', 'net': 'רשת', 'fault': 'עבירה', 'excluded': 'לא צולם'}


def n(p):
    return HE.get(p, p)


def f(x, d=2):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'


def E(text, kind, ids, title='', note=''):
    """A number that opens its evidence: kind t = touches, p = points, r = rally clips."""
    ids = [int(i) for i in ids if i is not None and not pd.isna(i)]
    if not ids:
        return str(text)
    t = title.replace('"', '&quot;'); nt = note.replace('"', '&quot;')
    return f'<button class="ev" data-k="{kind}" data-ids="{",".join(map(str, ids))}" data-t="{t}" data-n="{nt}">{text}</button>'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


S = {}
ix = json.load(open(EV + 'index.json'))
rallies = pd.read_csv('data/points/rallies_v3video.csv')
ALLR = rallies.rally.tolist()
NOTE_POS = 'בכל פריים בקליף: הריבוע והמהירות מתחת לרגלי השחקן, והמגרש הקטן למעלה מראה את מיקום ארבעת השחקנים.'
NOTE_PAIR = 'בכל פריים בקליף: הקו המקווקו בין שני בני הזוג ועליו המרחק במטרים.'
NOTE_DIR = 'בכל פריים בקליף: החץ הלבן מהרגליים מראה לאן פונות הכתפיים.'

t = pd.read_csv(R + 'touches.csv')          # one row per in-play touch, final action, tip, success, stance
t['touch'] = t.touch.astype(int)
o = pd.read_csv(A + 'touch_outcomes.csv')[['touch', 'quality']]
o['touch'] = o.touch.astype(int)
t = t.merge(o, on='touch', how='left')
ba = pd.read_csv(A + 'ball_after_touch.csv')
ba['touch'] = ba.touch.astype(int)


def T(**kw):
    m = pd.Series(True, index=t.index)
    for k, v in kw.items():
        m &= t[k].isin(v) if isinstance(v, (list, tuple, set)) else (t[k] == v)
    return t[m].touch.tolist()


# ---------- coverage, speed
cov = pd.read_csv(A + 'coverage_match_v3.csv').set_index('player')
spr = pd.read_csv(A + 'player_speed_rallies_v3.csv')
spd = spr.groupby('player').agg(dist=('distance', 'sum'), mean=('mean_speed', 'mean'), top=('top_speed', 'max'), sprint=('sprint_sec', 'sum'))
ph = pd.read_csv(A + 'phase_heatmaps_summary.csv')
allp = ph[ph.phase == 'all'].set_index('player')
rows = []
for p in PLAYERS:
    by_top = spr[spr.player == p].sort_values('top_speed', ascending=False).rally.tolist()
    by_spr = spr[(spr.player == p) & (spr.sprint_sec > 0)].sort_values('sprint_sec', ascending=False).rally.tolist()
    ti = f'{n(p)}: כל הראלים'
    rows.append([n(p)] + [E(v, 'r', ALLR, ti, NOTE_POS) for v in
                          (f(cov.area50[p], 1), f(cov.area90[p], 1), f(cov.hull_area[p], 1), f'{allp.left_half[p] * 100:.0f}%',
                           f(allp.from_net_m[p]), f(spd.dist[p], 0), f(spd['mean'][p]))] +
                [E(f(spd.top[p]), 'r', by_top, f'{n(p)}: ראלים לפי מהירות השיא, הגבוהה ראשונה', NOTE_POS),
                 E(f(spd.sprint[p], 0), 'r', by_spr, f'{n(p)}: ראלים עם ספרינט (מעל 3 מ׳/ש׳), הארוך ראשון', NOTE_POS)])
S['coverage'] = table(['שחקן', 'שטח ליבה 50% (מ״ר)', 'שטח 90% (מ״ר)', 'מעטפת כל המיקומים (מ״ר)', 'זמן בחצי השמאלי שלו',
                       'מרחק ממוצע מהרשת (מ׳)', 'מרחק כולל בראלים (מ׳)', 'מהירות ממוצעת (מ׳/ש׳)', 'מהירות שיא (מ׳/ש׳)',
                       'שניות ספרינט'], rows)

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
HM = dict(cell=hm['cell'] * 2, x0=hm['x0'], y0=hm['y0'], maps=maps)
pairs = pd.read_csv(A + 'phase_heatmaps_pairs.csv')
S['phase_table'] = table(['שלב'] + [f'{n(p)}<br><span class="mut">זמן בחצי השמאלי</span>' for p in PLAYERS] +
                         ['חפיפה ישראל', 'חפיפה ברזיל'],
                         [[PHASE[k]] + [E(f"{ph[(ph.player == p) & (ph.phase == k)].left_half.iloc[0] * 100:.0f}%", 'r', ALLR,
                                          f'{n(p)} · {PHASE[k]}', NOTE_POS) for p in PLAYERS] +
                          [E(f"{pairs[(pairs.team == tm) & (pairs.phase == k)].overlap.iloc[0]:.2f}", 'r', ALLR,
                             f'{n(tm)} · {PHASE[k]}', NOTE_POS) for tm in ('ISR', 'BRA')] for k in PHASE])
S['heat_ev'] = E('צפו בכל 60 הקליפים שמהם נבנו המפות', 'r', ALLR, 'כל הראלים', NOTE_POS)

# ---------- distances between partners
ff = pd.read_csv(A + 'formation_frames.csv')
cd = pd.read_csv(A + 'conceded_partner_distance.csv')
ts = pd.read_csv(A + 'team_spacing_v3.csv')
FORM = {'both_net': 'שניהם ברשת', 'front_back': 'אחד קדימה אחד אחורה', 'side_side': 'זה לצד זה'}
rows = []
for tm in ('ISR', 'BRA'):
    g = ff[ff.team == tm]
    fr = g.formation.value_counts(normalize=True)
    c = cd[cd.loser == tm]
    tt = ts[ts.defending_team == tm]

    def dfn(o):
        x = tt[tt.outcome == o]
        return E(f(x.distance.median()), 't', x.attack_touch.tolist(),
                 f'{n(tm)} בהגנה · התקפות של היריב', 'בתמונה: הקו המקווקו בין שני המגינים ועליו המרחק, ברגע ההתקפה.')
    rows.append([n(tm), E(f(g.distance.median()), 'r', ALLR, f'{n(tm)}: כל הראלים', NOTE_PAIR),
                 f(g.distance.quantile(.25)) + '–' + f(g.distance.quantile(.75)),
                 ' · '.join(E(f'{FORM[k]} {fr.get(k, 0) * 100:.0f}%', 'r', ALLR, f'{n(tm)}: מערך', NOTE_PAIR) for k in FORM),
                 E(f(c.distance_m.median()), 'p', c.sort_values('distance_m').rally.tolist(), f'{n(tm)} הפסידה: מרחק בין השחקנים',
                   'בתמונה: הרגע שבו נמדד המרחק, והמרחק עצמו בפאנל.') + f' <span class="mut">(n={len(c)})</span>',
                 dfn('returned'), dfn('kill'), dfn('failed_defence')])
S['spacing'] = table(['קבוצה', 'מרחק חציוני (מ׳)', 'טווח 25–75%', 'מערך', 'מרחק ברגע שהפסידו נקודה',
                      'בהגנה: החזירו התקפה', 'בהגנה: ספגו ווינר', 'בהגנה: נגעו ולא החזירו'], rows)

# ---------- chest
cp = pd.read_csv(A + 'chest_profile_by_player.csv').set_index('player')
cpt = pd.read_csv(A + 'chest_profile.csv')
tb = pd.read_csv(A + 'touch_body.csv')
c = tb[(tb.body_part == 'chest') & tb.in_play].copy()
for col in ['knee_l', 'knee_r', 'hip_l', 'hip_r', 'trunk_lean']:
    c.loc[~c[col + '_ok'].astype(bool), col] = np.nan
cm = c.groupby('player')[['knee_l', 'knee_r', 'hip_l', 'hip_r', 'trunk_lean']].median()
back = c.groupby('player').trunk_lean_dir.apply(lambda s: (s == 'back').mean())
rows = []
NOTE_CHEST = 'בתמונה: הקו הוורוד בין הקרסוליים ועליו המרחק, השלד בזום, והזוויות בפאנל.'
for p in PLAYERS:
    ids = cpt[cpt.player == p].sort_values('stance_m').touch.tolist()
    okid = lambda col: c[(c.player == p) & c[col].notna()].touch.tolist()
    ti = f'{n(p)}: נגיעות חזה'
    rows.append([n(p), E(int(cp.chest_touches[p]), 't', ids, ti, NOTE_CHEST)] +
                [E(f(cp[k][p]), 't', ids, ti + ' (מהצרה לרחבה)', NOTE_CHEST) for k in ('stance_m', 'towards_net_m', 'side_to_side_m')] +
                [E(f(cm[k][p], 0), 't', okid(k), ti + ' (רק כשהמפרק נראה היטב)', NOTE_CHEST) for k in ('knee_l', 'knee_r', 'hip_l', 'hip_r', 'trunk_lean')] +
                [E(f'{back[p] * 100:.0f}%', 't', ids, ti, NOTE_CHEST)] +
                [E(f(cp[k][p], 1 if k == 'ball_speed_in' else 2), 't', ids, ti, NOTE_CHEST) for k in ('ball_height_m', 'ball_speed_in', 'net_dist_m')] +
                [E(f'{cp.both_feet_planted_pct[p]:.0f}%', 't', ids, ti, NOTE_CHEST)])
S['chest'] = table(['שחקן', 'נגיעות חזה', 'מרחק בין הרגליים (מ׳)', 'מזה: לכיוון הרשת', 'מזה: מצד לצד',
                    'ברך שמאל°', 'ברך ימין°', 'ירך שמאל°', 'ירך ימין°', 'הטיית גו°', 'נשען אחורה', 'גובה הכדור (מ׳)',
                    'מהירות הכדור הנכנס (מ׳/ש׳)', 'מרחק מהרשת (מ׳)', 'שתי רגליים נטועות'], rows)
S['chest46'] = E('נקודה 46', 'p', [46], 'נקודה 46', '')

# ---------- points: where, which part, error chain
pe = pd.read_csv('data/points/point_end_verified.csv')
pts = pd.read_csv(R + 'points.csv')
land = pe[pe.x.notna() & pe.end_type.isin(['landed_in', 'out', 'landed_out'])]


def isr_left(r):
    flip = (r.half_team == 'ISR') == (r.x > 8) if isinstance(r.half_team, str) else False
    return (16 - r.x, 8 - r.y) if flip else (r.x, r.y)


LAND = [dict(p=int(r.point), x=round(isr_left(r)[0], 2), y=round(isr_left(r)[1], 2), w=r.winner, e=r.end_type,
             lt=n(r.last_touch) if isinstance(r.last_touch, str) else '') for r in land.itertuples()]
grid = {}
v = pe[(pe.end_type == 'landed_in') & pe.lane.notna()]
for tm in ('ISR', 'BRA'):
    g = v[v.half_team == tm]
    grid[tm] = {f'{d}_{l}': g[(g.depth == d) & (g.lane == l)].point.astype(int).tolist() for d in DEPTH for l in LANE}
S['end_types'] = table(['איך הסתיימה הנקודה', 'ישראל הפסידה', 'ברזיל הפסידה'],
                       [[lbl] + [E(int(((pe.loser == tm) & (pe.end_type == k)).sum()), 'p',
                                   pe[(pe.loser == tm) & (pe.end_type == k)].point.tolist(), f'{n(tm)} הפסידה · {lbl}')
                                 for tm in ('ISR', 'BRA')]
                        for k, lbl in [('landed_in', 'הכדור נחת בתוך המגרש שלה'), ('out', 'הכדור שלה יצא החוצה'),
                                       ('landed_out', 'נחת בחוץ'), ('net', 'רשת'), ('fault', 'עבירה'), ('excluded', 'לא צולם')]])
win = pts[pts.category == 'winning_touch']
S['win_parts'] = table(['שחקן', 'ווינרים'] + [PART[k] for k in ('foot', 'head', 'shoulder')],
                       [[n(p), E(int((win.last_player == p).sum()), 'p', win[win.last_player == p].point.tolist(), f'ווינרים של {n(p)}')] +
                        [E(int(((win.last_player == p) & (win.last_part == k)).sum()), 'p',
                           win[(win.last_player == p) & (win.last_part == k)].point.tolist(), f'ווינרים של {n(p)} ב{PART[k]}')
                         for k in ('foot', 'head', 'shoulder')] for p in PLAYERS])
err = pts[pts.point_lost_by.notna()].copy()
err['tip'] = err.last_tip.fillna(0).astype(int)
CAT = {'out': 'אאוט', 'net': 'רשת', 'fault': 'עבירה', 'landed_on_own_side': 'נפל אצלנו'}
S['err_rows'] = table(['נקודה', 'מערכה', 'מי נגע אחרון', 'איבר', 'פעולה', 'נגיעה מתוך 3', 'איך'],
                      [[E(int(r.point), 'p', [r.point], f'נקודה {int(r.point)}'), int(r.set), n(r.point_lost_by),
                        PART.get(r.last_part, r.last_part or '–'), ACT.get(r.last_action, r.last_action), r.tip or '–',
                        CAT.get(r.category, r.category)] for r in err[err.loser == 'ISR'].itertuples()])
ec = []
for tm in ('ISR', 'BRA'):
    e = err[err.loser == tm]
    lp = pts[pts.loser == tm]
    ti = f'{n(tm)} הפסידה'
    ec.append([n(tm), E(len(lp), 'p', lp.point.tolist(), ti), E(len(e), 'p', e.point.tolist(), ti + ' · טעויות שלה'),
               E(int((lp.category == 'winning_touch').sum()), 'p', lp[lp.category == 'winning_touch'].point.tolist(), ti + ' · ווינר של היריב')] +
              [E(int((e.tip == k).sum()), 'p', e[e.tip == k].point.tolist(), ti + f' · טעות בנגיעה {k}') for k in (1, 2, 3)] +
              [' · '.join(E(f'{PART[k]} {cnt}', 'p', e[e.last_part == k].point.tolist(), ti + f' · טעות ב{PART[k]}')
                          for k, cnt in e.last_part.value_counts().items())])
S['err_summary'] = table(['קבוצה', 'נקודות שהפסידה', 'מטעות שלה', 'מווינר של היריב', 'בנגיעה 1', 'בנגיעה 2', 'בנגיעה 3',
                          'איבר בטעות'], ec)
pp = pd.read_csv(A + 'point_positions.csv')
S['pp'] = table(['שחקן', 'תוצאה', 'ברשת', 'באמצע', 'בבסיס'],
                [[n(p), 'ניצחנו' if res == 'won' else 'הפסדנו'] +
                 [E(int(((pp.player == p) & (pp.result == res) & (pp.depth == d)).sum()), 'p',
                    pp[(pp.player == p) & (pp.result == res) & (pp.depth == d)].rally.tolist(),
                    f'{n(p)} {DEPTH[d]} ברגע סוף הנקודה', 'בתמונה: הריבוע שמתחת לרגלי השחקן ברגע שנמדד.')
                  for d in ('net', 'mid', 'base')] for p in PLAYERS for res in ('won', 'lost')])

# ---------- touches by part, success
parts = ['chest', 'foot', 'head', 'shoulder', 'knee', 'shin']
rows = []
for p in PLAYERS:
    for a in ('reception', 'set', 'attack'):
        ids = T(player=p, action=a)
        rows.append([n(p), ACT[a]] + [E(len(T(player=p, action=a, body_part=k)) or '·', 't', T(player=p, action=a, body_part=k),
                                        f'{n(p)} · {ACT[a]} · {PART[k]}') for k in parts] + [E(len(ids), 't', ids, f'{n(p)} · {ACT[a]}')])
rows.append(['<b>הכל</b>', ''] + [E(f'<b>{len(T(body_part=k))}</b>', 't', T(body_part=k), f'כל הנגיעות ב{PART[k]}') for k in parts] +
            [f'<b>{len(t)}</b>'])
S['touch_parts'] = table(['שחקן', 'פעולה'] + [PART[k] for k in parts] + ['סה״כ'], rows)

su = pd.read_csv(R + 'touch_success.csv')
rows = []
for p in PLAYERS:
    r_ = [n(p)]
    for a in ('reception', 'set', 'attack', 'all'):
        r = su[(su.player == p) & (su.action == a)].iloc[0]
        g = t[(t.player == p) & ((t.action == a) if a != 'all' else t.action.isin(ACT))].sort_values('success')
        r_.append(E(f'<b>{r.pct}%</b> <span class="mut">{r.success}/{r.eligible} · {r.ci95}</span>', 't', g.touch.tolist(),
                    f'{n(p)} · {ACT.get(a, "הכל")} (הכושלות ראשונות)', 'בפאנל: התוצאה של כל נגיעה.'))
    rows.append(r_)
S['success'] = table(['שחקן'] + [ACT[a] for a in ('reception', 'set', 'attack')] + ['הכל'], rows)

att = set(t[t.action == 'attack'].pkey)
rec = t[(t.action == 'reception') & (t.tip == 1)].copy()
rec['situation'] = np.where(rec.poss == 1, 'serve', 'attack')
rec['to_attack'] = rec.pkey.isin(att)
rows = []
for w in ['ISR', 'BRA'] + PLAYERS:
    r_ = [n(w)]
    for s_ in ('serve', 'attack'):
        g = rec[((rec.team == w) | (rec.player == w)) & (rec.situation == s_)].sort_values('to_attack')
        r_.append(E(f"{g.to_attack.mean() * 100:.0f}% <span class=\"mut\">{int(g.to_attack.sum())}/{len(g)}</span>", 't', g.touch.tolist(),
                    f'{n(w)} · קבלת {"הגשה" if s_ == "serve" else "התקפה"} (אלה שלא הגיעו להתקפה ראשונות)') if len(g) else '–')
    rows.append(r_)
S['rec_team'] = table(['', 'קבלת הגשה (מצולמת)', 'קבלת התקפה (הגנה)'], rows)
rows = []
for p in PLAYERS:
    for k in ('chest', 'foot', 'head'):
        g = t[(t.action == 'reception') & (t.player == p) & (t.body_part == k)].sort_values('success')
        b = ba[ba.touch.isin(g.touch)]
        if len(g):
            ti = f'{n(p)} · קבלה ב{PART[k]}'
            rows.append([n(p), PART[k]] + [E(x, 't', g.touch.tolist(), ti) for x in
                                           (len(g), f'{g.success.mean() * 100:.0f}%', f(b.apex_m.median()), f(b.air_time_s.median()))])
S['reception'] = table(['שחקן', 'טכניקה', 'קבלות', 'לא איבד את הכדור', 'גובה שיא אחרי הקבלה (מ׳)', 'זמן עד הנגיעה הבאה (ש׳)'], rows)

# ---------- sets, ball after touch
sets = ba[ba.action == 'set']
sq = pd.read_csv(R + 'set_quality_sets.csv').set_index('player')
rows = []
for p in PLAYERS:
    g = sets[sets.player == p].sort_values('apex_m')
    ids, ti = g.touch.tolist(), f'הגבהות של {n(p)} (מהנמוכה לגבוהה)'
    rows.append([n(p)] + [E(x, 't', ids, ti) for x in
                          (int(sq.sets[p]), int(sq.partner_attack[p]), f(sq.air_s_median[p]), f(g.air_time_s.std()), f(g.apex_m.median()),
                           f(g.apex_m.quantile(.25)) + '–' + f(g.apex_m.quantile(.75)))] +
                [E(int((g.body_part == 'chest').sum()), 't', g[g.body_part == 'chest'].touch.tolist(), f'הגבהות חזה של {n(p)}')])
S['sets'] = table(['מגביה', 'הגבהות', 'הגיעו להתקפה', 'זמן באוויר עד ההתקפה (ש׳)', 'פיזור הזמן (סטיית תקן)', 'גובה שיא (מ׳)',
                   'טווח גובה 25–75%', 'מהן בחזה'], rows)
S['apex_parts'] = table(['פעולה', 'איבר', 'נגיעות', 'גובה הכדור בנגיעה (מ׳)', 'גובה שיא אחרי (מ׳)', 'זמן עד הנגיעה הבאה (ש׳)'],
                        [[ACT[a], PART[k]] + [E(x, 't', g.sort_values('apex_m').touch.tolist(), f'{ACT[a]} ב{PART[k]}')
                                              for x in (len(g), f(g.height_at_touch_m.median()), f(g.apex_m.median()), f(g.air_time_s.median()))]
                         for (a, k), g in ba.groupby(['action', 'body_part']) if len(g) >= 3])
sp = pd.read_csv(A + 'set_partner.csv')
NOTE_P = 'בתמונה: הקו המקווקו בין המגביה לשותף, והחץ הלבן מראה לאן השותף פונה. הערכים בפאנל.'
S['partner'] = table(['מגביה → תוקף', 'הגבהות', 'מרחק (מ׳)', 'התוקף רחוק מהרשת יותר ב- (מ׳)', 'הפרש לאורך הרשת (מ׳)',
                      'התוקף בתנועה', 'מהירות התוקף (מ׳/ש׳)', 'זווית התוקף לרשת°', 'סטיית המבט מהכדור°'],
                     [[f'{n(s_)} → {n(p)}'] + [E(x, 't', g.touch.tolist(), f'{n(s_)} מגביה ל{n(p)}', NOTE_P) for x in
                                              (len(g), f(g.distance_m.median()), f(-g.partner_towards_net_m.median()),
                                               f(g.partner_along_net_m.median()), f'{g.partner_moving.mean() * 100:.0f}%',
                                               f(g.partner_speed_ms.median()), f(g.partner_angle_to_net_deg.median(), 0),
                                               f(g.partner_facing_vs_ball_deg.median(), 0))]
                      for (s_, p), g in sp.groupby(['setter', 'partner'])])
fs = pd.read_csv(A + 'set_foot_side.csv')
NOTE_F = 'בזום: השלד והכדור. בפאנל: כמה ימינה (+) או שמאלה (−) מהגוף הייתה הרגל.'
S['footside'] = table(['שחקן', 'רגל', 'הגבהות ברגל', 'שמאל לגוף', 'מרכז', 'ימין לגוף', 'חציון: ימינה מהגוף (מ׳)', 'חציון: לפני הגוף (מ׳)'],
                      [[n(p), {'left': 'שמאל', 'right': 'ימין'}[ft], E(len(g), 't', g.touch.tolist(), f'הגבהות ברגל של {n(p)}', NOTE_F)] +
                       [E(int((g.side == s_).sum()), 't', g[g.side == s_].touch.tolist(), f'{n(p)}: הרגל {LANE.get(s_, "במרכז")}', NOTE_F)
                        for s_ in ('left', 'middle', 'right')] +
                       [E(f(g.right_of_body_m.median()), 't', g.touch.tolist(), f'הגבהות ברגל של {n(p)}', NOTE_F),
                        E(f(g.in_front_m.median()), 't', g.touch.tolist(), f'הגבהות ברגל של {n(p)}', NOTE_F)]
                       for (p, ft), g in fs.groupby(['player', 'foot'])])

# ---------- stance (owner labels)
STN = {'set': 'עמד יציב', 'moving': 'הגיע בתנועה', 'jump': 'קפץ'}
rows = []
for p in PLAYERS:
    for a in ('reception', 'set', 'attack'):
        g = t[(t.player == p) & (t.action == a) & t.stance.notna()]
        if len(g):
            rows.append([n(p), ACT[a]] + [E(int((g.stance == k).sum()), 't', g[g.stance == k].touch.tolist(), f'{n(p)} · {ACT[a]} · {STN[k]}')
                                          for k in ('set', 'moving', 'jump')] + [E(len(g), 't', g.touch.tolist(), f'{n(p)} · {ACT[a]} · תיוגים')])
S['stance'] = table(['שחקן', 'פעולה', 'עמד יציב', 'הגיע בתנועה', 'קפץ', 'תויגו'], rows)

# ---------- return to base
rbt = pd.read_csv(A + 'return_to_base.csv')
rb = pd.read_csv(A + 'return_to_base_summary.csv')
NOTE_B = 'בפאנל: זמן החזרה של הנגיעה. בקליף, מהנגיעה והלאה, אפשר לראות את השחקן חוזר.'
S['base'] = table(['שחקן', 'אחרי', 'פעמים', 'התרחק מהבסיס (מ׳)', 'זמן חזרה חציוני (ש׳)', 'חזר לפני סוף הראלי'],
                  [[n(r.player), ACT[r.action]] + [E(x, 't', rbt[(rbt.player == r.player) & (rbt.action == r.action)].sort_values('return_s').touch.tolist(),
                                                     f'{n(r.player)} · חזרה לבסיס אחרי {ACT[r.action]}', NOTE_B)
                                                   for x in (r.n, f(r.dist_from_base), f(r.return_s_median), f'{r.returned_share * 100:.0f}%')]
                   for r in rb.itertuples()])

# ---------- body direction
bd = pd.read_csv(A + 'body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'angle_to_net_deg'])
s = bd[bd.part == 'shoulders'].groupby(['player', 'frame']).angle_to_net_deg.mean().abs()
bins = [(0, 30, 'פנים לרשת (0–30°)'), (30, 60, 'באלכסון (30–60°)'), (60, 120, 'צידי (60–120°)'), (120, 181, 'גב לרשת (120–180°)')]
DIR = {p: [round(float(((s[p] >= a) & (s[p] < b)).mean()) * 100, 1) for a, b, _ in bins] for p in PLAYERS}
tbo = tb[tb.in_play]
FAC = {'camera': 'פנים למצלמה', 'side': 'צידי', 'away': 'גב למצלמה'}
S['facing_touch'] = table(['פעולה'] + list(FAC.values()),
                          [[ACT[a]] + [E(f"{(g.facing == k).mean() * 100:.0f}%", 't', g[g.facing == k].touch.astype(int).tolist(), f'{ACT[a]} · {FAC[k]}')
                                       for k in FAC] for a, g in tbo.groupby('action') if a in ACT])
S['dir_ev'] = E('צפו בקליפים (החץ הלבן)', 'r', ALLR, 'כיווני גוף', NOTE_DIR)

# ---------- key findings links
S['k_roles'] = E('מפות החום', 'r', ALLR, 'חלוקת תפקידים', NOTE_POS)
S['k_rec'] = E('77%', 't', rec[(rec.team == 'ISR') & (rec.situation == 'attack')].sort_values('to_attack').touch.tolist(), 'ישראל · קבלת התקפה')
S['k_rec_b'] = E('89%', 't', rec[(rec.team == 'BRA') & (rec.situation == 'attack')].sort_values('to_attack').touch.tolist(), 'ברזיל · קבלת התקפה')
S['k_err'] = E('16 נקודות', 'p', err[err.loser == 'ISR'].point.tolist(), 'טעויות של ישראל')
S['k_ronrec'] = E('2.3 מ׳', 't', T(player='Ron_Ben_Ishay', action='reception', body_part='foot'), 'רון · קבלה ברגל')
S['k_ronset'] = E('39 מ־40', 't', sets[sets.player == 'Ron_Ben_Ishay'].sort_values('apex_m').touch.tolist(), 'הגבהות של רון')
fd = ts[(ts.defending_team == 'ISR') & (ts.outcome == 'failed_defence')]
S['k_def'] = E('3.65 מ׳', 't', fd.attack_touch.tolist(), 'ישראל · נגעו ולא החזירו', 'בתמונה: הקו המקווקו בין רון למאור.')

# ---------- evidence index for the page
TL = {int(r.touch): f'נגיעה {int(r.touch)} · {n(r.player)} · {PART.get(r.body_part, r.body_part)} · {ACT.get(r.action, r.action)}'
      for r in t.itertuples()}
PL_ = {}
for r in pe.itertuples():
    PL_[int(r.point)] = f'נקודה {int(r.point)} · {n(r.winner)} זכתה · {END.get(r.end_type, r.end_type)}'
EVI = dict(touches={k: [v['rally'], v['t'], TL.get(int(k), f'נגיעה {k}')] for k, v in ix['touches'].items()},
           points={k: [v['rally'], v['t'], PL_.get(int(k), f'נקודה {k}')] for k, v in ix['points'].items()},
           clips={k: [v['start'], v['end']] for k, v in ix['clips'].items()})
missing = [x for x in t.touch if str(x) not in ix['touches']]
print('touches without a still:', missing)

data = dict(HM=HM, LAND=LAND, GRID=grid, DIR=DIR, DIRBINS=[b[2] for b in bins], NAMES={p: n(p) for p in PLAYERS},
            PHASES=PHASE, EV=EVI, NOTE_DIR=NOTE_DIR)
html = open(TEMPLATE, encoding='utf-8').read().replace('/*DATA*/', 'const D=' + json.dumps(data, ensure_ascii=False, separators=(',', ':')) + ';')
for k, val in S.items():
    html = html.replace('{{' + k + '}}', val)
assert '{{' not in html, html[html.index('{{'):html.index('{{') + 40]
open(OUT, 'w', encoding='utf-8').write(html)
print('written', OUT, len(html) // 1024, 'KB')
