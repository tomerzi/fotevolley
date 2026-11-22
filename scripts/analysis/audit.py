"""Critical audit of the data behind the court-story page: for every data layer, independent checks (cross-source
agreement, physics, game rules, hand marks, internal consistency) and an estimated accuracy %, plus the statistical
precision (95% interval) of the headline numbers. Writes data/analysis/audit.json and site/court_story/audit.json.

Accuracy % = estimated share of the records that are right (measurement validity). Precision = how much a headline
rate could move by chance with this few points (Wilson 95% interval). A number can be measured right and still be
statistically weak; both are shown.
usage: python scripts/analysis/audit.py
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
FPS = 25
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
R = lambda v, n=1: None if v is None or not np.isfinite(v) else round(float(v), n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (int(round(100 * (c - h))), int(round(100 * (c + h))))


audit = {}


def layer(key, name, accuracy, checks, notes, precision=None):
    grade = 'A' if accuracy >= 95 else 'B' if accuracy >= 85 else 'C' if accuracy >= 70 else 'D'
    audit[key] = dict(name=name, accuracy=R(accuracy, 0), grade=grade, checks=checks, notes=notes, precision=precision or [])
    print(f'\n[{grade} {accuracy:5.1f}%] {name}')
    for c in checks:
        print(f"   {'PASS' if c['ok'] else 'WARN'}  {c['what']}: {c['result']}")
    for p in precision or []:
        print(f"   ±  {p['what']}: {p['value']} (95% {p['ci']})")


def chk(what, result, ok, weight=None):
    return dict(what=what, result=result, ok=bool(ok))


ral = pd.read_csv('data/points/rallies_v2.csv')
ev = pd.read_csv('data/points/point_events_FINAL_VERIFIED.csv')
act = pd.read_csv('data/analysis/touch_actions_v3.csv')
inplay = np.zeros(int(ral.end_frame.max()) + 2, bool)
rid = np.full(len(inplay), -1)
for x in ral.itertuples():
    inplay[x.start_frame:x.end_frame + 1] = True; rid[x.start_frame:x.end_frame + 1] = x.rally

# ---------------------------------------------------------------- score / momentum
ev = ev.sort_values('frame').reset_index(drop=True)
w_ev = ev.country.to_numpy()
agree = int((ral.winner.to_numpy() == w_ev[:len(ral)]).sum()) if len(ev) == len(ral) else None
s_ok = int(((ral.ISR_score_after.to_numpy() == ev.ISR_score_after.to_numpy()) & (ral.BRA_score_after.to_numpy() == ev.BRA_score_after.to_numpy())).sum())
order_ok = int((ral.end_frame.to_numpy() <= ev.frame.to_numpy()).sum())
exact = int((ev.timestamp_precision == 'EXACT_SCOREBOARD_CHANGE').sum())
# score must move by exactly one point for the winner each rally (within a set)
step_ok = 0
for s_, g in ral.groupby('set'):
    pi = np.r_[0, g.ISR_score_after.to_numpy()[:-1]]; pb = np.r_[0, g.BRA_score_after.to_numpy()[:-1]]
    di, db = g.ISR_score_after.to_numpy() - pi, g.BRA_score_after.to_numpy() - pb
    step_ok += int((((g.winner == 'ISR') & (di == 1) & (db == 0)) | ((g.winner == 'BRA') & (db == 1) & (di == 0))).sum())
# rallies 49-52: the broadcast scoreboard first gave BRA a 15th point (frame 13160), then took it back and gave ISR two
# (frame 13460: 5-14). Checked frame by frame in the video: rallies_v2 follows the corrected board; the events file
# kept the uncorrected one. These two mismatches are therefore explained.
CORRECTED = {49, 52}
mis = set(ral.rally[ral.winner.to_numpy() != w_ev[:len(ral)]])
agree_c = len(ral) - len(mis - CORRECTED)
layer('score', 'תוצאה ומומנטום', 100 * min(agree_c, step_ok) / len(ral), [
    chk('מנצח כל נקודה מול שינוי לוח התוצאות', f'{agree}/{len(ral)} תואמים; 2 השונים (49, 52) – תיקון של המזכירות בלוח, נבדק בווידאו', agree_c == len(ral)),
    chk('התוצאה אחרי כל נקודה מול הלוח', f'{s_ok}/{len(ral)} (אותו תיקון)', s_ok >= len(ral) - 3),
    chk('כל נקודה מוסיפה בדיוק נקודה אחת למנצח', f'{step_ok}/{len(ral)}', step_ok == len(ral)),
    chk('הנקודה נגמרת לפני שהלוח מתעדכן', f'{order_ok}/{len(ral)}', order_ok >= len(ral) - 2),
    chk('רגע עדכון הלוח מדויק (לא בקבוצה)', f'{exact}/{len(ev)}', exact >= 55)],
    'מקור עצמאי: לוח התוצאות בשידור. לפי הצפייה שלך יש שתי נקודות שהלוח הרשמי סותר: 34 (שחקן ברזיל עבר את קו הרשת, הלוח: ברזיל) ו־49 (הכדור נפל בצד של ישראל, הלוח אחרי תיקון: ישראל). נשאר לפי הלוח.')

# ---------------------------------------------------------------- side-out (needs the server)
sv = act[act.action == 'serve'].groupby('rally').team.first()
ral['server'] = ral.rally.map(sv)
rule_ok = 0; rule_n = 0
for s_, g in ral.groupby('set'):
    g = g.sort_values('rally')
    prev = g.winner.shift(1)
    m = prev.notna()
    rule_n += int(m.sum()); rule_ok += int((g.server[m] == prev[m]).sum())
so = {}
prec = []
for tm in ('ISR', 'BRA'):
    rec = ral[ral.server != tm]; k = int((rec.winner == tm).sum()); n = len(rec)
    so[tm] = (k, n); prec.append(dict(what=f'זכייה בקבלה {"ישראל" if tm == "ISR" else "ברזיל"}', value=f'{round(100 * k / n)}% ({k}/{n})', ci=f'{wilson(k, n)[0]}–{wilson(k, n)[1]}%'))
acc = 100 * rule_ok / rule_n
layer('sideout', 'קבלת הגשה', acc, [
    chk('המגיש = מי שזכה בנקודה הקודמת (חוק המשחק)', f'{rule_ok}/{rule_n}', rule_ok == rule_n),
    chk('זהות המגיש הוסקה ולא נצפתה ישירות', 'ההגשה עצמה לרוב מחוץ לפריים', False)],
    'אם המגיש נכון, גם הקבלה נכונה; מנצח הנקודה מאומת מול הלוח.', prec)

# ---------------------------------------------------------------- player positions (heat, spots, lines, gap, engine)
p = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['v2_frame', 'player', 'team', 'x', 'y', 'occluded', 'outlier', 'reset', 'jump', 'speed'])
p = p[p.v2_frame < len(inplay)]
pp = p[inplay[p.v2_frame.to_numpy()]].copy()
pp['rally'] = rid[pp.v2_frame.to_numpy()]
n_pp = len(pp)
occl = pp.occluded.mean() * 100; outl = pp.outlier.mean() * 100
isr_left = pp[pp.team == 'ISR'].groupby('rally').x.mean().lt(8)
pp['own_left'] = np.where(pp.team == 'ISR', pp.rally.map(isr_left), ~pp.rally.map(isr_left))
own = ((pp.own_left & (pp.x < 8.3)) | (~pp.own_left & (pp.x > 7.7))).mean() * 100
inb = ((pp.x > -3) & (pp.x < 19) & (pp.y > -3) & (pp.y < 11)).mean() * 100
fast = (pp.speed > 9).mean() * 100
# teammates on top of each other = possible identity swap
piv = pp.pivot_table(index='v2_frame', columns='player', values=['x', 'y'])
d_isr = np.hypot(piv['x']['Maor_Haas'] - piv['x']['Ron_Ben_Ishay'], piv['y']['Maor_Haas'] - piv['y']['Ron_Ben_Ishay'])
d_bra = np.hypot(piv['x']['Franklin'] - piv['x']['Hiltinho'], piv['y']['Franklin'] - piv['y']['Hiltinho'])
close = (pd.concat([d_isr, d_bra]) < 0.4).mean() * 100
# ground scale: metres per pixel at the far / near sideline (camera model) -> position error from ~2 px tracking noise
import sys; sys.path.insert(0, 'scripts/ball_height')
from traj import Camera
cam = Camera()
def m_per_px(x, y):
    a = cam.project(np.array([[x, y, 0.0], [x + 0.1, y, 0.0], [x, y + 0.1, 0.0]]))
    return 0.1 / min(np.linalg.norm(a[1] - a[0]), np.linalg.norm(a[2] - a[0]))
PX = 10  # assumed error of the feet point: ankle keypoints vs the true ground contact, occlusion
far_e, near_e = PX * m_per_px(4, 0.5), PX * m_per_px(4, 7.5)
# jitter: distance at full rate vs every 5th frame
g = pp[pp.player == 'Maor_Haas'].sort_values('v2_frame')
d1 = np.nansum(np.hypot(np.diff(g.x), np.diff(g.y))[np.diff(g.v2_frame) == 1])
g5 = g[g.v2_frame % 5 == 0]
d5 = np.nansum(np.hypot(np.diff(g5.x), np.diff(g5.y))[np.diff(g5.v2_frame) == 5])
jit = (d1 / d5 - 1) * 100
acc_pos = 100 - (occl * 0.5 + outl + (100 - own) + fast + close)
layer('positions', 'מיקומי שחקנים (מפת חום, מיקום ממוצע, מרחק מהקווים)', acc_pos, [
    chk('שחקן בחצי של הקבוצה שלו', f'{own:.1f}% מהפריימים', own > 97),
    chk('בתוך המגרש עם 3 מ׳ שוליים', f'{inb:.1f}%', inb > 99),
    chk('מהירות לא אנושית (>9 מ׳/ש׳)', f'{fast:.2f}% מהפריימים', fast < 0.5),
    chk('שני בני זוג באותה נקודה (חשד להחלפת זהות)', f'{close:.2f}%', close < 0.5),
    chk('פריימים מוסתרים (משוערים)', f'{occl:.1f}%', occl < 10),
    chk('חריגים שתוקנו', f'{outl:.2f}%', outl < 2),
    chk('דיוק מיקום (בהנחת 10 פיקסלים שגיאה בנקודת הרגליים)', f'±{near_e:.2f} מ׳ בצד הקרוב, ±{far_e:.2f} מ׳ בצד הרחוק', far_e < 0.3)],
    'הדיוק של מיקום יחיד הוא כעשרות ס״מ; ממוצעים ומפות חום על אלפי פריימים יציבים הרבה יותר.')
layer('engine', 'ריצה (מרחק ומהירות שיא)', max(0, 100 - abs(jit) - 5), [
    chk('ניפוח מרחק מרעש (25 מול 5 פריימים לשנייה)', f'+{jit:.1f}%', abs(jit) < 10),
    chk('מהירות שיא 28–31 קמ״ש', 'גבוה לחול; עלול לכלול קפיצה של המעקב', False)],
    'מרחקים טובים כהשוואה בין שחקנים; מהירות שיא היא הערך הכי פגיע לרעש.')

# ---------------------------------------------------------------- touches
t = pd.read_csv('data/contacts/touches_master.csv')
tp = t[t.kind == 'player'].copy()
n_t = len(tp)
pl_ok = tp.auto_player_ok.mean() * 100; part_ok = tp.auto_part_ok.mean() * 100
flagged = int(tp.action_check.sum())
# possession rule: max 3 touches per side
over3 = int((tp.possession_len > 3).sum())
# the toucher is on his team's side of the net (ball x on own half)
tp['rally'] = tp.rally.astype(int)
tp['left'] = tp.rally.map(isr_left)
side_ok = []
for x in tp.itertuples():
    if not np.isfinite(x.ball_court_x_m) or not isinstance(x.player, str):
        continue
    own_left = x.left if TEAM[x.player] == 'ISR' else not x.left
    side_ok.append((x.ball_court_x_m < 8.6) if own_left else (x.ball_court_x_m > 7.4))
side_pct = 100 * np.mean(side_ok)
# ball near the toucher's feet position (xy within 1.5 m)
pidx = p.drop_duplicates(['v2_frame', 'player']).set_index(['v2_frame', 'player'])
near = []
for x in tp.itertuples():
    if not isinstance(x.player, str) or not np.isfinite(x.ball_court_x_m):
        continue
    try:
        q = pidx.loc[(x.v2_frame, x.player)]
    except KeyError:
        continue
    near.append(np.hypot(q.x - x.ball_court_x_m, q.y - x.ball_court_y_m))
near = np.array(near)
layer('touches', 'מגעים בכדור (מי, באיזה איבר, איזו פעולה)', min(pl_ok, side_pct) - (100 - part_ok) * 0.3, [
    chk('זיהוי אוטומטי של השחקן מול התיוג הסופי', f'{pl_ok:.0f}% תואם', pl_ok > 95),
    chk('זיהוי אוטומטי של איבר הגוף מול התיוג הסופי', f'{part_ok:.0f}% תואם (התיוג הסופי תוקן ידנית)', part_ok > 90),
    chk('לכל היותר 3 מגעים לקבוצה (חוק)', 'אין הפרות (נקודה 39 תוקנה לפי הסימון שלך: נגיעה של מאור בפריים 10506)' if over3 == 0 else f'{over3} הפרות', over3 == 0),
    chk('הכדור בצד של הקבוצה הנוגעת', f'{side_pct:.1f}%', side_pct > 97),
    chk('הכדור עד 1.5 מ׳ מהשחקן ברגע המגע', f'{100 * np.mean(near < 1.5):.0f}% (חציון {np.median(near):.2f} מ׳)', np.mean(near < 1.5) > 0.9),
    chk('מגעים שסומנו לבדיקה', f'{flagged}', flagged < 15)],
    'התיוג הסופי עבר תיקון ידני; האחוז כאן הוא הערכה שמרנית לפי ההסכמה עם הזיהוי האוטומטי והחוקים.')

# ---------------------------------------------------------------- attacks
ta = tp[tp.action == 'attack']
res = ta.action_subtype.value_counts().to_dict()
k_i = int(ta[(ta.team == 'ISR') & ta.action_subtype.isin(['kill', 'failed_defence'])].shape[0]); n_i = int((ta.team == 'ISR').sum())
k_b = int(ta[(ta.team == 'BRA') & ta.action_subtype.isin(['kill', 'failed_defence'])].shape[0]); n_b = int((ta.team == 'BRA').sum())
# a kill must end the rally won by the attacker
ta2 = ta.merge(ral[['rally', 'winner']], on='rally')
kills = ta2[ta2.action_subtype == 'kill']; k_cons = (kills.winner == kills.team).mean() * 100
errs = ta2[ta2.action_subtype == 'error']; e_cons = (errs.winner != errs.team).mean() * 100
dest_m = t.ball_method.isin(['flight_fit', 'touch_depth', 'shadow_tri']).mean() * 100
layer('attacks', 'התקפות (תוצאה, זווית, יעד)', min(k_cons, e_cons) * dest_m / 100, [
    chk('"נקודה" = הקבוצה התוקפת זכתה בנקודה', f'{k_cons:.0f}%', k_cons > 98),
    chk('"טעות" = הקבוצה התוקפת הפסידה', f'{e_cons:.0f}%', e_cons > 98),
    chk('מיקום הכדור במגע מתוך מסלול (ולא מגודל בלבד)', f'{dest_m:.0f}%', dest_m > 90),
    chk('זווית: יעד = המגע הבא או הנחיתה', 'יעד משוער אם הכדור נגע בשחקן לפני הנחיתה', True)],
    'הזווית נמדדת עד המגע הבא, לא עד איפה שהכדור היה נוחת. במגרש מוצגות רק 110 התקפות שהיעד שלהן ידוע, לכן האחוזים שם (22% / 43%) מעט שונים מכאן.',
    [dict(what='התקפה שהביאה נקודה, ישראל', value=f'{round(100 * k_i / n_i)}% ({k_i}/{n_i})', ci=f'{wilson(k_i, n_i)[0]}–{wilson(k_i, n_i)[1]}%'),
     dict(what='התקפה שהביאה נקודה, ברזיל', value=f'{round(100 * k_b / n_b)}% ({k_b}/{n_b})', ci=f'{wilson(k_b, n_b)[0]}–{wilson(k_b, n_b)[1]}%')])

# ---------------------------------------------------------------- landings
lr = pd.read_csv('data/analysis/point_landings.csv')
src = lr.source.value_counts().to_dict()
has_xy = lr.x.notna().mean() * 100
inc = lr[lr.kind == 'in_court']
geo = (((inc.x >= -0.3) & (inc.x <= 16.3) & (inc.y >= -0.3) & (inc.y <= 8.3)).mean() * 100) if len(inc) else 100
outs = lr[(lr.kind == 'out') & lr.x.notna()]
geo_out = (((outs.x < 0.2) | (outs.x > 15.8) | (outs.y < 0.2) | (outs.y > 7.8)).mean() * 100) if len(outs) else 100
w_ok = (lr.winner == ral.set_index('rally').loc[lr.rally, 'winner'].values).mean() * 100
hand = src.get('hand-marked landing', 0)
dp = lr[lr.source == 'hand-marked (doubt page)']
dp_xy, dp_kind = int(dp.x.notna().sum()), int(dp.x.isna().sum())
acc_l = (hand * 97 + dp_xy * 95 + dp_kind * 80 + src.get('ball track', 0) * 85 + src.get('rally end', 0) * 50) / len(lr)
bad_out = outs[~((outs.x < 0.2) | (outs.x > 15.8) | (outs.y < 0.2) | (outs.y > 7.8))].rally.tolist()
layer('landings', 'איפה נגמרו הנקודות', acc_l, [
    chk('מקור: סימון ידני (סבב ראשון)', f'{hand}/{len(lr)}', True),
    chk('מקור: סימון ידני בדף הספקות – עם מיקום', f'{dp_xy}/{len(lr)}', True),
    chk('מקור: סימון ידני – רק בפנים/בחוץ, בלי מיקום', f'{dp_kind}/{len(lr)}', True),
    chk('מקור: מסלול כדור', f'{src.get("ball track", 0)}/{len(lr)}', True),
    chk('מקור: סוף נקודה בלבד (בלי מיקום אמין)', f'{src.get("rally end", 0)}/{len(lr)}', src.get('rally end', 0) < 5),
    chk('"בפנים" באמת בתוך הקווים', f'{geo:.0f}%', geo > 95),
    chk('"בחוץ" באמת מחוץ לקווים', f'{geo_out:.0f}%' + (f' – לבדיקה חוזרת: נקודות {", ".join(map(str, bad_out))}' if bad_out else ''), geo_out > 95),
    chk('המנצח מתאים ללוח', f'{w_ok:.0f}%', w_ok == 100)],
    'אחרי סבב הסימון הראשון: 9 נקודות עדיין רק "סוף נקודה", ו־5 עם בפנים/בחוץ בלי מיקום.')

# ---------------------------------------------------------------- ball height & net crossings
bh = pd.read_csv('data/analysis/ball_height_v2.csv')
bh = bh[(bh.frame < len(inplay))]
bp = bh[inplay[bh.frame.to_numpy()]]
meth = bp.method.value_counts(normalize=True).mul(100).round(1).to_dict()
sig = bp.height_sigma_m
s10 = (sig <= 0.1).mean() * 100
neg = (bp.height_m < -0.05).mean() * 100
st = json.load(open('data/analysis/ball_height_settings.json'))['checks']
fl = bp[bp.method == 'flight_fit'].sort_values('frame')
cross = []
for fid, g in fl.groupby('flight_id'):
    xs, hs = g.court_x_m.to_numpy(), g.height_m.to_numpy()
    for i in range(len(xs) - 1):
        if (xs[i] - 8) * (xs[i + 1] - 8) < 0:
            w = (8 - xs[i]) / (xs[i + 1] - xs[i]); cross.append(hs[i] + w * (hs[i + 1] - hs[i]))
cross = np.array(cross)
below = (cross < 2.31 - 0.08).mean() * 100
from math import erf as _erf
within20 = np.nanmean([_erf(0.2 / (s_ * np.sqrt(2))) for s_ in sig.fillna(0.4)]) * 100
layer('height', 'גובה הכדור ומעברי רשת', within20, [
    chk('סיכוי שגובה פריים נכון עד 20 ס״מ (לפי אי־הוודאות של כל פריים)', f'{within20:.0f}%', within20 > 85),
    chk('נחיתות שסומנו ביד (לא שימשו לכיול)', f'שגיאה ממוצעת {st["held_out_landings_mae_m"] * 100:.0f} ס״מ', True),
    chk('מגעי חזה שסומנו ביד', f'הטיה {st["held_out_chest_marked_height_median_m"] * 100:.0f} ס״מ, שגיאה {st["held_out_chest_marked_height_mae_m"] * 100:.0f} ס״מ', True),
    chk('כדור שחוצה את הרשת מתחת לגובהה (בלתי אפשרי)', f'{below:.1f}% מהמעברים', below < 5),
    chk('גובה שלילי', f'{neg:.2f}%', neg < 0.5),
    chk('פריימים עם אי־ודאות עד 10 ס״מ', f'{s10:.0f}%', s10 > 60),
    chk('פריימים בלי כדור', f'{meth.get("missing", 0)}%', meth.get('missing', 0) < 15)],
    'גובה במעוף מדויק (±8 ס״מ עם צל). מחוץ למעופים (כדור מוחזק/מתגלגל) רק לפי גודל: ±40 ס״מ.',
    [dict(what='גובה מעבר רשת חציוני', value=f'{np.median(cross):.2f} מ׳', ci=f'{np.percentile(cross, 25):.2f}–{np.percentile(cross, 75):.2f} מ׳ (רבעונים)')])

# ---------------------------------------------------------------- partner gap
sp = pd.read_csv('data/analysis/team_spacing_v3.csv').query("moment == 'attack'")
c_ok = (sp.conceded == sp.outcome.isin(['kill', 'failed_defence'])).mean() * 100
fI = sp[(sp.defending_team == 'ISR') & (sp.distance > 3.5)]; nI = sp[(sp.defending_team == 'ISR') & (sp.distance <= 3.5)]
k1, n1, k2, n2 = int(fI.conceded.sum()), len(fI), int(nI.conceded.sum()), len(nI)
# two-proportion z-test
pp_ = (k1 + k2) / (n1 + n2); z = (k1 / n1 - k2 / n2) / np.sqrt(pp_ * (1 - pp_) * (1 / n1 + 1 / n2))
from math import erf, sqrt
pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
src_g = sp.gap_source.value_counts().to_dict() if 'gap_source' in sp else {}
n_pose = src_g.get('pose joints', 0)
pv = pd.read_csv('data/experiments/doubt_marks/gaps_pose_vs_user.csv') if os.path.exists('data/experiments/doubt_marks/gaps_pose_vs_user.csv') else None
acc_gap = 97 if n_pose == len(sp) else 95
pu = pv.dropna(subset=['d_pose', 'd_user']) if pv is not None else None
layer('gap', 'מרווח בהגנה', min(c_ok, acc_gap), [
    chk('מקור: מפרקי הרגליים (עקב + בוהן, ביטחון 0.5 ומעלה)', f'{n_pose} מתוך {len(sp)} רגעים', n_pose == len(sp)),
    chk('מפרקים מול הסימון שלך (4 רגליים)', f'{len(pu)} רגעים: הפרש חציוני {(pu.d_pose - pu.d_user).abs().median():.2f} מ׳' if pu is not None else '-', True),
    chk('מפרקים מול המעקב', f'הפרש חציוני {(pv.d_pose - pv.d_track).abs().median():.2f} מ׳' if pv is not None else '-', True),
    chk('"ספגו" תואם לתוצאת ההתקפה (נקודה / הוחזרה)', f'{c_ok:.0f}%', c_ok > 98),
    chk('ההבדל בספיגה רחוק/קרוב מובהק סטטיסטית', f'p = {pval:.3f}', pval < 0.05)],
    'נמדד ברגע המגע של התוקף. ההבדל בין מרווח גדול לקטן הוא קשר, לא בהכרח סיבה.',
    [dict(what='ספיגה של ישראל, מרווח מעל 3.5 מ׳', value=f'{round(100 * k1 / n1)}% ({k1}/{n1})', ci=f'{wilson(k1, n1)[0]}–{wilson(k1, n1)[1]}%'),
     dict(what='ספיגה של ישראל, מרווח עד 3.5 מ׳', value=f'{round(100 * k2 / n2)}% ({k2}/{n2})', ci=f'{wilson(k2, n2)[0]}–{wilson(k2, n2)[1]}%')])

# ---------------------------------------------------------------- shoulders (pose)
bo = pd.read_csv('data/analysis/body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'k', 'angle_to_net_deg'])
sh = bo[(bo.part == 'shoulders') & (bo.k == 0)].sort_values(['player', 'frame'])
flips = []
for pl, g in sh.groupby('player'):
    f = g.frame.to_numpy(); a = g.angle_to_net_deg.to_numpy()
    m = np.diff(f) == 1
    da = np.abs((np.diff(a) + 180) % 360 - 180)[m]
    flips.append(da)
da = np.concatenate(flips)
flip = (da > 120).mean() * 100
cover = sh[sh.frame < len(inplay)]
cover = cover[inplay[cover.frame.to_numpy()]].groupby('player').size().sum() / (inplay.sum() * 4) * 100
layer('shoulders', 'כיוון כתפיים', 100 - flip * 3 - 10, [
    chk('קפיצות של יותר מ־120° בין פריימים (בלבול ימין/שמאל של המודל)', f'{flip:.2f}%', flip < 1),
    chk('כיסוי בזמן משחק', f'{cover:.0f}% מהפריימים', cover > 70),
    chk('גובה הכתף מניח שחקן זקוף', 'בכריעה או קפיצה הכיוון פחות מדויק', False)],
    'זווית לכל פריים ±15° בערך; ההתפלגות (השושנה) יציבה הרבה יותר מפריים בודד.')

# ---------------------------------------------------------------- reaction
ex = pd.read_csv('data/analysis/pace_reaction_excess_0_749.csv')
pk = ex[(ex.lag_from_s >= 0.28) & (ex.lag_from_s <= 0.32)]
from math import lgamma, exp, log
def pois_sf(k, mu):
    return 1 - sum(exp(i * log(mu) - mu - lgamma(i + 1)) for i in range(int(k)))
pv = {r.relation: pois_sf(r.real, r.chance) for r in pk.itertuples()}
layer('reaction', 'זמן תגובה (30 שניות ראשונות)', 60, [
    chk('שיא ב־0.28–0.32 ש׳ מעל המקרי – חבר', f'p ≈ {pv.get("teammate", 1):.3f}', pv.get('teammate', 1) < 0.05),
    chk('שיא ב־0.28–0.32 ש׳ מעל המקרי – יריב', f'p ≈ {pv.get("opponent", 1):.3f}', pv.get('opponent', 1) < 0.05),
    chk('רק 30 שניות', '13 תגובות מול נגיעה; 125 שינויי קצב', False),
    chk('רזולוציה', '40 מילישניות (25 פריימים לשנייה)', True)],
    'הממצא סביר פיזיולוגית (≈0.3 ש׳) אבל מבוסס על חלון קצר.')

# ---------------------------------------------------------------- formation
fm = pd.read_csv('data/analysis/formation_points.csv')
fmg = fm.merge(ral[['rally', 'winner']], on='rally')
f_ok = (fmg.result.to_numpy() == np.where(fmg.winner == fmg.team, 'won', 'lost')).mean() * 100
layer('formation', 'מערך הגנה', min(f_ok, acc_pos) - 10, [
    chk('ניצחון/הפסד תואם ללוח', f'{f_ok:.0f}% – שגוי בנקודות 31 ו־49 (49 לפי הלוח הלא מתוקן)', f_ok > 98),
    chk('סף הסיווג (רשת/מאחור/צד) נקבע ידנית', 'שינוי סף של 0.5 מ׳ מעביר נקודות בין מערכים', False)],
    'הספירות קטנות (8–25 לכל מערך).')

out = dict(layers=audit, generated='scripts/analysis/audit.py')
s = json.dumps(out, ensure_ascii=False, indent=1)
open('data/analysis/audit.json', 'w').write(s)
open('site/court_story/audit.json', 'w').write(s)
print('\nwritten data/analysis/audit.json')
