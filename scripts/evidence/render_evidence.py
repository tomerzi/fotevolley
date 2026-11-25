"""Evidence media for the coach summary: every number in it can be checked on the video.

Writes into OUT (default coach_analysis/evidence/):
  clips/r{rally}.mp4     every rally (v3 video, 1 s before to ~1 s after), 960x540, with live overlays:
                         player names, speed (m/s) and court square under the feet, the partner distance line
                         (metres), the body facing arrow (shoulders), the ball and its height above the sand,
                         a mini court with the four players, a caption at every touch and at the end of the point.
  touches/t{touch}.jpg   the contact frame of every in-play touch with the same overlay + a side panel: zoom on the
                         player with his skeleton (feet line at chest touches) and every value measured on that touch.
  points/p{point}.jpg    the moment each point was decided (the frame the partner distance is measured at) with the
                         landing spot and a side panel.
  index.json             clip start frame per rally, touch -> (rally, seconds into clip), point -> (rally, seconds).
usage: python scripts/evidence/render_evidence.py [OUT] [--rallies 1,2,3] [--stills-only]
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = next((a for a in sys.argv[1:] if not a.startswith('--') and ',' not in a), 'coach_analysis/evidence/')
ONLY = None
STILLS = '--stills-only' in sys.argv   # re-make the jpgs, keep the clips
if '--rallies' in sys.argv:
    ONLY = {int(x) for x in sys.argv[sys.argv.index('--rallies') + 1].split(',')}
VIDEO = 'video/main_camera_clean_v3.mp4'
W, H, S = 960, 540, 0.5          # output size, scale from 1920x1080
PRE, POST = 12, 30               # frames before the rally start / after its end
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
F = {k: ImageFont.truetype(FONT, k) for k in (11, 13, 15, 17)}
FB = {k: ImageFont.truetype(FONTB, k) for k in (13, 15, 17, 20)}
for d in ('clips', 'touches', 'points'):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

HE = {'Franklin': 'פרנקלין', 'Hiltinho': 'הילטיניו', 'Maor_Haas': 'מאור האס', 'Ron_Ben_Ishay': 'רון בן ישי',
      'ISR': 'ישראל', 'BRA': 'ברזיל'}
TEAM = {'Franklin': 'BRA', 'Hiltinho': 'BRA', 'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR'}
PARTNER = {'Franklin': 'Hiltinho', 'Hiltinho': 'Franklin', 'Maor_Haas': 'Ron_Ben_Ishay', 'Ron_Ben_Ishay': 'Maor_Haas'}
COL = {'ISR': (60, 140, 235), 'BRA': (245, 160, 40)}
PART = {'chest': 'חזה', 'foot': 'רגל', 'head': 'ראש', 'knee': 'ברך', 'shin': 'שוק', 'shoulder': 'כתף', 'unknown': 'לא ידוע'}
ACT = {'reception': 'קבלה', 'set': 'הגבהה', 'attack': 'התקפה', 'serve': 'הגשה'}
QUAL = {'to_attack': 'הגיע להתקפה', 'returned': 'הכדור חזר מהיריב', 'kill': 'ווינר', 'lost': 'איבד את הכדור',
        'kept': 'נשאר בידיים', 'unknown': 'לא ידוע'}
STANCE = {'set': 'עמד יציב', 'moving': 'הגיע בתנועה', 'jump': 'קפץ'}
END = {'landed_in': 'נחת בתוך המגרש', 'out': 'יצא החוצה', 'landed_out': 'נחת בחוץ', 'net': 'רשת', 'fault': 'עבירה',
       'excluded': 'לא צולם'}
LANE = {'left': 'שמאל', 'mid': 'מרכז', 'right': 'ימין'}
DEPTH = {'net': 'ליד הרשת', 'mid': 'באמצע', 'base': 'בבסיס'}

Hpx2m = np.load('data/mappings/homography_px_to_m.npy')
Hm2px = np.linalg.inv(Hpx2m)


def m2px(x, y):
    p = Hm2px @ np.array([x, y, 1.0])
    return p[0] / p[2] * S, p[1] / p[2] * S


def f2(x, d=2):
    return '–' if x is None or pd.isna(x) else f'{x:.{d}f}'


# ---------------- data
fm = pd.read_csv('data/mappings/frame_mapping_v3.csv')
v2_to_v3 = fm[~fm.deleted].set_index('v2_frame').v3_frame
rallies = pd.read_csv('data/points/rallies_v3video.csv')
rallies['isr_before'] = rallies.ISR_score_after.shift(fill_value=0)
rallies['bra_before'] = rallies.BRA_score_after.shift(fill_value=0)
rallies.loc[rallies.set != rallies.set.shift(), ['isr_before', 'bra_before']] = 0

pos = pd.read_csv('data/analysis/player_positions_v3.csv', usecols=['frame', 'v2_frame', 'player', 'u', 'v', 'x', 'y', 'speed', 'square'])
POS = {(r.frame, r.player): r for r in pos.itertuples(index=False)}
box = pd.read_csv('data/boxes/player_boxes_v3video.csv', usecols=['frame', 'player', 'x1', 'y1', 'x2', 'y2'])
BOX = {}
for r in box.itertuples(index=False):
    BOX.setdefault(r.frame, []).append(r)
ball = pd.read_csv('data/boxes/ball_boxes_v3video.csv', usecols=['frame', 'v2_frame', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y'])
BALL = {r.frame: r for r in ball.dropna(subset=['center_x']).itertuples(index=False)}
bh = pd.read_csv('data/analysis/ball_height_v2.csv', usecols=['frame', 'height_m', 'height_sigma_m']).set_index('frame')
BH = {f: (h, s) for f, h, s in zip(bh.index, bh.height_m, bh.height_sigma_m)}
bd = pd.read_csv('data/analysis/body_direction_full_match.csv', usecols=['frame', 'player', 'part', 'fx', 'fy', 'angle_to_net_deg'])
bd = bd[bd.part == 'shoulders'].groupby(['frame', 'player'])[['fx', 'fy', 'angle_to_net_deg']].mean()
FACE = {k: (r.fx, r.fy, r.angle_to_net_deg) for k, r in zip(bd.index, bd.itertuples(index=False))}

t = pd.read_csv('data/analysis/touch_outcomes.csv')
t['touch'] = t.touch.astype(int)
t['v2_frame'] = t.v2_frame.fillna(t.frame.map(fm[~fm.deleted].set_index('v3_frame').v2_frame))
mas = pd.read_csv('data/contacts/touches_master.csv', usecols=['touch', 'ball_height_m'])
ba = pd.read_csv('data/analysis/ball_after_touch.csv', usecols=['touch', 'apex_m', 'air_time_s'])
st = pd.read_csv('coach_analysis/recomputed_2026-11-24/touches.csv', usecols=['touch', 'tip', 'stance'])
rb = pd.read_csv('data/analysis/return_to_base.csv', usecols=['touch', 'return_s', 'censored', 'dist_from_base_m'])
sp = pd.read_csv('data/analysis/set_partner.csv', usecols=['touch', 'distance_m', 'partner_speed_ms', 'partner_moving',
                                                        'partner_angle_to_net_deg', 'partner_towards_net_m'])
fs = pd.read_csv('data/analysis/set_foot_side.csv', usecols=['touch', 'side', 'right_of_body_m', 'in_front_m'])
cp = pd.read_csv('data/analysis/chest_profile.csv', usecols=['touch', 'stance_m', 'ball_speed_in', 'net_dist_m'])
tb = pd.read_csv('data/analysis/touch_body.csv', usecols=['touch', 'knee_l', 'knee_l_ok', 'knee_r', 'knee_r_ok',
                                                       'trunk_lean', 'trunk_lean_ok', 'trunk_lean_dir', 'facing'])
for d in (mas, ba, st, rb, sp, fs, cp, tb):
    d['touch'] = d.touch.astype(int)
    t = t.merge(d.drop_duplicates('touch'), on='touch', how='left')
T_BY_FRAME = {}
for r in t.itertuples(index=False):
    T_BY_FRAME.setdefault(int(r.frame), []).append(r)

pe = pd.read_csv('data/points/point_end_verified.csv').set_index('point')
cd = pd.read_csv('data/analysis/conceded_partner_distance.csv').set_index('rally')
pts = pd.read_csv('coach_analysis/recomputed_2026-11-24/points.csv').set_index('point')
POINT_FRAME = {}
for rr in rallies.itertuples():
    f = cd.point_frame.get(rr.rally)
    f3 = v2_to_v3.get(int(f)) if f is not None and not pd.isna(f) else None
    POINT_FRAME[int(f3) if f3 is not None and not pd.isna(f3) else int(rr.end_frame)] = rr.rally

# skeleton points for the touch stills
KP = ['nose', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
      'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
      'left_big_toe', 'right_big_toe']
BONES = [('neck', 'nose'), ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
         ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
         ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
         ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('left_ankle', 'left_heel'), ('left_ankle', 'left_big_toe'),
         ('right_ankle', 'right_heel'), ('right_ankle', 'right_big_toe')]
need = set(t.frame.astype(int))
OLD = {'BRA_A': 'Franklin', 'BRA_B': 'Hiltinho', 'ISR_A': 'Ron_Ben_Ishay', 'ISR_B': 'Maor_Haas'}  # the Sapiens2 file keeps the old codes
cols = ['frame', 'player'] + [f'{k}_{c}' for k in KP for c in 'xys']
POSE = {}
for c in pd.read_csv('data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv', usecols=cols, chunksize=200000):
    c = c[c.frame.isin(need)]
    for r in c.to_dict('records'):
        POSE[(r['frame'], OLD.get(r['player'], r['player']))] = r
print('pose rows for touches:', len(POSE))


# ---------------- drawing
def text(d, xy, s, font, fill=(255, 255, 255), anchor='ra', bg=None, pad=3):
    if bg is not None:
        b = d.textbbox(xy, s, font=font, anchor=anchor)
        d.rectangle((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), fill=bg)
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def dashed(d, a, b, fill, width=2, dash=7):
    a, b = np.array(a, float), np.array(b, float)
    L = np.linalg.norm(b - a)
    if L < 1:
        return
    u = (b - a) / L
    for s0 in np.arange(0, L, 2 * dash):
        p, q = a + u * s0, a + u * min(s0 + dash, L)
        d.line((*p, *q), fill=fill, width=width)


def arrow(d, a, b, fill, width=2):
    d.line((*a, *b), fill=fill, width=width)
    v = np.array(b) - np.array(a); L = np.linalg.norm(v)
    if L < 2:
        return
    u = v / L; n = np.array([-u[1], u[0]])
    p1 = np.array(b) - u * 9 + n * 5; p2 = np.array(b) - u * 9 - n * 5
    d.polygon([tuple(b), tuple(p1), tuple(p2)], fill=fill)


def overlay(img, f, rr, touch_banner=None, end_banner=None, highlight=None, landing=None):
    d = ImageDraw.Draw(img, 'RGBA')
    v2 = None
    # partner lines + players
    ground = {}
    for p in TEAM:
        r = POS.get((f, p))
        if r is not None and not pd.isna(r.u):
            ground[p] = (r.u * S, r.v * S, r)
            v2 = r.v2_frame
    for a, b in (('Ron_Ben_Ishay', 'Maor_Haas'), ('Franklin', 'Hiltinho')):
        if a in ground and b in ground:
            ra, rb_ = ground[a][2], ground[b][2]
            dist = np.hypot(ra.x - rb_.x, ra.y - rb_.y)
            c = COL[TEAM[a]]
            dashed(d, ground[a][:2], ground[b][:2], c + (230,), 2)
            mx, my = (ground[a][0] + ground[b][0]) / 2, (ground[a][1] + ground[b][1]) / 2
            text(d, (mx, my), f'{dist:.1f} מ׳', FB[13], anchor='mm', bg=(0, 0, 0, 160))
    for b in BOX.get(f, []):
        c = COL[TEAM.get(b.player, 'ISR')]
        w = 4 if b.player == highlight else 2
        d.rectangle((b.x1 * S, b.y1 * S, b.x2 * S, b.y2 * S), outline=c + (255,), width=w)
        text(d, ((b.x1 + b.x2) / 2 * S, b.y1 * S - 4), HE.get(b.player, b.player), FB[13], anchor='md', bg=c + (220,), pad=2)
    for p, (u, v, r) in ground.items():
        fc = FACE.get((r.v2_frame, p))
        if fc is not None and not pd.isna(fc[0]):
            a = m2px(r.x, r.y); bpt = m2px(r.x + 1.1 * fc[0], r.y + 1.1 * fc[1])
            arrow(d, (u, v), (u + bpt[0] - a[0], v + bpt[1] - a[1]), (255, 255, 255, 230), 2)
        sq = r.square if isinstance(r.square, str) else ''
        text(d, (u, v + 14), f'{r.speed:.1f} מ׳/ש׳ · {sq}', F[11], anchor='mm', bg=(0, 0, 0, 150), pad=2)
    # ball
    bl = BALL.get(f)
    if bl is not None:
        cx, cy = bl.center_x * S, bl.center_y * S
        rad = max(6, (bl.x2 - bl.x1) * S / 2 + 3)
        d.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=(255, 235, 60, 255), width=2)
        h = BH.get(int(bl.v2_frame)) if not pd.isna(bl.v2_frame) else None
        if h is not None and h[1] <= 0.4:
            text(d, (cx + rad + 4, cy), f'{h[0]:.1f} מ׳', FB[13], fill=(255, 235, 60), anchor='lm', bg=(0, 0, 0, 160))
    # header
    sc = f'ראלי {rr.rally} · מערכה {rr.set} · ישראל {int(rr.isr_before)} – ברזיל {int(rr.bra_before)} · {(f - rr.start_frame) / 25:+.1f} ש׳'
    text(d, (W - 10, 10), sc, FB[15], anchor='ra', bg=(0, 0, 0, 170), pad=5)
    # mini court
    ox, oy, k = 290, 14, 11   # right of the on-screen scoreboard
    d.rectangle((ox - 4, oy - 4, ox + 16 * k + 4, oy + 8 * k + 4), fill=(0, 0, 0, 150))
    d.rectangle((ox, oy, ox + 16 * k, oy + 8 * k), outline=(230, 220, 190, 255), width=1)
    for q in (8 / 3, 16 / 3, 8 + 8 / 3, 8 + 16 / 3):
        d.line((ox + q * k, oy, ox + q * k, oy + 8 * k), fill=(230, 220, 190, 90))
    for q in (8 / 3, 16 / 3):
        d.line((ox, oy + q * k, ox + 16 * k, oy + q * k), fill=(230, 220, 190, 90))
    d.line((ox + 8 * k, oy - 3, ox + 8 * k, oy + 8 * k + 3), fill=(255, 255, 255, 255), width=2)
    for p, (u, v, r) in ground.items():
        c = COL[TEAM[p]]
        d.ellipse((ox + r.x * k - 4, oy + r.y * k - 4, ox + r.x * k + 4, oy + r.y * k + 4), fill=c + (255,), outline=(255, 255, 255, 255))
    if landing is not None:
        lx, ly = landing
        d.line((ox + lx * k - 5, oy + ly * k - 5, ox + lx * k + 5, oy + ly * k + 5), fill=(255, 60, 60, 255), width=2)
        d.line((ox + lx * k - 5, oy + ly * k + 5, ox + lx * k + 5, oy + ly * k - 5), fill=(255, 60, 60, 255), width=2)
        px = m2px(lx, ly)
        d.line((px[0] - 9, px[1] - 9, px[0] + 9, px[1] + 9), fill=(255, 60, 60, 255), width=3)
        d.line((px[0] - 9, px[1] + 9, px[0] + 9, px[1] - 9), fill=(255, 60, 60, 255), width=3)
    for i, s in enumerate(x for x in (touch_banner, end_banner) if x):
        text(d, (W / 2, H - 14 - 30 * i), s, FB[17], anchor='md', bg=(0, 0, 0, 200), pad=6)
    return img


def touch_line(r):
    tip = f' (נגיעה {int(r.tip)} מתוך 3)' if not pd.isna(r.tip) else ''
    s = f'נגיעה {r.touch} · {HE[r.player]} · {PART.get(r.body_part, r.body_part)} · {ACT.get(r.action, r.action)}{tip}'
    if not pd.isna(r.ball_height_m):
        s += f' · כדור {r.ball_height_m:.1f} מ׳'
    if not pd.isna(r.apex_m):
        s += f' → שיא {r.apex_m:.1f} מ׳'
    return s


def end_line(rally):
    if rally not in pe.index:
        return None
    e = pe.loc[rally]
    s = f'נקודה ל{HE.get(e.winner, e.winner)} · {END.get(e.end_type, e.end_type)}'
    if isinstance(e.square, str) and e.square != 'out':
        s += f' · {e.square}'
    if rally in pts.index and isinstance(pts.loc[rally].last_player, str):
        s += f' · נגיעה אחרונה: {HE[pts.loc[rally].last_player]}'
    return s


def panel(lines, crop=None, h=450, w=300):
    pnl = Image.new('RGB', (w, h), (20, 28, 36))
    d = ImageDraw.Draw(pnl)
    y = 8
    if crop is not None:
        pnl.paste(crop, ((w - crop.width) // 2, 8)); y = crop.height + 16
    for i, s in enumerate(lines):
        if y > h - 18:
            break
        d.text((w - 10, y), s, font=FB[15] if i == 0 else F[13], fill=(255, 255, 255) if i == 0 else (215, 225, 232), anchor='ra')
        y += 21 if i == 0 else 18
    return pnl


def player_crop(frame_bgr, f, r):
    b = next((b for b in BOX.get(f, []) if b.player == r.player), None)
    if b is None:
        return None
    bw, bh_ = b.x2 - b.x1, b.y2 - b.y1
    side = max(bw, bh_) * 1.5
    cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
    bl = BALL.get(f)
    if bl is not None and abs(bl.center_x - cx) < side and abs(bl.center_y - cy) < side:
        cx, cy = (cx * 2 + bl.center_x) / 3, (cy * 2 + bl.center_y) / 3
        side = max(side, max(abs(bl.center_x - cx), abs(bl.center_y - cy)) * 2 + 40)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(1920, cx + side / 2)), int(min(1080, cy + side / 2))
    im = Image.fromarray(cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    sc = 210 / max(im.width, im.height)
    im = im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)
    d = ImageDraw.Draw(im, 'RGBA')
    P = POSE.get((f, r.player))
    if P is not None:
        def pt(k):
            if P.get(f'{k}_s', 0) >= 0.3 and not pd.isna(P.get(f'{k}_x')):
                return ((P[f'{k}_x'] - x0) * sc, (P[f'{k}_y'] - y0) * sc)
        for a, b2 in BONES:
            pa, pb = pt(a), pt(b2)
            if pa and pb:
                d.line((*pa, *pb), fill=(80, 255, 160, 230), width=3)
        for k in KP:
            p = pt(k)
            if p:
                d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(255, 255, 255, 255))
        if r.body_part == 'chest':
            la, ra = pt('left_ankle'), pt('right_ankle')
            if la and ra:
                d.line((*la, *ra), fill=(255, 80, 200, 255), width=3)
                if not pd.isna(r.stance_m):
                    text(d, ((la[0] + ra[0]) / 2, max(la[1], ra[1]) + 6), f'{r.stance_m:.2f} מ׳', FB[13], anchor='ma', bg=(0, 0, 0, 180))
    if bl is not None:
        bx, by = (bl.center_x - x0) * sc, (bl.center_y - y0) * sc
        rad = max(5, (bl.x2 - bl.x1) * sc / 2 + 2)
        d.ellipse((bx - rad, by - rad, bx + rad, by + rad), outline=(255, 235, 60, 255), width=2)
    return im


def touch_lines(r):
    L = [f'נגיעה {r.touch} · {HE[r.player]}',
         f'{PART.get(r.body_part, r.body_part)} · {ACT.get(r.action, r.action)}' + (f' · נגיעה {int(r.tip)} מתוך 3' if not pd.isna(r.tip) else ''),
         f'גובה הכדור בנגיעה: {f2(r.ball_height_m)} מ׳',
         f'שיא אחרי הנגיעה: {f2(r.apex_m)} מ׳',
         f'זמן עד הנגיעה הבאה: {f2(r.air_time_s)} ש׳',
         f'תוצאה: {QUAL.get(r.quality, r.quality)}']
    if isinstance(r.stance, str):
        L.append(f'עמידה (תיוג ידני): {STANCE.get(r.stance, r.stance)}')
    if r.body_part == 'chest':
        L.append(f'מרחק בין הרגליים: {f2(r.stance_m)} מ׳')
        kl = f2(r.knee_l, 0) if r.knee_l_ok is True else '–'
        kr = f2(r.knee_r, 0) if r.knee_r_ok is True else '–'
        L.append(f'ברך שמאל {kl}° · ימין {kr}°')
        if r.trunk_lean_ok is True:
            L.append(f'הטיית גו {f2(r.trunk_lean, 0)}° ' + ('אחורה' if r.trunk_lean_dir == 'back' else 'לרשת'))
        L.append(f'מרחק מהרשת {f2(r.net_dist_m)} מ׳ · כדור נכנס {f2(r.ball_speed_in, 1)} מ׳/ש׳')
    if r.action == 'set' and not pd.isna(r.distance_m):
        L.append(f'השותף: {f2(r.distance_m)} מ׳, ' + ('בתנועה' if r.partner_moving is True else 'עומד') +
                 f' ({f2(r.partner_speed_ms, 1)} מ׳/ש׳)')
        L.append(f'השותף פונה {f2(r.partner_angle_to_net_deg, 0)}° לרשת')
    if isinstance(r.side, str):
        L.append(f'הרגל {"במרכז" if r.side == "middle" else LANE[r.side] + " לגוף"} ({f2(r.right_of_body_m)} מ׳)')
    if not pd.isna(r.return_s):
        L.append(f'חזרה לבסיס: {f2(r.return_s, 1)} ש׳' + (' (הראלי נגמר קודם)' if r.censored is True else ''))
    return L


def point_lines(rally):
    L = [f'נקודה {rally}']
    if rally in pe.index:
        e = pe.loc[rally]
        L.append(f'זכתה: {HE.get(e.winner, e.winner)} · {END.get(e.end_type, e.end_type)}')
        if isinstance(e.square, str):
            sq = e.square + (f' ({LANE.get(e.lane, "")}, {DEPTH.get(e.depth, "")})' if isinstance(e.lane, str) else '')
            L.append(f'נחיתה: {sq}')
    if rally in pts.index:
        q = pts.loc[rally]
        if isinstance(q.last_player, str):
            L.append(f'נגיעה אחרונה: {HE[q.last_player]}, {PART.get(q.last_part, q.last_part)}')
            L.append(f'{ACT.get(q.last_action, q.last_action)}' + (f', נגיעה {int(q.last_tip)} מתוך 3' if not pd.isna(q.last_tip) else ''))
    if rally in cd.index:
        c = cd.loc[rally]
        L.append(f'מרחק בין שחקני {HE[c.loser]}: {f2(c.distance_m)} מ׳')
        L.append('(הרגע: ' + {'own error (last touch)': 'הטעות', 'hand-marked landing': 'נחיתה שסומנה ידנית',
                              'hand-marked moment': 'רגע שסומן ידנית'}.get(c.source, 'סוף הראלי') + ')')
    return L


# ---------------- render
cap = cv2.VideoCapture(VIDEO)
index = {'fps': 25, 'clips': {}, 'touches': {}, 'points': {}}
todo = rallies if ONLY is None else rallies[rallies.rally.isin(ONLY)]
for rr in todo.itertuples():
    a, b = max(0, rr.start_frame - PRE), min(13494, rr.end_frame + POST)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out = os.path.join(OUT, 'clips', f'r{rr.rally}.mp4')
    ff = None if STILLS else subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '25',
                           '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30', '-pix_fmt', 'yuv420p',
                           '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    index['clips'][rr.rally] = dict(start=a, end=b)
    active = []   # (until_frame, banner, player)
    end_txt = end_line(rr.rally)
    pe_row = pe.loc[rr.rally] if rr.rally in pe.index else None
    landing = (pe_row.x, pe_row.y) if pe_row is not None and not pd.isna(pe_row.x) else None
    for f in range(a, b + 1):
        if STILLS and f not in T_BY_FRAME and POINT_FRAME.get(f) != rr.rally:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        for r in T_BY_FRAME.get(f, []):
            if r.rally == rr.rally:
                active.append((f + 30, touch_line(r), r.player))
        active = [x for x in active if x[0] >= f]
        img = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        tb_ = active[-1][1] if active else None
        hl = active[-1][2] if active else None
        eb = end_txt if f >= rr.end_frame - 3 else None
        overlay(img, f, rr, tb_, eb, hl, landing if f >= rr.end_frame - 3 else None)
        if ff:
            ff.stdin.write(img.tobytes())
        for r in T_BY_FRAME.get(f, []):
            if r.rally != rr.rally:
                continue
            still = Image.new('RGB', (1100, 450))
            fi = overlay(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)),
                         f, rr, touch_line(r), None, r.player)
            still.paste(fi.resize((800, 450), Image.LANCZOS), (300, 0))
            still.paste(panel(touch_lines(r), player_crop(fr, f, r)), (0, 0))
            still.save(os.path.join(OUT, 'touches', f't{r.touch}.jpg'), quality=72)
            index['touches'][r.touch] = dict(rally=rr.rally, t=round((f - a) / 25, 2))
        if POINT_FRAME.get(f) == rr.rally:
            fi = overlay(Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)),
                         f, rr, None, end_txt, None, landing)
            still = Image.new('RGB', (1100, 450))
            still.paste(fi.resize((800, 450), Image.LANCZOS), (300, 0))
            still.paste(panel(point_lines(rr.rally)), (0, 0))
            still.save(os.path.join(OUT, 'points', f'p{rr.rally}.jpg'), quality=72)
            index['points'][rr.rally] = dict(rally=rr.rally, t=round((f - a) / 25, 2))
    if ff:
        ff.stdin.close(); ff.wait()
    print('rally', rr.rally, 'frames', b - a + 1, flush=True)
json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'))
print('touches', len(index['touches']), 'points', len(index['points']))
