"""Pick one ball per frame from ball/candidates.csv by trajectory DP within each shot, then fill short
in-shot gaps with a curve fit. Never links or interpolates across scene cuts (data/mappings/scene_cuts.csv).
-> ball/ball_boxes.csv (frame,time_sec,x1,y1,x2,y2,center_x,center_y,confidence,source,model,shot,missing_reason)
source: detected (DP pick) | rescued (skipped candidate on the gap's extrapolated curve) | tracked (template match from a gap end)
        | interpolated (curve fit) | missing
usage: python scripts/ball/fill_ball.py [MAX_FRAME] [--track]"""
import os, sys
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H, FPS = 1920, 1080, 25
N_FRAMES = 15586

# known static false positives: (cx, cy, radius_px) — spectator holding a yellow ball above the banner
BLACKLIST = [(522, 250, 22)]

# off the sand (crowd, chairs, graphics) a candidate below conf C whose 8 px neighbourhood has detections in
# >= MIN of the +-WIN frames is background (bags on chairs, shirts/balls in the crowd). On the sand a still
# ball is a real ball at rest, so it is kept. Sand = ground point within these court-metre bounds (fixed camera).
SAND_X, SAND_Y = (-5.0, 23.0), (-2.3, 12.0)
STATIC_PX = 8
STATIC_TIERS = [(0.6, 50, 25)]     # (C, WIN, MIN)
BD_ONLY_PENALTY = 0.25
OVERLAY = (0, 0, 620, 215)          # scoreboard graphics (x1,y1,x2,y2): ballDedect-only hits there are dropped             # ballDedect alone fires on yellow shorts/bags; exp-2 is the reliable model
# chains (DP segments) that barely move are kept only if confident; short weak chains are dropped
CHAIN_STILL_PX, CHAIN_STILL_CONF = 20, 0.5
CHAIN_MIN_LEN, CHAIN_MIN_MAXCONF, CHAIN_SHORT_CONF = 3, 0.3, 0.45   # shorter chains need CHAIN_SHORT_CONF
MERGE_PX = 14          # candidates closer than this in one frame are one ball
K = 12                 # max frames a chain may skip
VMAX = 110             # px/frame, hardest kicks
TOL = 18               # px allowed deviation from constant-velocity prediction (per step)
GAP_COST = 0.12        # per skipped frame inside a chain
RESTART = 1.2          # cost of starting a new chain
UNARY_A, UNARY_B = 2.0, 0.12   # unary = A*score - B  (score = conf + agreement bonus)
AGREE = 0.15           # bonus when both models see it
MAX_FILL = 30          # longest gap to try to fill (frames)
FIT_N = 6              # anchor points each side for the curve fit
FIT_RES = 10           # px max residual of the fit on anchors
RESCUE_TOL = (12, 4, 50)   # candidate accepted in a gap if within a+b*k px (max c) of the curve extrapolated k frames
TRACK_MIN, TRACK_R, TRACK_EDGE, TRACK_MAX = 0.8, 50, 100, 8   # template tracking into gaps (score, search px, anchor margin, max frames)
VIDEO = os.path.join(ROOT, 'video', 'main_camera_clean_v2.mp4')


def load_candidates(max_frame):
    c = pd.read_csv(os.path.join(ROOT, 'data', 'ball_pipeline', 'candidates.csv'))
    c = c[c.frame < max_frame]
    c['cx'] = (c.x1 + c.x2) / 2
    c['cy'] = (c.y1 + c.y2) / 2
    for bx, by, r in BLACKLIST:
        c = c[np.hypot(c.cx - bx, c.cy - by) > r]
    out = []
    for f, g in c.groupby('frame', sort=True):
        g = g.sort_values('conf', ascending=False)
        used = np.zeros(len(g), bool)
        xs, ys, ms, cs = g.cx.values, g.cy.values, g.model.values, g.conf.values
        for i in range(len(g)):
            if used[i]:
                continue
            near = (~used) & (np.hypot(xs - xs[i], ys - ys[i]) < MERGE_PX)
            used |= near
            models = set(ms[near])
            r = g.iloc[i]
            if models == {'bd'} and OVERLAY[0] <= r.cx < OVERLAY[2] and OVERLAY[1] <= r.cy < OVERLAY[3]:
                continue
            bonus = AGREE if len(models) > 1 else (-BD_ONLY_PENALTY if models == {'bd'} else 0)
            out.append((f, r.x1, r.y1, r.x2, r.y2, r.cx, r.cy, r.conf, r.conf + bonus, '+'.join(sorted(models))))
    return pd.DataFrame(out, columns=['frame', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'conf', 'score', 'model'])


def drop_static(c, shot):
    """remove off-sand low-conf candidates at a location that has detections in most of the surrounding frames."""
    c = c.copy()
    H = np.load(os.path.join(ROOT, 'data', 'mappings', 'homography_px_to_m.npy'))
    g = cv2.perspectiveTransform(np.c_[c.cx, c.y2].astype(float)[None], H)[0]
    on_sand = (g[:, 0] > SAND_X[0]) & (g[:, 0] < SAND_X[1]) & (g[:, 1] > SAND_Y[0]) & (g[:, 1] < SAND_Y[1])
    c['shot'] = shot[c.frame.values]
    c['gx'] = (c.cx // STATIC_PX).astype(int); c['gy'] = (c.cy // STATIC_PX).astype(int)
    cells = {}
    for key, g in c.groupby(['shot', 'gx', 'gy']):
        cells[key] = np.unique(g.frame.values)
    drop = np.zeros(len(c), bool)
    sh, gx, gy, fr, conf = c.shot.values, c.gx.values, c.gy.values, c.frame.values, c.conf.values
    for i in np.where((conf < max(t[0] for t in STATIC_TIERS)) & ~on_sand)[0]:
        frames = [cells.get((sh[i], gx[i] + dx, gy[i] + dy), ()) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        frames = np.unique(np.concatenate([np.asarray(f) for f in frames]))
        for c_max, win, need in STATIC_TIERS:
            if conf[i] < c_max:
                lo, hi = np.searchsorted(frames, [fr[i] - win, fr[i] + win + 1])
                if hi - lo >= need:
                    drop[i] = True; break
    print(f'static filter: dropped {drop.sum()} of {len(c)} merged candidates')
    return c[~drop].drop(columns=['shot', 'gx', 'gy'])


def shots_of(n):
    cuts = pd.read_csv(os.path.join(ROOT, 'data', 'mappings', 'scene_cuts.csv'))
    cuts = sorted(int(f) for f in cuts[cuts.is_cut].frame)
    shot = np.zeros(n, int)
    for c in cuts:
        if c < n:
            shot[c:] += 1
    return shot


def dp_shot(d):
    """d: candidates of one shot sorted by frame. returns indices (into d) of the chosen chain(s)."""
    n = len(d)
    if n == 0:
        return []
    fr = d.frame.values; P = d[['cx', 'cy']].values
    wid = (d.x2 - d.x1).values
    u = UNARY_A * d.score.values - UNARY_B
    best = np.full(n, -np.inf); back = np.full(n, -1); vel = np.full((n, 2), np.nan)
    # prefix best of finished chains ending strictly before frame t
    order_frames = np.unique(fr)
    ends_best = {}          # frame -> best dp value of any candidate at frame
    run_best = 0.0          # best total of chains ending before current frame (0 = nothing chosen yet)
    run_arg = -1
    by_frame = {f: np.where(fr == f)[0] for f in order_frames}
    prefix = {}             # frame -> (run_best, run_arg) before that frame
    for f in order_frames:
        prefix[f] = (run_best, run_arg)
        for j in by_frame[f]:
            # new chain
            b, a = run_best - RESTART + u[j], -(run_arg + 2)   # encode restart pointer
            for k in range(1, K + 1):
                for i in by_frame.get(f - k, ()):
                    step = P[j] - P[i]
                    dist = np.hypot(*step)
                    if dist > VMAX * k:
                        continue
                    if not 0.5 < wid[j] / wid[i] < 2.0:
                        continue
                    if np.isnan(vel[i, 0]):
                        mc = 0.15 * dist / (VMAX * k)
                    else:
                        dev = np.hypot(*(P[i] + vel[i] * k - P[j]))
                        mc = 0.6 * max(0.0, dev - TOL * k) / (TOL * k)
                    v = best[i] - GAP_COST * (k - 1) - mc + u[j]
                    if v > b:
                        b, a = v, i
            best[j] = b; back[j] = a
            if a >= 0:
                vel[j] = (P[j] - P[a]) / (f - fr[a])
        fb = by_frame[f][np.argmax(best[by_frame[f]])]
        if best[fb] > run_best:
            run_best, run_arg = best[fb], fb
    # backtrack from overall best, splitting into chains at restarts
    chains, cur = [], []
    j = run_arg
    while j >= 0:
        cur.append(j)
        a = back[j]
        if a >= 0:
            j = a
        else:
            chains.append(cur[::-1]); cur = []
            j = -a - 2      # restart: jump to best chain ending before this one (-1 = none)
    chosen = []
    for ch in chains:
        conf = d.score.values[ch]
        ext = np.ptp(P[ch], axis=0).max()
        if conf.max() < (CHAIN_MIN_MAXCONF if len(ch) >= CHAIN_MIN_LEN else CHAIN_SHORT_CONF):
            continue
        if ext < CHAIN_STILL_PX and np.median(conf) < CHAIN_STILL_CONF:
            continue
        chosen += ch
    return sorted(chosen)


def gaps_in_shot(det, shot, max_len):
    n = len(det); i = 0
    while i < n:
        if det[i]:
            i += 1; continue
        j = i
        while j < n and not det[j]:
            j += 1
        if i > 0 and j < n and shot[i - 1] == shot[j] and j - i <= max_len:
            yield i, j - 1
        i = j


def extrapolate(fr, xs, ys, f):
    fr = np.asarray(fr, float)
    deg = 2 if len(fr) >= 4 else 1
    t0 = fr.mean()
    return np.polyval(np.polyfit(fr - t0, xs, deg), f - t0), np.polyval(np.polyfit(fr - t0, ys, deg), f - t0)


def rescue(track, cand, shot):
    """walk into each in-shot gap from both sides along the extrapolated curve, taking candidates the DP skipped
    (e.g. weak hits right after a header changed the ball's direction)."""
    by_f = {f: g for f, g in cand.groupby('frame')}
    n_added = 0
    det = ~track.cx.isna().values
    for s, e in list(gaps_in_shot(det, shot, MAX_FILL)):
        for direction in (1, -1):
            frames = range(s, e + 1) if direction == 1 else range(e, s - 1, -1)
            for f in frames:
                if not np.isnan(track.at[f, 'cx']):
                    continue
                src = [g for g in range(f - direction, f - direction * (3 * FIT_N), -direction)
                       if 0 <= g < len(track) and shot[g] == shot[f] and not np.isnan(track.at[g, 'cx'])][:5]
                if len(src) < 2 or f not in by_f:
                    break
                px, py = extrapolate(src, track.cx.values[src], track.cy.values[src], f)
                k = abs(f - src[0])
                tol = min(RESCUE_TOL[0] + RESCUE_TOL[1] * k, RESCUE_TOL[2])
                g = by_f[f]
                dist = np.hypot(g.cx.values - px, g.cy.values - py)
                if dist.min() > tol:
                    break
                r = g.iloc[int(dist.argmin())]
                track.loc[f, ['cx', 'cy', 'w', 'h', 'conf', 'model', 'src']] = [
                    r.cx, r.cy, r.x2 - r.x1, r.y2 - r.y1, r.conf, r.model, 'rescued']
                n_added += 1
    print('rescued', n_added, 'frames from skipped candidates')


def track_gaps(track, shot):
    """template-track the ball a few frames into the remaining in-shot gaps from both ends (ball in shadow or blurred,
    where both detectors are silent even on upscaled crops). Strict: tight ball-only template, anchor well inside the
    frame, match must move with the ball (a match that stays at the anchor is background), stop at the first miss."""
    import cv2
    det = ~track.cx.isna().values
    cap = cv2.VideoCapture(VIDEO)
    pos = 0
    n_added = 0
    frames = {}
    for s, e in list(gaps_in_shot(det, shot, MAX_FILL)):
        frames = {k: v for k, v in frames.items() if k >= s - 1}   # adjacent gaps can share an anchor frame
        while pos < s - 1:
            cap.grab(); pos += 1
        while pos <= e + 1:
            frames[pos] = cap.read()[1]; pos += 1
        for direction in (1, -1):
            a = s - 1 if direction == 1 else e + 1
            if track.at[a, 'src'] != 'detected':
                continue
            ax, ay = int(track.at[a, 'cx']), int(track.at[a, 'cy'])
            half = max(4, int(min(track.at[a, 'w'], track.at[a, 'h']) * 0.35))     # inner part of the ball only
            if not (TRACK_EDGE < ax < W - TRACK_EDGE and TRACK_EDGE < ay < H - TRACK_EDGE):
                continue
            tmpl = frames[a][ay - half:ay + half, ax - half:ax + half]
            steps = range(s, e + 1) if direction == 1 else range(e, s - 1, -1)
            for f in list(steps)[:TRACK_MAX]:
                if not np.isnan(track.at[f, 'cx']):
                    break
                src = [g for g in range(f - direction, f - direction * 6, -direction)
                       if shot[g] == shot[f] and not np.isnan(track.at[g, 'cx'])][:3]
                if len(src) < 2:
                    break
                px, py = extrapolate(src, track.cx.values[src], track.cy.values[src], f)
                x0, y0 = int(max(0, px - TRACK_R - half)), int(max(0, py - TRACK_R - half))
                x1, y1 = int(min(W, px + TRACK_R + half)), int(min(H, py + TRACK_R + half))
                win = frames[f][y0:y1, x0:x1]
                if win.shape[0] <= 2 * half or win.shape[1] <= 2 * half:
                    break
                _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED))
                mx, my = x0 + loc[0] + half, y0 + loc[1] + half
                moved_pred = np.hypot(px - ax, py - ay); moved = np.hypot(mx - ax, my - ay)
                if score < TRACK_MIN or (moved_pred > 3 * half and moved < half):
                    break
                track.loc[f, ['cx', 'cy', 'w', 'h', 'conf', 'model', 'src']] = [
                    mx, my, track.at[a, 'w'], track.at[a, 'h'], round(score, 3), 'template', 'tracked']
                n_added += 1
    print('tracked', n_added, 'frames by template matching')


def fill(track, shot):
    """track: DataFrame indexed by frame with cx,cy,w,h for detected frames (NaN elsewhere)."""
    det = ~track.cx.isna().values
    src = np.where(det, track.src.fillna('detected').values, 'missing').astype(object)
    reason = np.where(det, '', 'no_detection').astype(object)
    cx, cy = track.cx.values.copy(), track.cy.values.copy()
    bw, bh = track.w.values.copy(), track.h.values.copy()
    n = len(track)
    i = 0
    while i < n:
        if det[i]:
            i += 1; continue
        j = i
        while j < n and not det[j]:
            j += 1
        s, e = i, j - 1          # gap [s, e]
        i = j
        if s == 0 or j >= n or shot[s - 1] != shot[j]:
            # gap touches a cut / video end: no anchor on one side -> leave missing
            continue
        if e - s + 1 > MAX_FILL:
            continue
        sh = shot[s]
        pre = [f for f in range(s - 1, max(-1, s - 1 - 3 * FIT_N), -1) if det[f] and shot[f] == sh][:FIT_N]
        post = [f for f in range(j, min(n, j + 3 * FIT_N)) if det[f] and shot[f] == sh][:FIT_N]
        anchors = np.array(sorted(pre + post))
        gap = np.arange(s, e + 1)
        if len(pre) >= 3 and len(post) >= 3:
            deg = 2
        else:
            deg = 1
        ax, ay = cx[anchors], cy[anchors]
        t0 = anchors.mean()
        px = np.polyfit(anchors - t0, ax, deg); py = np.polyfit(anchors - t0, ay, deg)
        res = np.hypot(np.polyval(px, anchors - t0) - ax, np.polyval(py, anchors - t0) - ay).max()
        if res > FIT_RES and e - s + 1 > 4:
            # trajectory changed inside the gap (touch hidden behind a player): only short gaps get linear fill
            reason[gap] = 'trajectory_break'
            continue
        if res > FIT_RES:
            px = np.polyfit([s - 1, j], [cx[s - 1], cx[j]], 1); py = np.polyfit([s - 1, j], [cy[s - 1], cy[j]], 1)
            gx, gy = np.polyval(px, gap), np.polyval(py, gap)
        else:
            gx, gy = np.polyval(px, gap - t0), np.polyval(py, gap - t0)
        ww = np.interp(gap, [s - 1, j], [bw[s - 1], bw[j]]); hh = np.interp(gap, [s - 1, j], [bh[s - 1], bh[j]])
        inside = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
        cx[gap[inside]] = gx[inside]; cy[gap[inside]] = gy[inside]
        bw[gap[inside]] = ww[inside]; bh[gap[inside]] = hh[inside]
        src[gap[inside]] = 'interpolated'; reason[gap[inside]] = ''
        reason[gap[~inside]] = 'off_screen'
    return cx, cy, bw, bh, src, reason


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    max_frame = int(args[0]) if args else N_FRAMES
    n = max_frame
    shot = shots_of(n)
    cand = drop_static(load_candidates(max_frame), shot)
    cand['shot'] = shot[cand.frame.values]
    picks = []
    for s, d in cand.groupby('shot'):
        d = d.sort_values('frame').reset_index(drop=True)
        picks.append(d.iloc[dp_shot(d)])
    sel = pd.concat(picks).set_index('frame')
    assert sel.index.is_unique
    track = pd.DataFrame(index=np.arange(n))
    track['cx'] = sel.cx; track['cy'] = sel.cy
    track['w'] = sel.x2 - sel.x1; track['h'] = sel.y2 - sel.y1
    track['conf'] = sel.conf; track['model'] = sel.model; track['src'] = np.where(track.cx.isna(), None, 'detected')
    track = track.astype({'model': object, 'src': object})
    rescue(track, cand, shot)
    if '--track' in sys.argv:          # off by default: ~65% precision when checked by eye
        track_gaps(track, shot)
    cx, cy, bw, bh, src, reason = fill(track, shot)
    out = pd.DataFrame({'frame': np.arange(n), 'time_sec': np.arange(n) / FPS})
    out['x1'] = np.clip(cx - bw / 2, 0, W); out['y1'] = np.clip(cy - bh / 2, 0, H)
    out['x2'] = np.clip(cx + bw / 2, 0, W); out['y2'] = np.clip(cy + bh / 2, 0, H)
    out['center_x'] = cx; out['center_y'] = cy
    out['confidence'] = track.conf.values
    out['source'] = src
    out['model'] = track.model.fillna('').values
    out['shot'] = shot
    out['missing_reason'] = reason
    out = out.round(2)
    out.to_csv(os.path.join(ROOT, 'data', 'ball_pipeline', 'ball_boxes.csv'), index=False)

    old = pd.read_csv(os.path.join(ROOT, 'data', 'tracks', 'ball_tracks_full.csv')).iloc[:n]
    print('new:', out.source.value_counts().to_dict())
    print('missing reasons:', out[out.source == 'missing'].missing_reason.value_counts().to_dict())
    print('old:', old.source.value_counts().to_dict())
    both = (out.source == 'detected').values & (old.source == 'yolo').values
    d = np.hypot(out.center_x.values[both] - old.center_x.values[both], out.center_y.values[both] - old.center_y.values[both])
    print(f'frames detected in both: {both.sum()}, picked a different ball (>20px): {(d > 20).sum()}')
    pd.DataFrame({'frame': out.frame[both][d > 20]}).to_csv(os.path.join(ROOT, 'data', 'ball_pipeline', 'changed_picks.csv'), index=False)


if __name__ == '__main__':
    main()
