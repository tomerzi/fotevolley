"""Ball height for every frame of the video.

usage: python reconstruct.py [out_dir]

Reads the reviewed ball boxes + player pose, the camera calibration (camera.json), the shadow candidates and the
per-shot sun direction, and writes
    ball_height.csv     one row per video frame (height of the ball centre above the sand + 3-D position + quality)
    ball_contacts.csv   one row per detected touch / bounce (where, how high, who)
    flights.csv         one row per reconstructed flight (diagnostics)
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera
from data import *
from players import Players, J, ARM
import pipeline as pp, segment as sg, chain as ch, sun as sunmod

HERE = os.path.dirname(os.path.abspath(__file__))
X_NET = 7.90            # net plane in court coordinates (measured from the edge-on net in the image)


# ---------------------------------------------------------------------------------------------------------------
def junction_has_cause(jn, x_net=X_NET):
    """is there a physical reason for the ball to change velocity at this flight/flight junction?"""
    a, b = jn['a']['flight'], jn['b']['flight']
    tau = jn['tau']
    p = a.pos(tau)[0]
    dv = np.linalg.norm(b.vel(tau)[0] - a.vel(tau)[0])
    jn['dv'] = float(dv); jn['p'] = p
    if dv < 1.0:
        return False
    c = jn['contact']
    if p[2] < 3.3:
        if jn['player_xy'] is not None and np.linalg.norm(p[:2] - jn['player_xy']) < 1.7:
            return True
        if c is not None and c['rel'] < 0.5:
            return True
    if p[2] < 0.45:
        return True                                        # bounce on the sand
    if abs(p[0] - x_net) < 0.35 and p[2] < 2.55:
        return True                                        # net
    return False


def merge_false_splits(cam, ball_shot, players, cd, sun_s, out, kd, verbose=False):
    """the DP may cut one flight in two (model error, a gap while the ball is out of the picture). Junctions without a
    physical cause are removed when a single flight explains both parts."""
    changed = True
    while changed:
        changed = False
        for jn in out['junctions']:
            if jn['kinds'] != ('flight', 'flight') or jn.get('loose'):
                continue
            if junction_has_cause(jn):
                continue
            sa, sb = jn['a'], jn['b']
            fa_, fb_ = sa['flight'], sb['flight']
            segs = []
            for s in out['segs']:
                if s is sb:
                    continue
                s2 = {k: v for k, v in s.items() if k != 'flight'}
                if s is sa:
                    s2['j'] = sb['j']; s2['fb'] = sb['fb']
                segs.append(s2)
            trial = pp.run_shot(cam, ball_shot, players, cd, sun_s, kd=kd, segs=segs)
            fm = [s for s in trial['segs'] if s['kind'] == 'flight' and s['i'] == sa['i']][0]['flight']
            lim = 1.3 * max(fa_.rms, fb_.rms) + 0.5
            ok = fm.rms < lim and (np.isnan(fm.shadow_rms) or fm.shadow_rms < max(0.07, 1.6 * np.nanmax([fa_.shadow_rms, fb_.shadow_rms, 0.0])))
            if verbose:
                print('   merge test %d|%d dv %.1f z %.2f: rms %.1f/%.1f -> %.1f shadow %.2f : %s' % (
                    sa['fb'], sb['fa'], jn['dv'], jn['p'][2], fa_.rms, fb_.rms, fm.rms, fm.shadow_rms, 'MERGED' if ok else 'kept'))
            if ok:
                out = trial
                changed = True
                break
            else:
                jn['kept_after_test'] = True
        # do not test the same junction twice
        if changed:
            continue
    return out


# ---------------------------------------------------------------------------------------------------------------
def is_held(cam, players, frames, uv, seg):
    """a 'flight' whose ball stays next to a player's hand for the whole segment is a carried ball"""
    ks = np.arange(seg['i'], seg['j'])
    near = 0
    for k in ks:
        nn = players.nearest(int(frames[k]), uv[k])
        if nn:
            pl, j, d = nn[0]
            sc = players.scale_px(pl, int(frames[k]))
            if np.isfinite(sc) and sc > 20 and d / sc < 0.30 and j in ARM:
                near += 1
    return near >= 0.7 * len(ks) and len(ks) >= 5


def player_depth_estimate(cam, players, frame, uv):
    """ball position assuming it is at the depth of the nearest player (ball carried / touched)"""
    nn = players.nearest(int(frame), uv)
    if not nn:
        return None
    pl, j, d = nn[0]
    sc = players.scale_px(pl, int(frame))
    if not np.isfinite(sc) or sc < 20 or d / sc > 0.9:
        return None
    g = players.ground_xy(pl, int(frame))
    if g is None:
        return None
    n = np.array([g[0] - cam.C[0], g[1] - cam.C[1], 0.0]); n /= np.linalg.norm(n)
    P = cam.to_vertical_plane(np.array([uv], float), np.array([g[0], g[1], 0.0]), n)[0]
    if P[2] < BALL_R:
        P = cam.to_plane(np.array([uv], float), BALL_R)[0]
    return P, pl, d / sc


def assemble(cam, players, ball_shot, out, sun_s):
    """per-frame table for one shot"""
    fr_all = ball_shot.frame.values
    n = len(fr_all)
    f_first = int(fr_all[0])
    cols = dict(x_m=np.full(n, np.nan), y_m=np.full(n, np.nan), z_m=np.full(n, np.nan), z_sigma_m=np.full(n, np.nan),
                z_shadow_m=np.full(n, np.nan), reproj_err_px=np.full(n, np.nan))
    state = np.array(['unknown'] * n, dtype=object)
    method = np.array([''] * n, dtype=object)
    flight_id = np.full(n, -1)
    shadow_used = np.zeros(n, bool)
    if out is None:
        return cols, state, method, flight_id, shadow_used, [], []
    frames, uv, t, f0 = out['frames'], out['uv'], out['t'], out['f0']
    obs_index = {int(f): k for k, f in enumerate(frames)}
    rays = cam.rays(uv)
    shad = out.get('shad')
    chosen = out.get('chosen')

    def put(f, P, st, me, sig, fid=-1):
        i = int(f) - f_first
        if 0 <= i < n:
            cols['x_m'][i], cols['y_m'][i], cols['z_m'][i] = P
            cols['z_sigma_m'][i] = sig
            state[i] = st; method[i] = me; flight_id[i] = fid

    flights_rows, fid = [], 0
    for s in out['segs']:
        if s['kind'] == 'flight':
            fl = s['flight']
            held = is_held(cam, players, frames, uv, s)
            fa = int(np.ceil(f0 + fl.t_start * FPS - 1e-6)); fb = int(np.floor(f0 + fl.t_end * FPS + 1e-6))
            ff = np.arange(fa, fb + 1)
            if len(ff) == 0:
                continue
            tt = (ff - f0) / FPS
            P = fl.pos(tt); sg_ = fl.sigma(tt)[:, 2]
            ns = fl.n_shadow
            if held:
                me = 'held'
            elif ns >= 4:
                me = 'flight+shadow'
            else:
                me = 'flight'
            for f, p, sz in zip(ff, P, sg_):
                put(f, p, 'held' if held else 'flight', me, sz, fid)
            ks = np.arange(fl.i, fl.j)
            for k in ks:
                i = int(frames[k]) - f_first
                cols['reproj_err_px'][i] = out['res'][k]
                if chosen is not None and chosen[k] >= 0:
                    shadow_used[i] = True
            tm = np.linspace(fl.t_start, fl.t_end, 60); zz = fl.pos(tm)[:, 2]
            pa, pb = fl.pos(fl.t_start)[0], fl.pos(fl.t_end)[0]
            flights_rows.append(dict(flight_id=fid, frame_start=fa, frame_end=fb, n_obs=len(ks), n_shadow=ns, state='held' if held else 'flight',
                                     x0=pa[0], y0=pa[1], z0=pa[2], x1=pb[0], y1=pb[1], z1=pb[2], z_max=zz.max(), z_min=zz.min(),
                                     speed0=np.linalg.norm(fl.vel(fl.t_start)), rms_px=fl.rms, shadow_rms_m=fl.shadow_rms,
                                     sigma_z_mid=fl.sigma([0.5 * (fl.t_start + fl.t_end)])[0, 2],
                                     ae_x=fl.ae[0], ae_y=fl.ae[1], ae_z=fl.ae[2], rich=fl.rich))
            fid += 1
        elif s['kind'] == 'ground':
            ks = np.arange(s['i'], s['j'])
            Pk = cam.to_plane(uv[ks], BALL_R)
            ff = np.arange(int(frames[ks[0]]), int(frames[ks[-1]]) + 1)
            for d in range(3):
                pass
            Pi = np.stack([np.interp(ff, frames[ks], Pk[:, d]) for d in range(3)], 1)
            for f, p in zip(ff, Pi):
                put(f, p, 'ground', 'ground', 0.03)
    # direct per-frame shadow triangulation (independent of any dynamics), wherever a matched shadow exists
    if shad is not None and sun_s is not None:
        for k, f in enumerate(frames):
            i = int(f) - f_first
            if chosen is not None and chosen[k] >= 0:
                Pt, lam, miss = sunmod.triangulate(cam, rays[k:k + 1], shad[k][chosen[k]][None], sun_s)
                cols['z_shadow_m'][i] = Pt[0, 2]
    # frames that are neither flight nor ground: ball carried / pinned / unclear
    for s in out['segs']:
        if s['kind'] != 'other':
            continue
        for k in range(s['i'], s['j']):
            f = int(frames[k]); i = f - f_first
            est = player_depth_estimate(cam, players, f, uv[k])
            Ps = None
            if shad is not None and shad[k] is not None and sun_s is not None:
                Pc, lam, miss = sunmod.triangulate(cam, np.repeat(rays[k:k + 1], len(shad[k]), 0), shad[k], sun_s)
                if est is not None:
                    d = np.linalg.norm(Pc - est[0], axis=1)
                    if d.min() < 1.0:
                        Ps = Pc[np.argmin(d)]
                elif len(Pc) == 1 and Pc[0, 2] < 2.5:
                    Ps = Pc[0]
            if Ps is not None:
                put(f, Ps, 'held', 'shadow', 0.08); cols['z_shadow_m'][i] = Ps[2]; shadow_used[i] = True
            elif est is not None:
                put(f, est[0], 'held', 'player-depth', 0.25)
    # fill short holes between known frames by interpolation
    z = cols['z_m']
    known = np.nonzero(np.isfinite(z))[0]
    if len(known) > 1:
        for a, b in zip(known[:-1], known[1:]):
            if 1 < b - a <= 12:
                for i in range(a + 1, b):
                    w = (i - a) / (b - a)
                    for c in ('x_m', 'y_m', 'z_m'):
                        cols[c][i] = (1 - w) * cols[c][a] + w * cols[c][b]
                    cols['z_sigma_m'][i] = max(cols['z_sigma_m'][a], cols['z_sigma_m'][b]) + 0.15
                    state[i] = state[a] if state[a] == state[b] else 'unknown'
                    method[i] = 'interp'
    # contacts
    contacts = []
    for jn in out['junctions']:
        ka, kb = jn['kinds']
        if 'flight' not in (ka, kb) or 'other' in (ka, kb):
            continue
        fl = jn['a']['flight'] if ka == 'flight' else jn['b']['flight']
        loose = jn.get('loose', False)
        tau = (jn['ta'] + 0.5 / FPS) if loose else jn['tau']
        if ka != 'flight':
            tau = (jn['tb'] - 0.5 / FPS) if loose else jn['tau']
        p = fl.pos(tau)[0]
        c = jn['contact']
        kind = 'unknown'
        who, part, rel = '', '', np.nan
        if c is not None:
            rel = c['rel']
        near_player = c is not None and c['rel'] < 0.6 and p[2] < 3.3
        if 'ground' in (ka, kb) or p[2] < 0.3:
            kind = 'sand'
            if near_player and c['rel'] < 0.35 and p[2] > 0.2:
                kind = 'player'
        elif near_player:
            kind = 'player'
        elif abs(p[0] - X_NET) < 0.35 and p[2] < 2.55:
            kind = 'net'
        if kind == 'player':
            who, part = c['player'], JOINTS[c['joint']]
        v_in = jn['a']['flight'].vel(tau)[0] if ka == 'flight' else np.full(3, np.nan)
        v_out = jn['b']['flight'].vel(tau)[0] if kb == 'flight' else np.full(3, np.nan)
        contacts.append(dict(frame=round(f0 + tau * FPS, 2), x_m=p[0], y_m=p[1], z_m=p[2], kind=kind, player=who, body_part=part,
                             ball_to_joint_m=rel, speed_in=np.linalg.norm(v_in), speed_out=np.linalg.norm(v_out),
                             between=ka + '>' + kb))
    return cols, state, method, flight_id, shadow_used, flights_rows, contacts


def main(out_dir=HERE, shots=None, verbose=True, kd=ch.KD_DEFAULT, cam=None, use_shadow=True, write=True):
    cam = cam or Camera.from_json(os.path.join(HERE, 'camera.json'))
    ball = load_ball()
    pose = load_pose()
    players = Players(cam, pose, int(ball.frame.max()) + 1)
    cd = pp.cands_dict(pd.read_csv(os.path.join(HERE, 'shadow_candidates.csv'))) if use_shadow else None
    sun = pd.read_csv(os.path.join(HERE, 'sun_per_shot.csv')).set_index('shot')
    rows, all_flights, all_contacts, outs = [], [], [], {}
    fid_off = 0
    for shot, g in ball.groupby('shot'):
        if shots is not None and shot not in shots:
            continue
        sun_s = sun.loc[shot, ['a_fit', 'b_fit']].values.astype(float) if (shot in sun.index and use_shadow) else None
        out = pp.run_shot(cam, g, players, cd, sun_s, kd=kd)
        if out is not None:
            out = merge_false_splits(cam, g, players, cd, sun_s, out, kd, verbose=False)
        outs[shot] = out
        cols, state, method, flight_id, shadow_used, frows, contacts = assemble(cam, players, g, out, sun_s)
        d = pd.DataFrame(dict(frame=g.frame.values, time_sec=g.time_sec.values, shot=shot, state=state, **cols, method=method,
                              ball_observed=g.obs.values, shadow_used=shadow_used,
                              flight_id=np.where(flight_id >= 0, flight_id + fid_off, -1)))
        for r in frows:
            r['flight_id'] += fid_off; r['shot'] = shot
        for c in contacts:
            c['shot'] = shot
        fid_off += len(frows)
        rows.append(d); all_flights += frows; all_contacts += contacts
        if verbose:
            nf = sum(1 for r in frows if r['state'] == 'flight')
            print('shot %2d frames %5d-%5d: %2d flights, z known in %3d/%3d frames, shadow in %3d' % (
                shot, g.frame.min(), g.frame.max(), nf, np.isfinite(d.z_m).sum(), len(d), d.shadow_used.sum()), flush=True)
    res = pd.concat(rows, ignore_index=True)
    fl = pd.DataFrame(all_flights)
    ct = pd.DataFrame(all_contacts)
    if write:
        res.round(dict(x_m=3, y_m=3, z_m=3, z_sigma_m=3, z_shadow_m=3, reproj_err_px=2)).to_csv(os.path.join(out_dir, 'ball_height.csv'), index=False)
        fl.round(3).to_csv(os.path.join(out_dir, 'flights.csv'), index=False)
        ct.round(3).to_csv(os.path.join(out_dir, 'ball_contacts.csv'), index=False)
    return res, fl, ct, outs


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
