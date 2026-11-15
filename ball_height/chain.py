"""Joint 3-D reconstruction of all ballistic flights of one shot.

Unknowns: initial position and velocity (6 numbers) of every flight segment.
Equations (all linear, solved by iteratively re-weighted least squares):
  * camera: reprojection of every observed ball centre (2 per observation),
  * sun:    the ball's shadow on the sand, when it is visible (2 per observation) - a second "view",
  * continuity of the ball position at every junction between two flights (a touch by a player / bounce),
  * a flight that starts / ends on the sand is tied to the ground position seen through the homography,
  * the touching player's position on the sand as a weak, robust prior for the junction,
  * the ball never goes below the sand.
Air drag (quadratic) is included as a known displacement that is re-integrated between iterations.
"""
import numpy as np
from data import G, BALL_R, FPS
import segment as sg
import sun as sunmod

GVEC = np.array([0.0, 0.0, -G])
KD_DEFAULT = 0.012      # drag constant k = rho*Cd*A/(2m)  [1/m]  (Cd~0.23; tuned on the shadow-tracked flights)
BALL_D_EFF = 0.205      # apparent ball diameter of the detector box [m] (calibrated on the shadow-tracked flights)
# typical unmodelled acceleration seen in the shadow-tracked flights (wind, spin, scale); used as the prior mean
# for flights that have no shadow.  Overwritten from physics.json when present.
AE_MEAN = np.zeros(3)
try:
    import json as _json, os as _os
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'physics.json')
    if _os.path.exists(_p):
        AE_MEAN = np.array(_json.load(open(_p))['ae_mean'], float)
except Exception:
    pass


def integrate_drag(p0, v0, t_lo, t_hi, kd, h=0.008, gvec=GVEC):
    """trajectory with quadratic drag through (p0,v0) at t=0, sampled on a fine grid covering [t_lo, t_hi]."""
    def acc(v):
        return gvec - kd * np.linalg.norm(v) * v

    def run(T):
        n = max(1, int(np.ceil(abs(T) / h)))
        hh = T / n
        ps, vs = [p0.copy()], [v0.copy()]
        p, v = p0.copy(), v0.copy()
        for _ in range(n):
            k1v = acc(v); k1p = v
            k2v = acc(v + 0.5 * hh * k1v); k2p = v + 0.5 * hh * k1v
            k3v = acc(v + 0.5 * hh * k2v); k3p = v + 0.5 * hh * k2v
            k4v = acc(v + hh * k3v); k4p = v + hh * k3v
            p = p + hh / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            v = v + hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            ps.append(p.copy()); vs.append(v.copy())
        return np.arange(n + 1) * hh, np.array(ps), np.array(vs)

    tf, pf, vf = run(max(t_hi, 1e-6))
    if t_lo < 0:
        tb, pb, vb = run(t_lo)
        return np.concatenate([tb[:0:-1], tf]), np.concatenate([pb[:0:-1], pf]), np.concatenate([vb[:0:-1], vf])
    return tf, pf, vf


class Flight:
    def __init__(self, seg, t, kd, gvec=GVEC):
        self.seg = seg
        self.i, self.j = seg['i'], seg['j']
        self.tref = t[self.i]                         # reference time of (p0, v0)
        self.gvec = gvec
        th = np.array(seg['theta'], float)            # DP parameters are at t=0 of the shot -> move to tref
        self.p0 = sg.flight_pos(th, self.tref)
        self.v0 = th[3:] + GVEC * self.tref
        self.kd = kd
        self.ae = np.zeros(3)                         # unmodelled constant acceleration (spin / wind / drag mismatch)
        self.rich = False
        self.t_start, self.t_end = t[self.i], t[self.j - 1]
        self._grid = None
        self.cov = None
        self.n_shadow = 0

    def grid(self):
        if self._grid is None:
            lo = min(self.t_start, self.tref) - self.tref - 0.12
            hi = max(self.t_end, self.tref) - self.tref + 0.12
            self._grid = integrate_drag(self.p0, self.v0, lo, hi, self.kd, gvec=self.gvec + self.ae)
        return self._grid

    def pos(self, t):
        tg, pg, _ = self.grid()
        dt = np.atleast_1d(np.asarray(t, float)) - self.tref
        return np.stack([np.interp(dt, tg, pg[:, k]) for k in range(3)], -1)

    def vel(self, t):
        tg, _, vg = self.grid()
        dt = np.atleast_1d(np.asarray(t, float)) - self.tref
        return np.stack([np.interp(dt, tg, vg[:, k]) for k in range(3)], -1)

    def delta(self, t):
        """drag displacement: pos(t) - vacuum trajectory through the same (p0, v0)"""
        dt = np.atleast_1d(np.asarray(t, float)) - self.tref
        vac = self.p0 + self.v0 * dt[:, None] + 0.5 * (self.gvec + self.ae) * dt[:, None] ** 2
        return self.pos(t) - vac

    def basis(self, t):
        """(3,9) matrix B and offset c with pos(t) = B theta + c + delta(t),  theta = (p0, v0, a_extra)"""
        dt = float(t) - self.tref
        return np.concatenate([np.eye(3), dt * np.eye(3), 0.5 * dt * dt * np.eye(3)], 1), 0.5 * self.gvec * dt * dt

    def sigma(self, t):
        """formal 1-sigma uncertainty (x,y,z) of the position at time(s) t"""
        if self.cov is None:
            return np.full((len(np.atleast_1d(t)), 3), np.nan)
        out = []
        for tt in np.atleast_1d(t):
            B, _ = self.basis(tt)
            out.append(np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', B, self.cov, B), 0)))
        return np.array(out)


def ransac_shadow(cam, t, rays, shad, sun_s, ks, tref, rng, tol=0.25, max_pairs=2500):
    """find the ballistic arc supported by the largest number of shadow candidates of one flight.
    returns dict obs index -> candidate index, and (p0, v0) at tref; or None"""
    kk, cc, S = [], [], []
    for k in ks:
        if shad[k] is not None and len(shad[k]):
            for c, s in enumerate(shad[k]):
                kk.append(k); cc.append(c); S.append(s)
    if len(set(kk)) < 4:
        return None
    kk, cc, S = np.array(kk), np.array(cc), np.array(S)
    P, lam, miss = sunmod.triangulate(cam, rays[kk], S, sun_s)
    tt = t[kk] - tref
    dur = t[ks[-1]] - t[ks[0]]
    ii, jj = np.nonzero((tt[None, :] - tt[:, None]) >= max(0.12, 0.3 * dur))
    if len(ii) == 0:
        return None
    if len(ii) > max_pairs:
        sel = rng.choice(len(ii), max_pairs, replace=False); ii, jj = ii[sel], jj[sel]
    ti, tj = tt[ii][:, None], tt[jj][:, None]
    v0 = (P[jj] - P[ii] - 0.5 * GVEC * (tj ** 2 - ti ** 2)) / (tj - ti)
    p0 = P[ii] - v0 * ti - 0.5 * GVEC * ti ** 2
    M = p0[:, None, :] + v0[:, None, :] * tt[None, :, None] + 0.5 * GVEC * (tt ** 2)[None, :, None]     # (pairs, pts, 3)
    dist = np.linalg.norm(M - P[None], axis=2)
    inl = dist < tol
    # count distinct frames
    uniq, inv = np.unique(kk, return_inverse=True)
    cnt = np.zeros((len(ii), len(uniq)), bool)
    for u in range(len(uniq)):
        cnt[:, u] = inl[:, inv == u].any(1)
    score = cnt.sum(1).astype(float)
    ok = (np.linalg.norm(v0, axis=1) < 40) & (M[..., 2].min(1) > -0.5)
    score[~ok] = 0
    b = int(np.argmax(score))
    if score[b] < max(4, 0.3 * len(uniq)):
        return None
    # refine on inliers (vacuum model) and re-select
    sel = inl[b]
    for _ in range(2):
        A = np.concatenate([np.eye(3)[None].repeat(sel.sum(), 0), tt[sel][:, None, None] * np.eye(3)[None]], 2).reshape(-1, 6)
        rhs = (P[sel] - 0.5 * GVEC * (tt[sel] ** 2)[:, None]).reshape(-1)
        th = np.linalg.lstsq(A, rhs, rcond=None)[0]
        Mb = th[:3] + th[3:] * tt[:, None] + 0.5 * GVEC * (tt ** 2)[:, None]
        d = np.linalg.norm(Mb - P, axis=1)
        sel = d < tol
        if sel.sum() < 4:
            return None
    chosen = {}
    for k in set(kk[sel]):
        m = np.nonzero((kk == k) & sel)[0]
        chosen[int(k)] = int(cc[m[np.argmin(d[m])]])
    return chosen, th[:3], th[3:]


def solve_shot(cam, frames, uv, w_obs, segs, players=None, kd=KD_DEFAULT, sun_s=None, shad=None, opts=None, gscale=1.0,
               size=None, ae_mean=None):
    """frames/uv/w_obs: observations of the shot (as given to the DP); segs: DP output.
    sun_s: sun slope (a,b) of the shot; shad: per-observation arrays (m,2) of candidate shadow ground points.
    returns dict with flights, junctions, diagnostics"""
    o = dict(sig_px=2.0, huber=2.5, sig_cont=0.04, sig_ground=0.10, sig_player=0.4, player_rel=1.1, use_player=True,
             n_iter=9, sig_other_cont=0.6, sig_shadow=0.06, gate=0.30, use_shadow=True, sig_ae=2.0, sig_ae_poor=0.05,
             sig_size_rel=0.12, use_size=True, sig_pos=(12.0, 6.0), sig_vel=30.0)
    o.update(opts or {})
    gvec = GVEC * gscale
    ae_mean = AE_MEAN if ae_mean is None else np.asarray(ae_mean, float)
    f0 = frames[0]
    t = (frames - f0) / FPS
    n_obs = len(frames)
    a1, a2 = sg.rows(cam, uv)
    rays = cam.rays(uv)
    mag = 1 + 2 * cam.k1 * (cam.undistort_norm(uv) ** 2).sum(1)      # local magnification of the lens distortion
    use_sh = o['use_shadow'] and sun_s is not None and shad is not None
    flights = []
    for s in segs:
        if s['kind'] == 'flight':
            fl = Flight(s, t, kd, gvec); s['flight'] = fl; flights.append(fl)
    res = np.zeros(n_obs)
    chosen = np.full(n_obs, -1)
    if not flights:
        return dict(flights=[], junctions=[], segs=segs, t=t, f0=f0, res=res, chosen=chosen)
    idx = {id(fl): k for k, fl in enumerate(flights)}
    nf = len(flights)
    NP = 9

    # ---- junctions -----------------------------------------------------------------------------------
    junctions = []
    for a, b in zip(segs[:-1], segs[1:]):
        ta, tb = t[a['j'] - 1], t[b['i']]
        junctions.append(dict(a=a, b=b, ta=ta, tb=tb, tau=0.5 * (ta + tb), kinds=(a['kind'], b['kind'])))
    for k, s in enumerate(segs):            # an 'other' run between two modelled segments = one loose junction
        if s['kind'] == 'other' and 0 < k < len(segs) - 1:
            a, b = segs[k - 1], segs[k + 1]
            if a['kind'] != 'other' and b['kind'] != 'other':
                junctions.append(dict(a=a, b=b, ta=t[a['j'] - 1], tb=t[b['i']], tau=None, kinds=(a['kind'], b['kind']),
                                      loose=True, other=s))

    def set_spans():
        for fl in flights:
            fl.t_start, fl.t_end = t[fl.i], t[fl.j - 1]
        for jn in junctions:
            if jn.get('loose'):
                if jn['a']['kind'] == 'flight': jn['a']['flight'].t_end = jn['ta'] + 0.5 / FPS
                if jn['b']['kind'] == 'flight': jn['b']['flight'].t_start = jn['tb'] - 0.5 / FPS
                continue
            if 'other' in jn['kinds']:
                continue
            if jn['a']['kind'] == 'flight': jn['a']['flight'].t_end = jn['tau']
            if jn['b']['kind'] == 'flight': jn['b']['flight'].t_start = jn['tau']
        for fl in flights:
            fl._grid = None
    set_spans()

    def ground_at(seg, tt):
        return sg.ground_pos(np.asarray(seg['theta']), tt)

    def find_contact(jn):
        if players is None:
            return None
        best = None
        ka, kb = jn['a']['j'] - 1, jn['b']['i']
        for k in ([ka, kb] if not jn.get('loose') else list(range(ka, kb + 1))):
            fr = int(frames[k])
            for pl, j, d in players.nearest(fr, uv[k])[:2]:
                sc = players.scale_px(pl, fr)
                if not np.isfinite(sc) or sc < 20:
                    continue
                rel = d / sc                                   # ~ metres in the image plane
                if best is None or rel < best['rel']:
                    best = dict(player=pl, joint=j, rel=rel, frame=fr, px=d)
        return best
    for jn in junctions:
        jn['contact'] = find_contact(jn)
        c = jn['contact']
        jn['player_xy'] = None
        if c is not None and c['rel'] < o['player_rel']:
            g = players.ground_xy(c['player'], c['frame'])
            if g is not None:
                jn['player_xy'] = g

    # ---- shadow initialisation: RANSAC per flight ----------------------------------------------------------
    rng = np.random.default_rng(0)
    if use_sh:
        for fl in flights:
            ks = np.arange(fl.i, fl.j)
            r = ransac_shadow(cam, t, rays, shad, sun_s, ks, fl.tref, rng)
            if r is not None:
                ch, p0, v0 = r
                for k, c in ch.items():
                    chosen[k] = c
                fl.p0, fl.v0 = p0, v0; fl._grid = None; fl.ransac = len(ch)

    active_floor = {}
    depth_w = sg.ref_weight(cam, uv)
    hub = np.ones(n_obs)
    hub_s = np.ones(n_obs)
    Am = None
    for it in range(o['n_iter']):
        A, b = [], []

        def add(rowsA, rowsb):
            A.append(rowsA); b.append(rowsb)
        for fl in flights:
            c0 = NP * idx[id(fl)]
            ks = np.arange(fl.i, fl.j)
            dt = t[ks] - fl.tref
            dl = fl.delta(t[ks]) if it > 0 or getattr(fl, 'ransac', 0) else np.zeros((len(ks), 3))
            w = depth_w[ks] * w_obs[ks] * hub[ks] / o['sig_px']
            for arow in (a1, a2):
                ar = arow[ks]
                R = np.zeros((len(ks), NP * nf))
                R[:, c0:c0 + 3] = ar * w[:, None]
                R[:, c0 + 3:c0 + 6] = ar * (w * dt)[:, None]
                R[:, c0 + 6:c0 + 9] = ar * (w * 0.5 * dt ** 2)[:, None]
                add(R, (ar @ cam.C - (ar @ gvec) * 0.5 * dt ** 2 - (ar * dl).sum(1)) * w)
            # prior on the extra acceleration: free only when the shadow pins the 3-D path
            sae = o['sig_ae'] if fl.rich else o['sig_ae_poor']
            R = np.zeros((3, NP * nf)); R[:, c0 + 6:c0 + 9] = np.eye(3) / sae
            add(R, ae_mean / sae)
            # very weak priors that only matter when a flight is (almost) unconstrained: stay around the court, sane speed
            tm = 0.5 * (t[ks[0]] + t[ks[-1]])
            B, c = fl.basis(tm)
            R = np.zeros((2, NP * nf)); R[:, c0:c0 + NP] = B[:2] / np.array(o['sig_pos'])[:, None]
            add(R, (np.array([8.0, 4.0]) - c[:2]) / np.array(o['sig_pos']))
            R = np.zeros((3, NP * nf)); R[:, c0 + 3:c0 + 6] = np.eye(3) / o['sig_vel']
            add(R, np.zeros(3))
            # apparent ball size as a weak range measurement (keeps camera-only flights from drifting in depth)
            if o['use_size'] and size is not None:
                rg = BALL_D_EFF * cam.f / size[ks] * mag[ks]
                okz = np.isfinite(rg)
                ws = np.where(okz, 1.0 / (o['sig_size_rel'] * np.where(okz, rg, 1.0)), 0.0) * w_obs[ks]
                ry = rays[ks]
                R = np.zeros((len(ks), NP * nf))
                R[:, c0:c0 + 3] = ry * ws[:, None]
                R[:, c0 + 3:c0 + 6] = ry * (ws * dt)[:, None]
                R[:, c0 + 6:c0 + 9] = ry * (ws * 0.5 * dt ** 2)[:, None]
                add(R, (np.where(okz, rg, 0.0) + ry @ cam.C - (ry @ gvec) * 0.5 * dt ** 2 - (ry * dl).sum(1)) * ws)
            if use_sh:
                sel = chosen[ks] >= 0
                if sel.any():
                    kk = ks[sel]
                    S = np.array([shad[k][chosen[k]] for k in kk])
                    ws = hub_s[kk] / o['sig_shadow']
                    dts = t[kk] - fl.tref
                    dls = dl[sel]
                    for ax in (0, 1):
                        R = np.zeros((len(kk), NP * nf))
                        R[:, c0 + ax] = ws
                        R[:, c0 + 2] = sun_s[ax] * ws
                        R[:, c0 + 3 + ax] = ws * dts
                        R[:, c0 + 5] = sun_s[ax] * ws * dts
                        R[:, c0 + 6 + ax] = ws * 0.5 * dts ** 2
                        R[:, c0 + 8] = sun_s[ax] * ws * 0.5 * dts ** 2
                        cz = 0.5 * gvec[2] * dts ** 2 + dls[:, 2]
                        add(R, (S[:, ax] - dls[:, ax] - cz * sun_s[ax]) * ws)
        for jn in junctions:
            ka, kb = jn['kinds']
            if 'other' in (ka, kb) or (ka == 'ground' and kb == 'ground'):
                continue
            loose = jn.get('loose', False)
            taua = jn['ta'] + 0.5 / FPS if loose else jn['tau']
            taub = jn['tb'] - 0.5 / FPS if loose else jn['tau']
            sig = o['sig_other_cont'] + 3.0 * max(0.0, taub - taua) if loose else o['sig_cont']
            R = np.zeros((3, NP * nf)); rhs = np.zeros(3)
            if ka == 'flight':
                fa_ = jn['a']['flight']; B, c = fa_.basis(taua); c0 = NP * idx[id(fa_)]
                R[:, c0:c0 + NP] += B; rhs -= c + fa_.delta(taua)[0]
            else:
                rhs -= ground_at(jn['a'], taua)
            if kb == 'flight':
                fb_ = jn['b']['flight']; B, c = fb_.basis(taub); c0 = NP * idx[id(fb_)]
                R[:, c0:c0 + NP] -= B; rhs += c + fb_.delta(taub)[0]
            else:
                rhs += ground_at(jn['b'], taub)
            if 'ground' in (ka, kb) and not loose:
                sig = o['sig_ground']
            add(R / sig, rhs / sig)
            if o['use_player'] and jn['player_xy'] is not None and ka == 'flight' and kb == 'flight':
                fa_ = jn['a']['flight']; B, c = fa_.basis(taua); c0 = NP * idx[id(fa_)]
                R = np.zeros((2, NP * nf)); R[:, c0:c0 + NP] = B[:2]
                rhs = jn['player_xy'] - (c + fa_.delta(taua)[0])[:2]
                w = jn.get('pw', 1.0) / o['sig_player']
                add(R * w, rhs * w)
        for (k, which) in active_floor:
            fl = flights[k]; tt = fl.t_start if which == 'start' else fl.t_end
            B, c = fl.basis(tt)
            R = np.zeros((1, NP * nf)); R[0, NP * k:NP * k + NP] = B[2]
            add(R / 0.03, np.array([(BALL_R - c[2] - fl.delta(tt)[0, 2]) / 0.03]))
        Am, bm = np.concatenate(A), np.concatenate(b)
        th, *_ = np.linalg.lstsq(Am, bm, rcond=None)
        for k, fl in enumerate(flights):
            fl.p0, fl.v0, fl.ae = th[NP * k:NP * k + 3], th[NP * k + 3:NP * k + 6], th[NP * k + 6:NP * k + 9]
            fl._grid = None
        # ---- update weights / junction times / floor set / shadow choice ---------------------------------
        sres = np.full(n_obs, np.nan)
        for fl in flights:
            ks = np.arange(fl.i, fl.j)
            P = fl.pos(t[ks])
            pr = cam.project(P)
            res[ks] = np.hypot(*(pr - uv[ks]).T)
            depth_w[ks] = cam.f / np.maximum((P - cam.C) @ cam.R[2], 1.0)
            if use_sh:
                Sp = sunmod.shadow_of(P, sun_s)
                gate = o['gate'] * (2.0 if it < 2 and not getattr(fl, 'ransac', 0) else 1.0)
                for n_, k in enumerate(ks):
                    chosen[k] = -1
                    if shad[k] is not None and len(shad[k]) and P[n_, 2] > 0.3:
                        d = np.linalg.norm(shad[k] - Sp[n_], axis=1)
                        c = int(np.argmin(d))
                        if d[c] < gate:
                            chosen[k] = c; sres[k] = d[c]
                sel = ks[chosen[ks] >= 0]
                fl.rich = len(sel) >= 8 and (t[sel[-1]] - t[sel[0]]) >= 0.5 * (t[ks[-1]] - t[ks[0]])
        hub = np.where(res > o['huber'] * o['sig_px'], o['huber'] * o['sig_px'] / np.maximum(res, 1e-9), 1.0)
        hs = o['huber'] * o['sig_shadow']
        hub_s = np.where(np.nan_to_num(sres) > hs, hs / np.maximum(np.nan_to_num(sres), 1e-9), 1.0)
        for jn in junctions:
            if jn.get('loose') or 'other' in jn['kinds']:
                continue
            cand = np.linspace(jn['ta'], jn['tb'], 11)

            def P_of(seg, tt):
                return seg['flight'].pos(tt) if seg['kind'] == 'flight' else ground_at(seg, tt)
            d = np.linalg.norm(P_of(jn['a'], cand) - P_of(jn['b'], cand), axis=1)
            jn['tau'] = float(cand[np.argmin(d)]); jn['gap'] = float(d.min())
        set_spans()
        for jn in junctions:
            if jn['player_xy'] is not None and jn['kinds'] == ('flight', 'flight'):
                taua = jn['ta'] + 0.5 / FPS if jn.get('loose') else jn['tau']
                d = np.linalg.norm(jn['a']['flight'].pos(taua)[0, :2] - jn['player_xy'])
                jn['pdist'] = float(d)
                jn['pw'] = 1.0 if d < 1.0 else 1.0 / d
        for k, fl in enumerate(flights):
            for which, tt in (('start', fl.t_start), ('end', fl.t_end)):
                if fl.pos(tt)[0, 2] < BALL_R - 0.02:
                    active_floor[(k, which)] = True
    # ---- formal covariance --------------------------------------------------------------------------------
    try:
        N = Am.T @ Am
        cov = np.linalg.pinv(N, rcond=1e-12)
    except np.linalg.LinAlgError:
        cov = None
    for k, fl in enumerate(flights):
        ks = np.arange(fl.i, fl.j)
        fl.res = res[ks]
        fl.rms = float(np.sqrt(np.mean(np.minimum(res[ks], 4 * o['sig_px']) ** 2)))
        fl.n_shadow = int((chosen[ks] >= 0).sum())
        sr = sres[ks][chosen[ks] >= 0]
        fl.shadow_rms = float(np.sqrt(np.mean(sr ** 2))) if len(sr) else np.nan
        if cov is not None:
            scale = max(1.0, fl.rms / o['sig_px']) ** 2
            fl.cov = cov[NP * k:NP * k + NP, NP * k:NP * k + NP] * scale
    return dict(flights=flights, junctions=junctions, segs=segs, t=t, f0=f0, res=res, chosen=chosen, sres=sres)
