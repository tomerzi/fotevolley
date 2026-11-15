"""Glue: per-shot reconstruction (DP segmentation -> shadow matching -> joint flight solve)."""
import numpy as np, pandas as pd
from data import *
import segment as sg, chain as ch, sun as sunmod

DP = dict(lam_f=150.0, lam_g=100.0, c_other=45.0)


def shadow_lists(cam, frames, uv, cands_by_frame, sun_s, tol_px=4.0):
    """per observation: ground points of the dark-blob candidates that satisfy the camera/sun epipolar constraint"""
    out = []
    rays = cam.rays(uv)
    for k, f in enumerate(frames):
        c = cands_by_frame.get(int(f))
        if c is None or sun_s is None:
            out.append(None); continue
        suv = c[:, :2]
        far = np.hypot(suv[:, 0] - uv[k, 0], suv[:, 1] - uv[k, 1]) > 22
        if not far.any():
            out.append(None); continue
        suv = suv[far]
        S = cam.to_plane(suv, 0.0)[:, :2]
        d = np.repeat(rays[k][None], len(S), 0)
        P, lam, miss = sunmod.triangulate(cam, d, S, sun_s)
        # epipolar miss converted to pixels at the shadow location
        Sfit = sunmod.shadow_of(P, sun_s)
        pfit = cam.project(np.c_[Sfit, np.zeros(len(S))])
        mpx = np.hypot(*(pfit - suv).T)
        ok = (mpx < tol_px) & (lam > 3) & (P[:, 2] > 0.2) & (P[:, 2] < 14) & (P[:, 1] > -5.5) & (P[:, 1] < 13)
        out.append(S[ok] if ok.any() else None)
    return out


def run_shot(cam, ball_shot, players, cands_by_frame=None, sun_s=None, kd=ch.KD_DEFAULT, opts=None, dp=None, gscale=1.0, segs=None):
    g = ball_shot[ball_shot.obs]
    if len(g) < 5:
        return None
    fr = g.frame.values
    uv = g[['center_x', 'center_y']].values
    w = np.where(g.source == 'detected', 1.0, 0.6)
    if segs is None:
        segs, cost = sg.segment_shot(cam, fr, uv, w, **(dp or DP))
    shad = shadow_lists(cam, fr, uv, cands_by_frame, sun_s) if (cands_by_frame is not None and sun_s is not None) else None
    out = ch.solve_shot(cam, fr, uv, w, segs, players, kd=kd, sun_s=sun_s, shad=shad, opts=opts, gscale=gscale)
    out.update(frames=fr, uv=uv, w=w, shad=shad, sun_s=sun_s)
    return out


def describe(out):
    t, fr, segs = out['t'], out['frames'], out['segs']
    for s in segs:
        if s['kind'] == 'flight':
            fl = s['flight']; pa, pb = fl.pos(fl.t_start)[0], fl.pos(fl.t_end)[0]
            tt = np.linspace(fl.t_start, fl.t_end, 50); zz = fl.pos(tt)[:, 2]
            sz = fl.sigma([0.5 * (fl.t_start + fl.t_end)])[0, 2]
            print('  flight %5d-%5d n=%3d (%5.1f,%5.1f,%5.2f)->(%5.1f,%5.1f,%5.2f) zmax %5.2f v0 %4.1f rms %.1fpx shadows %2d (%.2fm) sig_z %.2f' % (
                s['fa'], s['fb'], s['j'] - s['i'], *pa, *pb, zz.max(), np.linalg.norm(fl.vel(fl.t_start)), fl.rms, fl.n_shadow, fl.shadow_rms, sz))
        elif s['kind'] == 'ground':
            pa, pb = sg.ground_pos(s['theta'], t[s['i']]), sg.ground_pos(s['theta'], t[s['j'] - 1])
            print('  ground %5d-%5d n=%3d (%5.1f,%5.1f)->(%5.1f,%5.1f)' % (s['fa'], s['fb'], s['j'] - s['i'], pa[0], pa[1], pb[0], pb[1]))
        else:
            print('  other  %5d-%5d n=%3d' % (s['fa'], s['fb'], s['j'] - s['i']))
    for jn in out['junctions']:
        c = jn['contact']
        if jn['kinds'] == ('flight', 'flight'):
            a = jn['a']['flight']
            tau = jn['tau'] if jn['tau'] is not None else jn['ta']
            p = a.pos(tau)[0]
            print('   junction%s f~%.1f at (%.1f,%.1f,%.2f) gap %.2f contact %s' % (' loose' if jn.get('loose') else '', out['f0'] + tau * FPS, *p, jn.get('gap', np.nan),
                  ('%s %s rel %.2f pdist %.2f' % (c['player'], JOINTS[c['joint']], c['rel'], jn.get('pdist', np.nan))) if c else None))


def cands_dict(cands):
    d = {}
    arr = cands[['frame', 'u', 'v', 'score']].values
    fr = arr[:, 0].astype(int)
    order = np.argsort(fr, kind='stable')
    arr, fr = arr[order], fr[order]
    bounds = np.nonzero(np.diff(fr))[0] + 1
    for chunk in np.split(arr, bounds):
        d[int(chunk[0, 0])] = chunk[:, 1:]
    return d
