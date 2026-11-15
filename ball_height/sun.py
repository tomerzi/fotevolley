"""Sun direction per shot and ball <-> shadow matching ("camera + sun" stereo).

A ball at P = (x, y, z) throws its shadow on the sand at  S = (x, y) + z * s,  s = (a, b) = horizontal shadow
displacement per metre of height.  With the ball on the camera ray  P = C + lam * d  this gives, for a correct
(ball pixel, shadow pixel) pair, ONE equation that is linear in s (the "epipolar" constraint):

    cross(S - Cxy - Cz*s , dxy + dz*s) = 0

so the sun direction of a shot is found by Hough voting over all (frame, dark-blob candidate) pairs followed by a
robust linear refinement, without needing any 3-D reconstruction of the ball.
"""
import numpy as np


def cross2(p, q):
    return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]


def epi_coeffs(cam, d, S):
    """coefficients (ca, cb, c0) of  ca*a + cb*b = c0  for ray directions d (n,3) and shadow ground points S (n,2)"""
    Sp = S - cam.C[:2]
    ca = -d[:, 2] * Sp[:, 1] - cam.C[2] * d[:, 1]
    cb = d[:, 2] * Sp[:, 0] + cam.C[2] * d[:, 0]
    c0 = -(Sp[:, 0] * d[:, 1] - Sp[:, 1] * d[:, 0])
    return ca, cb, c0


def triangulate(cam, d, S, s):
    """least-squares ball position on the camera ray for shadow ground point S and sun slope s.
    returns P (n,3), lam (n,), miss (n,) = distance on the sand between S and the shadow of the solution"""
    A = d[:, :2] + d[:, 2:3] * s                      # (n,2)
    rhs = S - cam.C[:2] - cam.C[2] * s
    lam = (A * rhs).sum(1) / (A * A).sum(1)
    P = cam.C + lam[:, None] * d
    miss = np.linalg.norm(A * lam[:, None] - rhs, axis=1)
    return P, lam, miss


def shadow_of(P, s):
    P = np.asarray(P, float)
    return P[..., :2] + P[..., 2:3] * np.asarray(s)


def pairs_for_shot(cam, ball, cands):
    """all (ball observation, candidate) pairs of a shot. ball: DataFrame(frame, center_x, center_y);
    cands: DataFrame(frame,u,v,score,...). returns dict of arrays"""
    m = ball[['frame', 'center_x', 'center_y']].merge(cands, on='frame')
    if len(m) == 0:
        return None
    buv = m[['center_x', 'center_y']].values
    suv = m[['u', 'v']].values
    d = cam.rays(buv)
    S = cam.to_plane(suv, 0.0)[:, :2]
    # blobs that are (almost) the ball itself carry no information
    far = np.hypot(*(buv - suv).T) > 30
    return dict(frame=m.frame.values[far], d=d[far], S=S[far], buv=buv[far], suv=suv[far], score=m.score.values[far])


A_RANGE, B_RANGE, STEP = (-2.0, 2.0), (-0.5, 2.5), 0.01


def hough(cam, pr, prior=None, prior_sigma=0.25):
    """vote for the sun slope (a,b) on a grid; returns accumulator, grid axes"""
    ga = np.arange(A_RANGE[0], A_RANGE[1] + 1e-9, STEP)
    gb = np.arange(B_RANGE[0], B_RANGE[1] + 1e-9, STEP)
    acc = np.zeros((len(ga), len(gb)))
    ca, cb, c0 = epi_coeffs(cam, pr['d'], pr['S'])
    nrm = np.hypot(ca, cb)
    ok = nrm > 1e-6
    ca, cb, c0, fr = ca[ok] / nrm[ok], cb[ok] / nrm[ok], c0[ok] / nrm[ok], pr['frame'][ok]
    # one vote per (frame, cell): a frame cannot vote twice for the same sun direction
    for use_a in (True, False):
        sel = (np.abs(cb) >= np.abs(ca)) if use_a else (np.abs(cb) < np.abs(ca))
        if not sel.any():
            continue
        if use_a:
            bb = (c0[sel][:, None] - ca[sel][:, None] * ga[None, :]) / cb[sel][:, None]
            jb = np.round((bb - B_RANGE[0]) / STEP).astype(int)
            ia = np.broadcast_to(np.arange(len(ga))[None, :], jb.shape)
        else:
            aa = (c0[sel][:, None] - cb[sel][:, None] * gb[None, :]) / ca[sel][:, None]
            ia = np.round((aa - A_RANGE[0]) / STEP).astype(int)
            jb = np.broadcast_to(np.arange(len(gb))[None, :], ia.shape)
        ff = np.broadcast_to(fr[sel][:, None], ia.shape)
        okc = (ia >= 0) & (ia < len(ga)) & (jb >= 0) & (jb < len(gb))
        key = np.unique(np.stack([ff[okc], ia[okc], jb[okc]], 1), axis=0)
        np.add.at(acc, (key[:, 1], key[:, 2]), 1.0)
    # smooth a little (3x3 box) so that a peak split over neighbouring cells is found
    sm = np.zeros_like(acc)
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            sm += np.roll(np.roll(acc, da, 0), db, 1)
    if prior is not None:
        A, B = np.meshgrid(ga, gb, indexing='ij')
        sm = sm * np.exp(-0.5 * ((A - prior[0]) ** 2 + (B - prior[1]) ** 2) / prior_sigma ** 2)
    return sm, ga, gb


def refine(cam, pr, s0, tol=0.04, iters=6):
    """robust linear refinement of the sun slope using pairs whose epipolar line passes within `tol` of s0"""
    ca, cb, c0 = epi_coeffs(cam, pr['d'], pr['S'])
    nrm = np.hypot(ca, cb)
    ca, cb, c0 = ca / nrm, cb / nrm, c0 / nrm
    s = np.array(s0, float)
    inl = np.zeros(len(ca), bool)
    for _ in range(iters):
        r = ca * s[0] + cb * s[1] - c0
        inl = np.abs(r) < tol
        # physically meaningful pairs only: ball above the sand, in front of the camera
        P, lam, miss = triangulate(cam, pr['d'], pr['S'], s)
        inl &= (lam > 3) & (P[:, 2] > 0.25) & (P[:, 2] < 14)
        if inl.sum() < 4:
            break
        A = np.stack([ca[inl], cb[inl]], 1)
        s_new, *_ = np.linalg.lstsq(A, c0[inl], rcond=None)
        if np.linalg.norm(s_new - s) < 1e-5:
            s = s_new; break
        s = s_new
    # conditioning: spread of the line directions among the inliers
    if inl.sum() >= 4:
        A = np.stack([ca[inl], cb[inl]], 1)
        ev = np.linalg.eigvalsh(A.T @ A)
        cond = float(ev[0])
    else:
        cond = 0.0
    return s, inl, cond
