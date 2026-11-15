"""Optimal (dynamic-programming) segmentation of the 2-D ball track of one shot into physically
interpretable pieces:

  flight : free ballistic flight in 3-D  (p0 + v0 t + g t^2/2), 6 parameters, linear in the unknowns
  ground : ball on the sand (z = ball radius), constant acceleration in the ground plane, 6 parameters
  other  : anything else (ball carried in the hands, pinned under a foot, bad detections) - fixed price per frame

For a calibrated camera every observation gives two equations that are *linear* in the unknown initial
position / velocity, so each candidate segment is a 6x6 solve built from prefix sums; the DP then picks the
segmentation with the lowest total (robustified) reprojection error + a price per segment.
"""
import numpy as np
from data import G, BALL_R, FPS

GVEC = np.array([0.0, 0.0, -G])
# plausible region for the ball (metres): the camera sits at y ~ -7, spectators/banners at y ~ 11
BOX = dict(x=(-10.0, 26.0), y=(-5.5, 13.0), zmax=14.0)


def rows(cam, uv):
    """the two linear constraint row-vectors a1,a2 (world frame) of each observation: a . (P - C) = 0"""
    n = cam.undistort_norm(uv)
    a1 = cam.R[0][None, :] - n[:, 0:1] * cam.R[2][None, :]
    a2 = cam.R[1][None, :] - n[:, 1:2] * cam.R[2][None, :]
    return a1, a2


def ref_weight(cam, uv):
    """approximate conversion of the algebraic residual to pixels: f / (camera depth of a typical ball position)"""
    d = cam.rays(uv)
    # typical position: on the ray, at the court mid-depth plane Y = 4, but not below the sand
    t_mid = (4.0 - cam.C[1]) / np.maximum(d[:, 1], 1e-3)
    t_gnd = np.where(d[:, 2] < -1e-6, (BALL_R - cam.C[2]) / np.minimum(d[:, 2], -1e-6), np.inf)
    t = np.minimum(t_mid, t_gnd)
    depth = t * (d @ cam.R[2])
    return cam.f / depth


class Prefix:
    """prefix sums of the weighted normal equations for a linear model  A theta = b"""

    def __init__(self, A, b, w, dim=3):
        # A: (n_obs, 2, k) ; b: (n_obs, 2) ; w: (n_obs,)
        Aw = A * w[:, None, None]
        bw = b * w[:, None]
        M = np.einsum('nri,nrj->nij', Aw, Aw)
        q = np.einsum('nri,nr->ni', Aw, bw)
        s = (bw ** 2).sum(1)
        z = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
        self.M, self.q, self.s = z(M), z(q), z(s)
        self.k = A.shape[2]
        self.dim = dim

    def solve(self, i, j, t0=0.0):
        """least squares on observations i..j-1 (i may be an array). returns theta, cost.
        The polynomial-in-time parameters are solved in a time frame centred at t0 (numerical conditioning: the prefix
        sums use the shot's time origin, which can be many seconds away from the segment)."""
        M = self.M[j] - self.M[i]
        q = self.q[j] - self.q[i]
        s = self.s[j] - self.s[i]
        S = self.shift(t0)
        Ml = S.T @ M @ S
        ql = q @ S
        Ml = Ml + 1e-10 * np.eye(self.k) * (np.trace(Ml, axis1=-2, axis2=-1)[..., None, None] / self.k)
        thl = np.linalg.solve(Ml, ql[..., None])[..., 0]
        cost = s - (ql * thl).sum(-1)
        return thl @ S.T, np.maximum(cost, 0.0)

    def shift(self, t0):
        """global parameters = S @ local parameters (local = same polynomial expressed around t0)"""
        nb = self.k // self.dim                  # number of polynomial blocks (2: p,v ; 3: p,v,a)
        I = np.eye(self.dim)
        S = np.zeros((self.k, self.k))
        c = [[1, -t0, 0.5 * t0 * t0], [0, 1, -t0], [0, 0, 1]]
        for r in range(nb):
            for cc in range(nb):
                S[r * self.dim:(r + 1) * self.dim, cc * self.dim:(cc + 1) * self.dim] = c[r][cc] * I
        return S


def build(cam, t, uv, w_obs):
    a1, a2 = rows(cam, uv)
    A3 = np.stack([a1, a2], 1)                                   # (n,2,3)
    wr = ref_weight(cam, uv) * w_obs
    # flight: theta = (p0, v0);  a.(p0 + v0 t + g t^2/2 - C) = 0
    Af = np.concatenate([A3, A3 * t[:, None, None]], 2)
    bf = A3 @ cam.C - (A3 @ GVEC) * (0.5 * t ** 2)[:, None]
    # ground: theta = (x0,y0,vx,vy,ax,ay); P = (x0 + vx t + ax t^2/2, y0 + ..., r)
    A2 = A3[:, :, :2]
    Ag = np.concatenate([A2, A2 * t[:, None, None], A2 * (0.5 * t ** 2)[:, None, None]], 2)
    bg = A3 @ cam.C - A3[:, :, 2] * BALL_R
    return Prefix(Af, bf, wr, 3), Prefix(Ag, bg, wr, 2)


def flight_pos(th, t):
    t = np.asarray(t, float)[..., None]
    return th[..., :3] + th[..., 3:] * t + 0.5 * GVEC * t * t


def ground_pos(th, t):
    t = np.asarray(t, float)[..., None]
    xy = th[..., 0:2] + th[..., 2:4] * t + 0.5 * th[..., 4:6] * t * t
    return np.concatenate([xy, np.full(xy.shape[:-1] + (1,), BALL_R)], -1)


def flight_ok(th, t0, t1, zmin=-0.35):
    ok = np.ones(th.shape[:-1], bool)
    for tt in (t0, t1):
        p = flight_pos(th, tt)
        ok &= (p[..., 0] > BOX['x'][0]) & (p[..., 0] < BOX['x'][1]) & (p[..., 1] > BOX['y'][0]) & \
              (p[..., 1] < BOX['y'][1]) & (p[..., 2] > zmin) & (p[..., 2] < BOX['zmax'])
    # apex below the ceiling
    tap = th[..., 5] / G
    zap = th[..., 2] + th[..., 5] * tap - 0.5 * G * tap ** 2
    ok &= (zap < BOX['zmax']) | (tap < t0) | (tap > t1)
    return ok


def ground_ok(th, t0, t1):
    ok = np.ones(th.shape[:-1], bool)
    for tt in (t0, t1):
        p = ground_pos(th, tt)
        ok &= (p[..., 0] > BOX['x'][0]) & (p[..., 0] < BOX['x'][1]) & (p[..., 1] > BOX['y'][0]) & (p[..., 1] < BOX['y'][1])
    a = th[..., 4:6]
    tm = 0.5 * (np.asarray(t0) + np.asarray(t1))
    v = th[..., 2:4] + a * tm[..., None]
    an = np.linalg.norm(a, axis=-1)
    vn = np.linalg.norm(v, axis=-1)
    # friction: acceleration small, or opposed to the motion and not larger than ~ g
    cosang = (a * v).sum(-1) / np.maximum(an * vn, 1e-9)
    ok &= (an < 1.5) | ((cosang < -0.8) & (an < 9.0))
    return ok


def segment_shot(cam, frames, uv, w_obs, lam_f=60.0, lam_g=40.0, c_other=25.0, min_f=5, min_g=4, max_gap=20,
                 huber=None):
    """frames: int array of observed frame numbers (sorted); uv: (n,2) ball centres; w_obs: per-observation weight.
    returns list of segments dict(kind, i, j (obs indices, j exclusive), theta, t0 (time origin frame))"""
    n = len(frames)
    f0 = frames[0]
    t = (frames - f0) / FPS
    Pf, Pg = build(cam, t, uv, w_obs)
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    back = [None] * (n + 1)
    gap_before = np.concatenate([[0], np.diff(frames)])      # gap (in frames) between obs k-1 and k
    for j in range(1, n + 1):
        # 'other' : single observation
        c = best[j - 1] + c_other
        if c < best[j]:
            best[j] = c; back[j] = ('other', j - 1, None)
        # candidate starts
        i = np.arange(0, j)
        # do not bridge very long gaps inside a segment
        if max_gap is not None:
            big = np.nonzero(gap_before[1:j] > max_gap)[0]
            if len(big):
                i = i[i >= big[-1] + 1]
        for kind, P, lam, mn, okf in (('flight', Pf, lam_f, min_f, flight_ok), ('ground', Pg, lam_g, min_g, ground_ok)):
            ii = i[(j - i) >= mn]
            if not len(ii):
                continue
            th, cost = P.solve(ii, j, t0=t[j - 1])
            ok = okf(th, t[ii], t[j - 1])
            tot = np.where(ok, best[ii] + cost + lam, np.inf)
            k = int(np.argmin(tot))
            if tot[k] < best[j]:
                best[j] = tot[k]; back[j] = (kind, int(ii[k]), th[k])
    segs = []
    j = n
    while j > 0:
        kind, i, th = back[j]
        segs.append(dict(kind=kind, i=i, j=j, theta=th))
        j = i
    segs.reverse()
    # merge runs of 'other'
    out = []
    for s in segs:
        if s['kind'] == 'other' and out and out[-1]['kind'] == 'other':
            out[-1]['j'] = s['j']
        else:
            out.append(s)
    for s in out:
        s['f0'] = int(f0)
        s['fa'], s['fb'] = int(frames[s['i']]), int(frames[s['j'] - 1])
    return out, best[n]
