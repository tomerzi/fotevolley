"""Pinhole camera with one radial distortion term + a small Levenberg-Marquardt solver (numpy only).

World frame (metres): X along the court length (0 = left baseline, 16 = right baseline),
Y across the court (0 = near sideline, 8 = far sideline), Z up (0 = sand).
"""
import json
import numpy as np

W, H = 1920, 1080
COURT_L, COURT_W, NET_X, NET_H = 16.0, 8.0, 8.0, 2.20


def rot(pan, tilt, roll):
    """World->camera rotation. pan about world Z, tilt = depression below horizontal, roll about optical axis.
    With pan=0 the camera looks along +Y (across the court from the near side)."""
    cp, sp = np.cos(pan), np.sin(pan)
    ct, st = np.cos(tilt), np.sin(tilt)
    cr, sr = np.cos(roll), np.sin(roll)
    fwd = np.array([sp * ct, cp * ct, -st])          # optical axis
    right = np.array([cp, -sp, 0.0])                 # image +x
    down = np.cross(fwd, right)                      # image +y
    R = np.stack([right, down, fwd])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
    return Rr @ R


class Camera:
    names = ['f', 'k1', 'pan', 'tilt', 'roll', 'Cx', 'Cy', 'Cz', 'cx', 'cy', 'k2']

    def __init__(self, f, k1, pan, tilt, roll, Cx, Cy, Cz, cx=W / 2, cy=H / 2, k2=0.0):
        self.f, self.k1, self.k2 = float(f), float(k1), float(k2)
        self.pan, self.tilt, self.roll = float(pan), float(tilt), float(roll)
        self.C = np.array([Cx, Cy, Cz], float)
        self.cx, self.cy = float(cx), float(cy)
        self.R = rot(self.pan, self.tilt, self.roll)

    def params(self):
        return np.array([self.f, self.k1, self.pan, self.tilt, self.roll, *self.C, self.cx, self.cy, self.k2])

    def to_json(self, path, extra=None):
        d = dict(zip(self.names, map(float, self.params())))
        d.update(extra or {})
        with open(path, 'w') as fh:
            json.dump(d, fh, indent=1)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(*[d[n] for n in cls.names])

    # --- projection -------------------------------------------------------------------------
    def project(self, P):
        """world points (...,3) -> distorted pixel coordinates (...,2)"""
        P = np.asarray(P, float)
        pc = (P - self.C) @ self.R.T
        xn, yn = pc[..., 0] / pc[..., 2], pc[..., 1] / pc[..., 2]
        r2 = xn * xn + yn * yn
        d = 1 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack([self.cx + self.f * xn * d, self.cy + self.f * yn * d], -1)

    def undistort_norm(self, uv):
        """distorted pixels (...,2) -> undistorted normalised image coordinates (...,2)"""
        uv = np.asarray(uv, float)
        xd, yd = (uv[..., 0] - self.cx) / self.f, (uv[..., 1] - self.cy) / self.f
        xn, yn = xd.copy(), yd.copy()
        for _ in range(12):
            r2 = xn * xn + yn * yn
            d = 1 + self.k1 * r2 + self.k2 * r2 * r2
            xn, yn = xd / d, yd / d
        return np.stack([xn, yn], -1)

    def rays(self, uv):
        """distorted pixels (...,2) -> unit ray directions in the world frame (...,3)"""
        n = self.undistort_norm(uv)
        dc = np.concatenate([n, np.ones(n.shape[:-1] + (1,))], -1)
        dw = dc @ self.R
        return dw / np.linalg.norm(dw, axis=-1, keepdims=True)

    def to_plane(self, uv, z=0.0):
        """intersect pixel rays with the horizontal plane Z=z -> world points (...,3)"""
        d = self.rays(uv)
        t = (z - self.C[2]) / d[..., 2]
        return self.C + d * t[..., None]

    def to_vertical_plane(self, uv, point, normal):
        """intersect pixel rays with the plane through `point` with (horizontal) `normal`"""
        d = self.rays(uv)
        point, normal = np.asarray(point, float), np.asarray(normal, float)
        t = ((point - self.C) * normal).sum(-1) / (d * normal).sum(-1)
        return self.C + d * np.asarray(t)[..., None]


def lm(fun, x0, iters=60, lam=1e-3, eps=None, verbose=False, bounds=None):
    """Levenberg-Marquardt with a forward-difference Jacobian. fun(x) -> residual vector."""
    x = np.array(x0, float)
    eps = np.full(len(x), 1e-6) if eps is None else np.asarray(eps, float)
    r = fun(x)
    c = r @ r
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            xp = x.copy(); xp[j] += eps[j]
            J[:, j] = (fun(xp) - r) / eps[j]
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(12):
            try:
                dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            xn = x + dx
            if bounds is not None:
                xn = np.clip(xn, bounds[0], bounds[1])
            rn = fun(xn)
            cn = rn @ rn
            if np.isfinite(cn) and cn < c:
                improved = True
                rel = (c - cn) / max(c, 1e-30)
                x, r, c = xn, rn, cn
                lam = max(lam / 3, 1e-9)
                break
            lam *= 4
        if verbose:
            print(it, c, lam)
        if not improved or rel < 1e-10:
            break
    return x, r, J
