import numpy as np
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
S = '.'
prm = np.load(f'{S}/calib.npy'); C = np.array([960., 540.]); F = 1000.
H = np.append(prm[:8], 1).reshape(3,3); Hi = np.linalg.inv(H)
def undist(p):
    d = (np.atleast_2d(np.asarray(p, float)) - C)/F; r2 = (d**2).sum(-1, keepdims=True)
    return C + d*F*(1 + prm[8]*r2 + prm[9]*r2**2)
def to_m(p):
    u = undist(p); v = np.c_[u, np.ones(len(u))] @ H.T; return v[:,:2]/v[:,2:]

# full pinhole camera from the ground homography (principal point at image centre, square pixels)
def camera():
    T = np.array([[1,0,C[0]],[0,1,C[1]],[0,0,1.]])
    A = np.linalg.inv(T) @ Hi
    a1, a2 = A[:,0], A[:,1]
    f2_orth = -(a1[0]*a2[0] + a1[1]*a2[1])/(a1[2]*a2[2])
    f2_norm = ((a1[0]**2+a1[1]**2) - (a2[0]**2+a2[1]**2))/(a2[2]**2 - a1[2]**2)
    return f2_orth, f2_norm, A, T
def projection(f):
    _, _, A, T = camera()
    K = np.diag([f, f, 1.]); M = np.linalg.inv(K) @ A
    s = 2/(np.linalg.norm(M[:,0]) + np.linalg.norm(M[:,1]))
    r1, r2, t = M[:,0]*s, M[:,1]*s, M[:,2]*s
    if t[2] < 0: r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    # make Z point up: a point above the ground must project above (smaller v) its base near the image centre
    P = T @ K @ np.c_[r1, r2, r3, t]
    base = P @ [9, 4.5, 0, 1]; up = P @ [9, 4.5, 1, 1]
    if up[1]/up[2] > base[1]/base[2]: P[:,2] *= -1; r3 = -r3
    R = np.c_[r1, r2, r3]; cam_pos = -R.T @ t
    return P, cam_pos
def project(P, X):
    v = P @ np.r_[X, 1.]; return v[:2]/v[2]
def distort(u):
    # invert undist() by fixed-point iteration
    u = np.asarray(u, float); p = u.copy()
    if np.abs(u - C).max() > 3000: return np.full(2, np.nan)
    for _ in range(40):
        e = u - undist(p)[0]; p = p + 0.7*e
        if np.abs(e).max() < 1e-3: break
    return p if np.abs(e).max() < 0.5 else np.full(2, np.nan)

NET_SLOPE = -0.014                       # net ground line X = 9 + NET_SLOPE*Y
NET_TOP = 2.2
POST_Y = 10.03                           # post base, from the post-shadow lines
def xnet(Y): return 9 + NET_SLOPE*Y
def sun_vec(top_off, theta_deg):
    """ground shadow displacement per metre of height, from net-top shadow offset and shadow-line angle"""
    th = np.radians(theta_deg); u = -np.array([np.cos(th), np.sin(th)])
    mag = top_off/(NET_TOP*(np.cos(th) - NET_SLOPE*np.sin(th)))
    return mag*u
VVP = np.load('./vvp.npy')
def p3_from(kappa): return kappa*np.r_[VVP, 1.]
def solve_kappa(base_xy, height, top_px):
    """scale of the vertical column so that base raised by `height` lands on top_px (undistorted)"""
    b = Hi @ np.r_[base_xy, 1.]; v = np.r_[VVP, 1.]; t = np.r_[undist(top_px)[0], 1.]
    # lam*t = b + height*kappa*v  -> least squares in (lam, kappa)
    A = np.c_[t, -height*v]; lam, kap = np.linalg.lstsq(A, b, rcond=None)[0]
    return kap, A @ [lam, kap] - b
def proj3(X, kappa):
    v = Hi @ np.r_[X[0], X[1], 1.] + X[2]*p3_from(kappa); return distort(v[:2]/v[2])
KAPPA = -0.012602121578728852
def height_of(ground_xy, px):
    """height above ground_xy of the point seen at pixel px (least squares along the vertical)"""
    b = Hi @ np.r_[ground_xy, 1.]; v = np.r_[VVP, 1.]*KAPPA; t = np.r_[undist(px)[0], 1.]
    lam, h = np.linalg.lstsq(np.c_[t, -v], b, rcond=None)[0]; return h

def ground_at_height(px, h):
    """court (X,Y) of the point seen at pixel px if it is at height h"""
    Hh = Hi.copy(); Hh[:,2] += h*KAPPA*np.r_[VVP, 1.]
    v = np.linalg.solve(Hh, np.r_[undist(px)[0], 1.]); return v[:2]/v[2]
def shadow_px(px, h, dv):
    g = ground_at_height(px, h); return proj3((g[0]+h*dv[0], g[1]+h*dv[1], 0), KAPPA)
