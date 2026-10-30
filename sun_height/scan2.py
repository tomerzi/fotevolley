# ball-shadow scan v2: per-shot median background, transient-darkness blob score, sunlit sand only, player-shadow mask
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import subprocess, numpy as np, pandas as pd, time
from cam import *; from sunclock import sunstate
HS = np.arange(0, 15.001, 0.05)
W, Hg = 1920, 1080
b = pd.read_csv('../correct_bbox/ball_boxes_corrected3.csv')
nfr = int(b.frame.max())+1
shot_all = b.set_index('frame').shot.reindex(range(nfr)).ffill().bfill().astype(int).to_numpy()
b = b[b.center_x.notna()].set_index('frame')
pb = pd.read_csv('../correct_bbox/player_boxes_v9.csv', usecols=['frame','x1','y1','x2','y2','feet_x','feet_y'])
pbg = {f: g.to_numpy()[:,1:] for f, g in pb.groupby('frame')}
v3 = np.r_[VVP, 1.]*KAPPA
def undist_v(p):
    d = (p - C)/F; r2 = (d**2).sum(-1, keepdims=True); return C + d*F*(1 + prm[8]*r2 + prm[9]*r2**2)
def distort_v(u):
    p = u.copy()
    for _ in range(30): p = p + 0.7*(u - undist_v(p))
    p[np.abs(u - undist_v(p)).max(1) > 0.5] = np.nan; return p
def curve(px, dv):
    u = np.r_[undist_v(np.array([px], float))[0], 1.]
    Hh = np.repeat(Hi[None], len(HS), 0); Hh[:,:,2] += HS[:,None]*v3[None]
    g = np.linalg.solve(Hh, np.broadcast_to(u, (len(HS),3))[...,None])[...,0]; g = g[:,:2]/g[:,2:]
    s = g + HS[:,None]*dv[None]
    w = np.c_[s, np.ones(len(s))] @ Hi.T; w = w[:,:2]/w[:,2:]; w[np.abs(w - C).max(1) > 3000] = np.nan
    return distort_v(w), s
def frames():
    pr = subprocess.Popen(['ffmpeg','-v','error','-i','../main_camera_clean_v2.mp4','-f','rawvideo','-pix_fmt','gray','-'], stdout=subprocess.PIPE, bufsize=W*Hg*4)
    i = 0
    while True:
        buf = pr.stdout.read(W*Hg)
        if len(buf) < W*Hg: return
        yield i, buf; i += 1
t0 = time.time()
# pass 1: per-shot median background from up to 31 evenly spaced frames
want = {}
for s in np.unique(shot_all):
    fr = np.where(shot_all == s)[0]; want.update({int(f): s for f in fr[np.linspace(0, len(fr)-1, min(31, len(fr))).astype(int)]})
bg, buf_s = {}, {}
for i, buf in frames():
    if i in want: buf_s.setdefault(want[i], []).append(np.frombuffer(buf, np.uint8).reshape(Hg, W).copy())
    s = shot_all[min(i, nfr-1)]
    for k in [k for k in buf_s if k < s]: bg[k] = np.median(np.stack(buf_s.pop(k)), 0).astype(np.float32)
for k in list(buf_s): bg[k] = np.median(np.stack(buf_s.pop(k)), 0).astype(np.float32)
print('backgrounds', len(bg), '%.0fs'%(time.time()-t0), flush=True)
np.savez_compressed('bg_sample.npz', **{f'b{k}': bg[k][::4, ::4] for k in list(bg)[::10]})
RI, RO = 5, 12
def boxes(ii, x, y, rr): return ii[y+rr+1, x+rr+1] - ii[y-rr, x+rr+1] - ii[y+rr+1, x-rr] + ii[y-rr, x-rr]
out_f, out_s, out_c, out_g = [], [], [], []
for i, buf in frames():
    if i not in b.index: continue
    img = np.frombuffer(buf, np.uint8).reshape(Hg, W).astype(np.float32); B = bg[shot_all[i]]
    diff = B - img                                    # positive where something new is darker
    ii = np.zeros((Hg+1, W+1)); ii[1:,1:] = diff.cumsum(0).cumsum(1)
    ib = np.zeros((Hg+1, W+1)); ib[1:,1:] = B.cumsum(0).cumsum(1)
    r = b.loc[i]; _, _, _, dv = sunstate(i/25)
    pts, S = curve((r.center_x, r.center_y), dv)
    sc = np.full(len(HS), np.nan)
    ok = np.isfinite(pts).all(1) & (pts[:,0] > RO+1) & (pts[:,0] < W-RO-2) & (pts[:,1] > 365) & (pts[:,1] < Hg-RO-2)
    ok &= (S[:,0] > -4) & (S[:,0] < 22) & (S[:,1] > -2.5) & (S[:,1] < 11)
    # ball box and player bodies
    ok &= ~((pts[:,0] > r.x1-8) & (pts[:,0] < r.x2+8) & (pts[:,1] > r.y1-8) & (pts[:,1] < r.y2+8))
    for x1, y1, x2, y2, fx, fy in pbg.get(i, []):
        ok &= ~((pts[:,0] > x1-4) & (pts[:,0] < x2+4) & (pts[:,1] > y1-4) & (pts[:,1] < y2+4))
        fg = to_m([(fx, fy)])[0]; seg = 2.1*dv                         # player shadow: feet -> 2.1 m of height
        tt = np.clip(((S - fg) @ seg)/(seg @ seg), 0, 1); dist = np.linalg.norm(S - fg - tt[:,None]*seg[None], axis=1)
        ok &= dist > 0.45
    x = np.round(pts[ok,0]).astype(int); y = np.round(pts[ok,1]).astype(int)
    si, so = boxes(ii, x, y, RI), boxes(ii, x, y, RO)
    inner = si/(2*RI+1)**2; ring = (so-si)/((2*RO+1)**2-(2*RI+1)**2)
    lit = boxes(ib, x, y, RO)/(2*RO+1)**2 > 105            # background sand must be sunlit
    v = inner - np.abs(ring); v[~lit] = np.nan; sc[ok] = v
    out_f.append(i); out_s.append(sc); out_c.append(pts); out_g.append(S)
    if len(out_f) % 2000 == 0: print(len(out_f), '%.0fs'%(time.time()-t0), flush=True)
np.savez_compressed('scan2.npz', frames=np.array(out_f), scores=np.array(out_s), curves=np.array(out_c), ground=np.array(out_g).astype(np.float32), hs=HS)
print('done', len(out_f), '%.0fs'%(time.time()-t0), flush=True)
