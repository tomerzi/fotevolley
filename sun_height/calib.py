# fit ground homography + one radial distortion term to the court lines
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
S = '.'
L = json.load(open(f'{S}/lines60.json'))
L['left'] = [[213.5,440],[141.7,520],[65.4,600],[29.4,640]]
L['right'] = [[1832.5,460],[1895.5,520]]
C = np.array([960., 540.]); F = 1000.
def undist(p, k):
    d = (np.asarray(p, float) - C)/F; r2 = (d**2).sum(-1, keepdims=True)
    return C + d*F*(1 + k[0]*r2 + k[1]*r2**2)
def to_m(p, prm):
    H = np.append(prm[:8], 1).reshape(3,3); u = undist(p, prm[8:])
    v = np.c_[u, np.ones(len(u))] @ H.T; return v[:,:2]/v[:,2:]
cons = [('far',1,0.),('near',1,9.),('left',0,0.),('right',0,18.)]
anchors = [([1007.5,418.75],(9.,0.))]   # net crosses the far sideline
def resid(prm):
    r = [to_m(L[n], prm)[:,ax]-val for n,ax,val in cons]
    for p,(X,Y) in anchors: m = to_m([p], prm)[0]; r.append(np.array([m[0]-X, m[1]-Y])*5)
    return np.concatenate(r)
def fit(prm, free):
    for _ in range(50):
        r = resid(prm); J = np.zeros((len(r), len(prm)))
        for j in free:
            e = 1e-6*max(abs(prm[j]),1e-3); q = prm.copy(); q[j] += e; J[:,j] = (resid(q)-r)/e
        dx = np.linalg.lstsq(J[:,free], -r, rcond=None)[0]; prm[free] += dx
        if np.abs(dx).max() < 1e-12: break
    return prm
H0 = np.load('../homography_px_to_m.npy'); H0 = H0/H0[2,2]
p = np.r_[H0.ravel()[:8], 0., 0.]
p0 = fit(p.copy(), list(range(8)))
print('no distortion  rms %.3f m'%np.sqrt((resid(p0)**2).mean()))
p1 = fit(p0.copy(), list(range(9)))
print('k1             rms %.3f m  k1=%.4f'%(np.sqrt((resid(p1)**2).mean()), p1[8]))
p2 = fit(p1.copy(), list(range(10)))
print('k1,k2          rms %.3f m  k=%s'%(np.sqrt((resid(p2)**2).mean()), p2[8:].round(4)))
for n,ax,val in cons: e = to_m(L[n], p2)[:,ax]-val; print(n, 'max err %.3f'%np.abs(e).max())
np.save(f'{S}/calib.npy', p2)
H = np.append(p2[:8],1).reshape(3,3); Hi = np.linalg.inv(H)
for Y in (0, 4.5, 9, 10.5):
    v = Hi@[9,Y,1]; print('net ground (9,%.1f) -> undistorted px'%Y, (v[:2]/v[2]).round(1))
