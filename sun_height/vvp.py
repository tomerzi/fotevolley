# vertical vanishing point from upright players (ankles -> neck axis), undistorted pixels
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from cam import undist
d = pd.read_csv('../pose/vitpose_body25.csv')
k = ['neck','mid_hip','r_knee','l_knee','r_ankle','l_ankle','r_hip','l_hip']
ok = np.all([d[f'{j}_s'] > 0.7 for j in k], axis=0); d = d[ok]
def P(j): return d[[f'{j}_x', f'{j}_y']].to_numpy()
ank = (P('r_ankle')+P('l_ankle'))/2; kn = (P('r_knee')+P('l_knee'))/2; hip = P('mid_hip'); neck = P('neck')
def ang(a, b):
    c = (a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1); return np.degrees(np.arccos(np.clip(c,-1,1)))
straight = (ang(kn-ank, hip-kn) < 6) & (ang(hip-ank, neck-hip) < 6) & (np.abs(ank[:,0]-P('r_ankle')[:,0]) < 40)
ank, neck = undist(ank[straight]), undist(neck[straight])
print('upright samples', len(ank))
# vvp lies far below; parametrise as homogeneous direction, minimise angular residual of each axis
def resid(v):
    vp = np.array([v[0], v[1]])
    a = neck - ank; b = vp - ank
    cr = (a[:,0]*b[:,1]-a[:,1]*b[:,0])/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1)
    return cr
v = np.array([1000., 6000.])
for it in range(100):
    r = resid(v); J = np.zeros((len(r),2))
    for j in range(2): e = np.zeros(2); e[j] = 1.; J[:,j] = (resid(v+e)-r)
    w = 1/(1+(r/np.median(np.abs(r))/3)**2)   # robust weights
    dv = np.linalg.lstsq(J*w[:,None], -r*w, rcond=None)[0]; v += dv
    if np.abs(dv).max() < 1e-3: break
r = resid(v)
print('vvp (undistorted px)', v.round(0), 'median axis error %.2f deg'%np.degrees(np.median(np.abs(np.arcsin(r)))))
# bootstrap spread
rng = np.random.default_rng(0); vs = []
A, N = ank.copy(), neck.copy()
for _ in range(30):
    i = rng.integers(0, len(A), len(A)); ank, neck = A[i], N[i]; vv = v.copy()
    for it in range(30):
        rr = resid(vv); J = np.zeros((len(rr),2))
        for j in range(2): e = np.zeros(2); e[j] = 1.; J[:,j] = (resid(vv+e)-rr)
        w = 1/(1+(rr/np.median(np.abs(rr))/3)**2); vv += np.linalg.lstsq(J*w[:,None], -rr*w, rcond=None)[0]
    vs.append(vv)
print('bootstrap sd', np.std(vs, axis=0).round(0))
np.save('vvp.npy', v)
