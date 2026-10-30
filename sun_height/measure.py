# per 5-s crop: net-top shadow line offset from the net, and post-shadow (pad) direction, in court metres
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob, sys, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from cam import to_m
S = '.'
OX, OY = 760, 620
NET_SLOPE = -0.014                      # net ground line: X = 9 + NET_SLOPE*Y (from equal drift of both net-shadow edges)
def xnet(Y): return 9 + NET_SLOPE*Y
def sm(p, k=3): return np.convolve(p, np.ones(k)/k, 'same')
rows = []
for f in sorted(glob.glob(f'{S}/fr/c*.png')):
    k = int(f[-8:-4]); t = (k-1)*5
    g = np.asarray(Image.open(f).convert('L')).astype(float)
    # net-top shadow: leftmost thin dark line in x 870..925, rows 640..780 (predicted x ~ 903 - 0.11*(y-620))
    top = []
    for y in range(640, 781, 10):
        xp = 903 - 0.11*(y-620); a, b = int(xp-14), int(xp+14)
        p = sm(g[y-OY, a-OX:b-OX]); ref = np.median(g[y-OY, a-OX-20:b-OX+20])
        i = int(np.argmin(p))
        if ref - p[i] > 25 and 2 <= i <= len(p)-3:
            c = p[i-1]-2*p[i]+p[i+1]; top.append((a+i+(0.5*(p[i-1]-p[i+1])/c if c else 0), y))
    # pad shadow: left edge = strongest falling step, right edge = strongest rising step, rows 875..985
    L, R = [], []
    for y in range(875, 986, 5):
        p = sm(g[y-OY, 820-OX:1000-OX], 5); gr = np.gradient(p)
        xc = 868 + 0.47*(y-875)           # rough left-edge prediction
        lo, hi = int(xc-12-820), int(xc+12-820)
        i = lo + int(np.argmin(gr[lo:hi]))
        if gr[i] < -6: L.append((820+i, y))
        lo2, hi2 = i+15, i+60
        if hi2 < len(gr):
            j = lo2 + int(np.argmax(gr[lo2:hi2]))
            if gr[j] > 6: R.append((820+j, y))
    rec = dict(t=t, ntop=len(top), nL=len(L), nR=len(R))
    if len(top) >= 8:
        T = to_m(top); off = xnet(T[:,1]) - T[:,0]; rec['top_off'] = np.median(off); rec['top_sd'] = off.std()
    if len(L) >= 15 and len(R) >= 15:
        def fitdir(P):
            M = to_m(P); A = np.c_[M[:,1], np.ones(len(M))]; (a, b), res, *_ = np.linalg.lstsq(A, M[:,0], rcond=None)
            return a, b, np.sqrt(res[0]/len(M)) if len(res) else 0
        aL, bL, rL = fitdir(L); aR, bR, rR = fitdir(R)
        a, b = (aL+aR)/2, (bL+bR)/2      # centre line X = a*Y + b
        rec['theta'] = np.degrees(np.arctan2(1, a))  # angle of shadow line from +X axis (degrees)
        rec['thL'], rec['thR'] = np.degrees(np.arctan2(1, aL)), np.degrees(np.arctan2(1, aR))
        Yp = (b - 9)/(NET_SLOPE - a); rec['Yp'] = Yp; rec['res'] = max(rL, rR)
    rows.append(rec)
import json; json.dump(rows, open(f'{S}/meas.json', 'w'))
for r in rows:
    print(r['t'], ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items() if k != 't'))
