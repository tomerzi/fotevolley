# fit court compass bearing + clock time per frame to the astronomical sun path (Graz, 26 Jun 2022)
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, math, datetime as dt, numpy as np
from cam import NET_SLOPE
exec(open('sun.py').read().split('for day in')[0])          # reuse sun(t_utc)
def sunpos(minutes_cest):                                    # minutes after 00:00 CEST
    t = dt.datetime(2022,6,26,tzinfo=dt.timezone.utc) + dt.timedelta(minutes=minutes_cest-120)
    return sun(t)
grid = np.arange(14*60, 19*60, 0.25); SP = np.array([sunpos(m) for m in grid])   # (el, az)
cot = 1/np.tan(np.radians(SP[:,0]))
M = [r for r in json.load(open('meas.json')) if 'top_off' in r]
def model(beta):
    th = SP[:,1] - beta                                         # shadow-line angle in court frame
    off = 2.2*cot*(np.cos(np.radians(th)) - NET_SLOPE*np.sin(np.radians(th)))
    return th, off
best = None
for beta in np.arange(150, 230, 0.02):
    th, off = model(beta); err = 0; n = 0
    for r in M:
        if r.get('res', 1) < 0.015:
            e = ((off - r['top_off'])/0.005)**2 + ((th - r['theta'])/0.6)**2
            err += e.min(); n += 1
    if best is None or err < best[0]: best = (err, beta, n)
err, beta, n = best
print('court X-axis bearing %.2f deg (from N), chi2/n %.2f over %d frames'%(beta, err/n, n))
th, off = model(beta)
out = []
for r in M:
    i = np.argmin(((off - r['top_off'])/0.005)**2 + (((th - r['theta'])/0.6)**2 if r.get('res',1) < 0.015 else 0))
    out.append((r['t'], grid[i], SP[i,0], SP[i,1], th[i], r.get('theta')))
o = np.array([[a,b,c,d,e] for a,b,c,d,e,_ in out])
A = np.c_[o[:,0], np.ones(len(o))]; k, c0 = np.linalg.lstsq(A, o[:,1], rcond=None)[0]
print('clock = %.1f min + %.3f * clip_seconds/60*60  -> start %02d:%02d CEST, %.2f real minutes per clip minute'%(c0, k, c0//60, c0%60, k*60))
for t, m, el, az, thm, thmeas in out[::6]:
    print('clip %4ds  clock %02d:%04.1f  elev %.2f  az %.1f  theta model %.1f meas %s'%(t, m//60, m%60, el, az, thm, '%.1f'%thmeas if thmeas else '-'))
json.dump(dict(beta=beta, k=k, c0=c0, pts=[list(map(float, x[:5])) for x in out]), open('sunfit.json','w'))
