# sun vector for any clip time, from the sun-path fit
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, datetime as dt, math
from cam import NET_SLOPE
exec(open('./sun.py').read().split('for day in')[0])
F = json.load(open('./sunfit.json'))
_p = np.array(F['pts']); _t, _m = _p[:,0], _p[:,1]
# monotone smoothing of clock vs clip time (running median then cumulative max), linear tail
_sm = np.array([np.median(_m[max(0,i-2):i+3]) for i in range(len(_m))]); _sm = np.maximum.accumulate(_sm)
def clock(t):
    if t <= _t[-1]: return float(np.interp(t, _t, _sm))
    return _sm[-1] + F['k']*(t - _t[-1])
def sunstate(t):
    m = clock(t); u = dt.datetime(2022,6,26,tzinfo=dt.timezone.utc) + dt.timedelta(minutes=m-120)
    el, az = sun(u); th = math.radians(az - F['beta'])
    dv = -1/math.tan(math.radians(el))*np.array([math.cos(th), math.sin(th)])
    return m, el, az, dv
