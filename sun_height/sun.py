import math, datetime as dt
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
LAT, LON = 47.0727, 15.4403  # Freiheitsplatz, Graz
def sun(t_utc):
    jd = t_utc.timestamp()/86400 + 2440587.5; T = (jd-2451545)/36525
    L0 = (280.46646+T*(36000.76983+0.0003032*T))%360
    M = 357.52911+T*(35999.05029-0.0001537*T); e = 0.016708634-T*(0.000042037+1.267e-7*T)
    Mr = math.radians(M)
    C = math.sin(Mr)*(1.914602-T*(0.004817+1.4e-5*T))+math.sin(2*Mr)*(0.019993-0.000101*T)+math.sin(3*Mr)*0.000289
    om = 125.04-1934.136*T; lam = L0+C-0.00569-0.00478*math.sin(math.radians(om))
    eps = 23+(26+(21.448-T*(46.815+T*(0.00059-T*0.001813)))/60)/60 + 0.00256*math.cos(math.radians(om))
    dec = math.asin(math.sin(math.radians(eps))*math.sin(math.radians(lam)))
    y = math.tan(math.radians(eps)/2)**2
    eqt = 4*math.degrees(y*math.sin(2*math.radians(L0))-2*e*math.sin(Mr)+4*e*y*math.sin(Mr)*math.cos(2*math.radians(L0))-0.5*y*y*math.sin(4*math.radians(L0))-1.25*e*e*math.sin(2*Mr))
    mins = t_utc.hour*60+t_utc.minute+t_utc.second/60
    ha = math.radians((mins+eqt+4*LON)/4-180); la = math.radians(LAT)
    el = math.asin(math.sin(la)*math.sin(dec)+math.cos(la)*math.cos(dec)*math.cos(ha))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha)*math.sin(la)-math.tan(dec)*math.cos(la)))+180
    return math.degrees(el), az%360
for day in (24,25,26):
    print(f'--- {day}.06.2022 (local CEST) ---')
    for h in range(9,20):
        for mi in (0,30):
            t = dt.datetime(2022,6,day,h-2,mi,tzinfo=dt.timezone.utc)
            el,az = sun(t)
            if day==26 or 40<=el<=47: print(f'{h:02d}:{mi:02d}  elev {el:5.1f}  az {az:5.1f}  shadow/height {1/math.tan(math.radians(el)):.2f}')
