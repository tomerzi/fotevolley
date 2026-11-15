import numpy as np, sys
sys.path.insert(0,'ball_height')
from data import *
import segment as sg, chain as ch
def fit_single(cam, fr, uv, kd=ch.KD_DEFAULT, n_iter=6, w=None):
    """drag-aware least-squares fit of ONE flight to observations (frames fr, pixels uv). returns flight, residual vectors"""
    t=(fr-fr[0])/FPS
    a1,a2=sg.rows(cam,uv)
    w0=np.ones(len(fr)) if w is None else w
    dw=sg.ref_weight(cam,uv)
    seg=dict(i=0,j=len(fr),theta=np.zeros(6))
    fl=ch.Flight.__new__(ch.Flight); fl.seg=seg; fl.i=0; fl.j=len(fr); fl.tref=0.0; fl.kd=kd; fl.p0=np.zeros(3); fl.v0=np.zeros(3); fl._grid=None; fl.gvec=ch.GVEC; fl.ae=np.zeros(3); fl.rich=False; fl.cov=None
    fl.t_start,fl.t_end=t[0],t[-1]
    first=True
    for it in range(n_iter):
        dl=np.zeros((len(fr),3)) if first else fl.delta(t)
        rowsA=[];rowsb=[]
        for ar in (a1,a2):
            W=(dw*w0)[:,None]
            rowsA.append(np.concatenate([ar*W,ar*W*t[:,None]],1))
            rowsb.append((ar@cam.C-(ar@ch.GVEC)*0.5*t**2-(ar*dl).sum(1))*W[:,0])
        th=np.linalg.lstsq(np.concatenate(rowsA),np.concatenate(rowsb),rcond=None)[0]
        fl.p0,fl.v0=th[:3],th[3:]; fl._grid=None; first=False
        P=fl.pos(t); dw=cam.f/np.maximum((P-cam.C)@cam.R[2],1.0)
    r=cam.project(fl.pos(t))-uv
    return fl,r
