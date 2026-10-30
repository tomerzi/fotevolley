# ball height from scan2: Viterbi with smoothness and a capped 'shadow hidden' penalty, then ballistic g check
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
Z = np.load('scan2.npz'); FR, SC, HS, GR = Z['frames'], Z['scores'], Z['hs'], Z['ground']
b = pd.read_csv('../correct_bbox/ball_boxes_corrected3.csv').set_index('frame')
shot = b.loc[FR, 'shot'].to_numpy()
MIN_C = 12.
E = np.clip(np.nan_to_num(SC, nan=-10.), -10, 50) - MIN_C
MAXV = 0.9/0.05; LAM = 0.03
seg = np.r_[0, np.cumsum((np.diff(FR) != 1) | (np.diff(shot) != 0))]
idx = np.arange(len(HS)); D = (idx[:,None]-idx[None,:]).astype(float)
T = -LAM*D**2; T[np.abs(D) > MAXV] = -1e9
H = np.full(len(FR), np.nan); Q = np.full(len(FR), np.nan); P = np.zeros(len(FR), int)
for s in np.unique(seg):
    ii = np.where(seg == s)[0]
    V = np.maximum(E[ii[0]], -2); BP = []
    for k in ii[1:]:
        M = V[None,:] + T; bp = M.argmax(1); V = M[idx, bp] + np.maximum(E[k], -2); BP.append(bp)
    p = [int(V.argmax())]
    for bp in BP[::-1]: p.append(int(bp[p[-1]]))
    p = p[::-1]; H[ii] = HS[p]; Q[ii] = SC[ii, p]; P[ii] = p
found = np.nan_to_num(Q, nan=-1) >= MIN_C
SG = GR[np.arange(len(FR)), P]
out = pd.DataFrame(dict(frame=FR, time_sec=FR/25, shot=shot, segment=seg,
    height_m=np.where(found, H, np.nan), shadow_contrast=np.round(Q, 1), shadow_found=found,
    shadow_court_x=np.where(found, SG[:,0], np.nan).round(3), shadow_court_y=np.where(found, SG[:,1], np.nan).round(3)))
out.to_csv('ball_height_raw.csv', index=False)
print('frames', len(out), 'shadow found', found.sum())
print(out.height_m.describe().round(2))
acc = []
for s, g in out.groupby('segment'):
    h = g.height_m.to_numpy()
    for i in range(0, len(h)-15, 3):
        hh = h[i:i+15]
        if np.isnan(hh).any() or hh.min() < 0.8 or hh.max()-hh.min() < 0.4: continue
        t = np.arange(15)/25; c = np.polyfit(t, hh, 2); r = hh - np.polyval(c, t)
        if np.abs(r).max() < 0.1: acc.append(-2*c[0])
acc = np.array(acc); print(len(acc), 'free-flight windows; g median %.2f IQR %.2f-%.2f'%(np.median(acc), *np.percentile(acc, [25, 75])))
