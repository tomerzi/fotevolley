import pandas as pd
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cam import *
d = pd.read_csv('../pose/vitpose_body25.csv')
bx = pd.read_csv('../correct_bbox/player_boxes_v9.csv')[['frame','player','x1','y1','x2','y2']]
d = d.merge(bx, on=['frame','player'])
kk=['neck','mid_hip','r_knee','l_knee','r_ankle','l_ankle','r_heel','l_heel','r_big_toe','l_big_toe','nose','r_wrist','l_wrist']
d = d[np.all([d[f'{j}_s']>0.6 for j in kk],axis=0)]
def P(j): return d[[f'{j}_x',f'{j}_y']].to_numpy()
ank=(P('r_ankle')+P('l_ankle'))/2; kn=(P('r_knee')+P('l_knee'))/2; hip=P('mid_hip'); neck=P('neck')
def ang(a,b): c=(a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1); return np.degrees(np.arccos(np.clip(c,-1,1)))
up=(ang(kn-ank,hip-kn)<6)&(ang(hip-ank,neck-hip)<6)&(P('r_wrist')[:,1]>neck[:,1]+20)&(P('l_wrist')[:,1]>neck[:,1]+20)
d=d[up].copy(); print('upright frames',len(d))
foot=(P('r_heel')+P('l_heel')+P('r_big_toe')+P('l_big_toe'))/4
G=to_m(foot)
d['H_top']=[height_of(g,(x,y)) for g,x,y in zip(G,d.nose_x,d.y1)]
d['H_nose']=[height_of(g,(x,y)) for g,x,y in zip(G,d.nose_x,d.nose_y)]
d['X'],d['Y']=G[:,0],G[:,1]
print(d.groupby('player')[['H_top','H_nose']].agg(['median','std','count']).round(3))
d['Yb']=pd.cut(d.Y,[-1,2,4,6,8,10]); print(d.groupby('Yb',observed=True).H_top.median().round(3))
d['Xb']=pd.cut(d.X,[-1,4,8,12,16,19]); print(d.groupby('Xb',observed=True).H_top.median().round(3))
d.to_csv('upright.csv',index=False)
