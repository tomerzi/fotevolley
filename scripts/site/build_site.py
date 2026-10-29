import sys,json;sys.path.insert(0,'scripts/ball_height')
import numpy as np,pandas as pd,cv2,glob,os
from traj import Camera
# usage: python scripts/site/build_site.py OUT_DIR  (OUT_DIR holds video/part*.mp4; writes index.html, frames/, touches/)
OUT=sys.argv[1] if len(sys.argv)>1 else '/tmp/claude-0/-home-user-fotevolley/0499e459-defb-588b-97ce-792185ae37d4/scratchpad/site'
TPL=os.path.join(os.path.dirname(os.path.abspath(__file__)),'template.html')
if not os.path.exists(TPL): TPL=OUT+'/template.html'
for f in glob.glob(OUT+'/frames/*'): os.remove(f)
cam=Camera()
h=pd.read_csv('data/analysis/ball_height_v2.csv');b=pd.read_csv('data/boxes/ball_boxes_corrected3.csv')
ev=pd.read_csv('data/points/point_events_corrected.csv')
X=h[['court_x_m','court_y_m']].fillna(0).values
g=cam.project(np.c_[X,np.zeros(len(X))])
edge=(b.center_x.values<120)|(b.center_x.values>1800)|(g[:,0]<0)|(g[:,0]>1920)|(g[:,1]<0)|(g[:,1]>1080)|(X[:,0]<-4)|(X[:,0]>22)|(X[:,1]<-3)|(X[:,1]>13)
h['edge']=edge&(h.method!='missing')
hc=[None if np.isnan(v) else int(round(v*100)) for v in h.height_m]
meth=''.join('e' if e else {'flight_fit':'f','size_only':'s','missing':'-'}[m] for m,e in zip(h.method,h.edge))
# dead time: anything outside the 60 rallies (data/points/rallies_v2.csv) is shown but not counted
ral=pd.read_csv('data/points/rallies_v2.csv')
inplay=np.zeros(len(h),bool)
for r in ral.itertuples(): inplay[r.start_frame:r.end_frame+1]=True
dead=[];i0=None
for f_ in range(len(h)):
    if not inplay[f_] and i0 is None: i0=f_
    if inplay[f_] and i0 is not None: dead.append([i0,f_-1]);i0=None
if i0 is not None: dead.append([i0,len(h)-1])
good=h[(h.method=='flight_fit')&~h.edge&b.center_x.notna().values&inplay]
pts=[]
for r in ral.itertuples():
    seg=good[(good.frame>=r.start_frame)&(good.frame<=r.end_frame)]
    if len(seg):
        i=seg.height_m.idxmax();pts.append(dict(n=int(r.rally),start=int(r.start_frame),end=int(r.end_frame),winner=r.winner,score=f"{r.ISR_score_after}-{r.BRA_score_after}",max=float(seg.height_m.max()),maxf=int(h.frame[i]),mean=round(float(seg.height_m.mean()),2)))
summary=dict(frames=len(h),fps=25,inplay=int(inplay.sum()),fit=int((h.method=='flight_fit').sum()),size=int((h.method=='size_only').sum()),missing=int((h.method=='missing').sum()),edge=int(h.edge.sum()),
  median=float(good.height_m.median()),p95=float(good.height_m.quantile(.95)),max=float(good.height_m.max()),maxf=int(good.height_m.idxmax()),flights=int(h.flight_id.max()+1),reproj=float(np.nanmedian(h.reproj_err_px)))
if os.path.exists('data/analysis/ball_height_settings.json'): summary['k_drag']=json.load(open('data/analysis/ball_height_settings.json')).get('k_drag',0)
cands=[summary['maxf']]
for q in [.5,.25,.9]:
    v=good.height_m.quantile(q);cands.append(int((good.height_m-v).abs().idxmin()))
low=good[good.height_m<0.4];cands.append(int(low.index[len(low)//3]))
cands+=[p['maxf'] for p in sorted(pts,key=lambda p:-p['max'])[1:3]]
cands=list(dict.fromkeys(cands))[:7]
cap=cv2.VideoCapture('video/main_camera_clean_v2.mp4');shots=[]
for fr in sorted(cands):
    cap.set(1,fr);ok,im=cap.read();hh=h.height_m[fr]
    base=np.array([X[fr,0],X[fr,1],0.0])
    seg=cam.project(np.array([base+[0,0,-hh*t] for t in np.linspace(0,1,30)])).astype(int)
    for k in range(0,29,2): cv2.line(im,tuple(seg[k]),tuple(seg[k+1]),(255,255,255),3)
    cv2.ellipse(im,tuple(seg[0]),(18,7),0,0,360,(255,255,255),3)
    u=(int(b.center_x[fr]),int(b.center_y[fr]));cv2.circle(im,u,22,(0,200,255),3)
    lab=f"{hh:.2f} m"
    for c,w in [((0,0,0),7),((255,255,255),3)]: cv2.putText(im,lab,(u[0]+28,u[1]+10),cv2.FONT_HERSHEY_SIMPLEX,1.3,c,w)
    im=cv2.resize(im,(1280,720));fn=f'frames/f{fr}.jpg';cv2.imwrite(OUT+'/'+fn,im,[cv2.IMWRITE_JPEG_QUALITY,78])
    shots.append(dict(frame=int(fr),t=round(fr/25,2),h=round(float(hh),2),src=fn))
players=json.load(open('data/analysis/player_statures.json'))
U=8/9  # court units (lines mapped to 18 x 9) -> m on a 16 x 8 court
pos=pd.read_csv('data/analysis/player_positions_v3.csv',usecols=['v2_frame','frame','player','x','y','speed','airborne','posture'])
speed={};sstats={}
rs=pd.read_csv('data/analysis/player_speed_rallies_v3.csv')
for pl,t in pos.groupby('player'):
    t=t.drop_duplicates('v2_frame').set_index('v2_frame').reindex(range(len(h)))
    ok=(~t.airborne.fillna(True).astype(bool))&(~t.posture.isin(['hidden','none']))
    v=(t.speed*U*3.6).where(ok)
    speed[pl]=[None if np.isnan(x) else int(round(x*10)) for x in v]
    r=rs[rs.player==pl]
    sstats[pl]=dict(top=round(float(r.top_speed.max()*U*3.6),1),median=round(float(v.dropna().median()),1),
        dist=int(round(r.distance.sum()*U)),fast=round(float(r.sprint_sec.sum()),1))
# coverage heat maps in the team's view (own half, net at the top), 0.5 m cells
from scipy.ndimage import gaussian_filter
ral=pd.read_csv('data/points/rallies_v3video.csv');cov={}
cm=pd.read_csv('data/analysis/coverage_match_v3.csv').set_index('player')
allr=[]
for r in ral.itertuples():
    seg=pos[pos.frame.between(r.start_frame,r.end_frame)]
    for pl,g in seg.groupby('player'):
        left=(g.x<9).mean()>0.5
        allr.append(pd.DataFrame(dict(player=pl,xr=(9-g.x if left else g.x-9)*U,yr=(g.y if left else 9-g.y)*U)))
allr=pd.concat(allr)
for pl,g in allr.groupby('player'):
    H2,_,_=np.histogram2d(g.xr.clip(0,7.99),g.yr.clip(0,7.99),bins=16,range=[[0,8],[0,8]])
    H2=gaussian_filter(H2/H2.sum(),1.0)
    z=cm.loc[pl]
    cov[pl]=dict(grid=[[round(float(x)/H2.max(),3) for x in row] for row in H2],a50=round(float(z.area50*U*U),1),a90=round(float(z.area90*U*U),1),
        net=round(float(z.share_left_net+z.share_mid_net+z.share_right_net)*100),mid=round(float(z.share_left_mid+z.share_mid_mid+z.share_right_mid)*100),
        base=round(float(z.share_left_base+z.share_mid_base+z.share_right_base)*100),
        zones=[[round(float(z[f'share_{lane}_{col}'])*100,1) for lane in ('left','mid','right')] for col in ('net','mid','base')],
        out=round(float(z.share_out)*100,1),mean=[round(float(g.yr.mean()),2),round(float(g.xr.mean()),2)])
# chest touches: labelled ones (v3 frames) -> v2 frames, stance in m
os.makedirs(OUT+'/touches',exist_ok=True)
for f in glob.glob(OUT+'/touches/*'): os.remove(f)
fm=pd.read_csv('data/mappings/frame_mapping_v3.csv');fm=fm[~fm.deleted].set_index('v3_frame').v2_frame
tc=pd.read_csv('data/analysis/chest_stance_v3.csv');pb=pd.read_csv('data/boxes/player_boxes_v9.csv',usecols=['frame','player','x1','y1','x2','y2'])
touches=[]
for _,r in tc.iterrows():
    fr=int(fm[int(r.frame)]);cap.set(1,fr);ok,im=cap.read()
    bx=pb[(pb.frame==fr)&(pb.player==r.player)]
    if len(bx): x,y=int((bx.x1.iloc[0]+bx.x2.iloc[0])/2),int(bx.y1.iloc[0]+(bx.y2.iloc[0]-bx.y1.iloc[0])*0.35)
    else: x,y=int(b.center_x.fillna(960)[fr]),int(b.center_y.fillna(540)[fr])
    x0,y0=max(0,min(1920-320,x-160)),max(0,min(1080-320,y-140))
    fn=f'touches/t{fr}.jpg';cv2.imwrite(OUT+'/'+fn,cv2.resize(im[y0:y0+320,x0:x0+320],(240,240)),[cv2.IMWRITE_JPEG_QUALITY,80])
    air='raised' in (r.left_foot,r.right_foot) or bool(r.jump)
    touches.append(dict(frame=fr,player=r.player,dist=round(float(r.stance)*U,2),air=bool(air),reliable=bool(r.reliable),inplay=bool(r.in_play),
        h=None if np.isnan(h.height_m[fr]) else round(float(h.height_m[fr]),2),src=fn))
# partner distance of the conceding team at the moment the point was lost (scripts/analysis/conceded_distance.py)
cd=pd.read_csv('data/analysis/conceded_partner_distance.csv')
SRC={'hand-marked moment':'hand','own error (last touch)':'error','hand-marked landing':'landing','ball track landing':'track','rally end (fallback)':'end'}
conceded=[dict(n=int(r.rally),set=int(r.set),loser=r.loser,score=r.score,frame=int(r.point_frame),d=None if np.isnan(r.distance_m) else float(r.distance_m),
    src=SRC[r.source],unc=bool(r.uncertain)) for r in cd.itertuples()]
d=json.dumps(dict(conceded=conceded,h=hc,m=meth,points=pts,summary=summary,shots=shots,players=players,speed=speed,sstats=sstats,touches=touches,cov=cov,dead=dead),separators=(',',':'))
t=open(TPL).read().replace('__NSEG__',str(len(glob.glob(OUT+'/video/part*.mp4'))));open(OUT+'/index.html','w').write(t.replace('__DATA__',d))
print(summary);print([s['src'] for s in shots])
