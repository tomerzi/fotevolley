import pandas as pd, numpy as np, subprocess, json, os, random
from PIL import Image, ImageDraw
OUT='/tmp/claude-0/-home-user-fotevolley/6e6e98b8-1b1e-552b-b644-e3a0c5274410/scratchpad/labels'
os.chdir('/home/user/fotevolley')
W,H=1920,1080
box=pd.read_csv('data/boxes/player_boxes_v3video.csv').drop_duplicates(['frame','player']).set_index(['frame','player'])
body=pd.read_csv('data/analysis/touch_body.csv')
pt=pd.read_csv('data/analysis/point_touches.csv'); e=pd.read_csv('data/points/point_end_verified.csv').set_index('point')
t=pd.read_csv('data/contacts/touches_master.csv')
pt['owner']=pt.rally.map(e.last_touch); dis=pt[pt.last_player!=pt.owner]
# --- feet sample: 40 in-play touches, stratified by body part, spread over players
b=body[body.in_play & body.player.notna()]
rng=np.random.RandomState(7); feet=[]
for part,n in (('chest',16),('foot',12),('head',12)):
    g=b[b.body_part==part]
    per=[g[g.player==p] for p in sorted(g.player.unique())]
    k=n//len(per); pick=pd.concat([x.sample(min(k,len(x)),random_state=7) for x in per])
    if len(pick)<n: pick=pd.concat([pick,g.drop(pick.index).sample(n-len(pick),random_state=7)])
    feet.append(pick)
feet=pd.concat(feet).sample(frac=1,random_state=3)
# --- ISR identity: frames with both ISR boxes large, apart, yolo
bb=pd.read_csv('data/boxes/player_boxes_v3video.csv').drop_duplicates(['frame','player'])
isr=bb[bb.team=='ISR'].copy(); isr['h']=isr.y2-isr.y1
pv=isr.pivot_table(index='frame',columns='player',values=['h','x1','x2'])
ok=pv[(pv['h'].min(axis=1)>230)]
ok=ok[(abs((ok['x1']['ISR_A'] if 'ISR_A' in ok['x1'] else ok['x1']['Maor_Haas'])-(ok['x1'].iloc[:,1]))>250)]
idf=list(ok.index[::max(1,len(ok)//6)][:6])
print('id frames',idf,len(ok))
# --- disputed points: frames around our last touch
dpts=[]
for r in dis.itertuples():
    tt=t[(t.rally==r.rally)&(t.kind=='player')&t.in_play].sort_values('frame').iloc[-1]
    dpts.append(dict(point=int(r.rally),frame=int(tt.frame),team=tt.team,end=r.end_type))
need=set()
for x in feet.itertuples(): need|={x.frame+d for d in (-2,-1,0,1,2)}
need|=set(idf)
for d in dpts: need|={d['frame']+k for k in (-8,-4,0,4)}
need=sorted(need)
tmp=OUT+'/_frames'; os.makedirs(tmp,exist_ok=True)
sel='+'.join(f'eq(n\\,{f})' for f in need)
open(tmp+'/sel.txt','w').write('select='+sel)
subprocess.run(['ffmpeg','-loglevel','error','-y','-i','video/main_camera_clean_v3.mp4','-filter_script:v',tmp+'/sel.txt','-vsync','0','-q:v','3',tmp+'/%05d.jpg'],check=True)
files=sorted(os.listdir(tmp)); files=[f for f in files if f.endswith('.jpg')]
fmap={f:os.path.join(tmp,files[i]) for i,f in enumerate(need)}
assert len(files)==len(need),(len(files),len(need))
OLD={'Franklin':'BRA_A','Hiltinho':'BRA_B','Maor_Haas':'ISR_A','Ron_Ben_Ishay':'ISR_B'}
def crop(f,x1,y1,x2,y2,hgt):
    im=Image.open(fmap[f]); x1,y1,x2,y2=max(0,x1),max(0,y1),min(W,x2),min(H,y2)
    c=im.crop((int(x1),int(y1),int(x2),int(y2))); return c.resize((max(1,int(c.width*hgt/c.height)),hgt))
def row(ims,gap=6,bg=(20,20,20)):
    w=sum(i.width for i in ims)+gap*(len(ims)-1); h=max(i.height for i in ims); o=Image.new('RGB',(w,h),bg); x=0
    for i in ims: o.paste(i,(x,0)); x+=i.width+gap
    return o
meta={'feet':[],'ident':[],'points':[]}
for i,x in enumerate(feet.itertuples()):
    r=box.loc[(x.frame,x.player)] if (x.frame,x.player) in box.index else None
    if r is None: continue
    bh=r.y2-r.y1; pad=0.25*bh
    X1,Y1,X2,Y2=r.x1-pad,r.y1-pad,r.x2+pad,r.y2+0.15*bh
    ims=[]
    for d in (-2,-1,0,1,2):
        c=crop(x.frame+d,X1,Y1,X2,Y2,300)
        if d==0: ImageDraw.Draw(c).rectangle([0,0,c.width-1,c.height-1],outline=(255,200,0),width=5)
        ims.append(c)
    # feet zoom on touch frame
    fz=crop(x.frame,r.x1-0.15*bh,r.y2-0.35*bh,r.x2+0.15*bh,r.y2+0.08*bh,300)
    name=f'feet_{i:02d}.jpg'; row(ims).save(f'{OUT}/img/{name}',quality=80)
    zname=f'feetzoom_{i:02d}.jpg'; fz.save(f'{OUT}/img/{zname}',quality=82)
    meta['feet'].append(dict(id=f'f{int(x.touch)}',touch=int(x.touch),frame=int(x.frame),player=x.player,body_part=x.body_part,action=x.action,img=name,zoom=zname,
        auto=dict(left=x.left_foot,right=x.right_foot,stance=x.stance,facing=x.facing)))
random.seed(5)
for j,f in enumerate(idf):
    a=box.loc[(f,'Maor_Haas')]; b_=box.loc[(f,'Ron_Ben_Ishay')]
    order=['Maor_Haas','Ron_Ben_Ishay']; random.shuffle(order)
    ims=[]
    for k,p in enumerate(order):
        r=box.loc[(f,p)]; bh=r.y2-r.y1; c=crop(f,r.x1-0.2*bh,r.y1-0.1*bh,r.x2+0.2*bh,r.y2+0.05*bh,360)
        d=ImageDraw.Draw(c); d.rectangle([0,0,46,40],fill=(0,0,0)); d.text((14,8),'AB'[k],fill=(255,255,0),font_size=26) if hasattr(d,'text') else None
        ims.append(c)
    name=f'ident_{j}.jpg'; row(ims,gap=14).save(f'{OUT}/img/{name}',quality=82)
    meta['ident'].append(dict(id=f'i{f}',frame=int(f),img=name,_answer={'A':order[0],'B':order[1]}))
for d in dpts:
    f=d['frame']
    # region: union of all 4 player boxes near the toucher's team side, padded
    rs=[box.loc[(f,p)] for p in ('Maor_Haas','Ron_Ben_Ishay','Franklin','Hiltinho') if (f,p) in box.index]
    team_rs=[box.loc[(f,p)] for p in (('Maor_Haas','Ron_Ben_Ishay') if d['team']=='ISR' else ('Franklin','Hiltinho')) if (f,p) in box.index]
    X1=min(r.x1 for r in team_rs)-150; X2=max(r.x2 for r in team_rs)+150; Y1=min(r.y1 for r in team_rs)-200; Y2=max(r.y2 for r in team_rs)+40
    ims=[crop(f+k,X1,Y1,X2,Y2,330) for k in (-8,-4,0,4)]
    ImageDraw.Draw(ims[2]).rectangle([0,0,ims[2].width-1,ims[2].height-1],outline=(255,200,0),width=5)
    name=f'point_{d["point"]:02d}.jpg'; row(ims).save(f'{OUT}/img/{name}',quality=80)
    meta['points'].append(dict(id=f'p{d["point"]}',point=d['point'],frame=f,end=d['end'],img=name,team=d['team'],
        touch_table=pt.set_index('rally').last_player[d['point']],owner_review=pt.set_index('rally').owner[d['point']]))
json.dump(meta,open(OUT+'/meta_full.json','w'),ensure_ascii=False,indent=1)
print(len(meta['feet']),len(meta['ident']),len(meta['points']))
