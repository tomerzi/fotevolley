# Usage: python scripts/court/court_squares.py <out_video.mp4> <out_squares.csv>   (~0.12 s/frame on CPU)
# Whole match: each player's 3x3 square per frame (CSV) + video with the grid painted on the sand,
# players' squares tinted, name tags. Player shapes come from Sapiens2 keypoints, ball from the ball track.
import cv2,numpy as np,pandas as pd,subprocess,imageio_ffmpeg,sys,os,time
sys.path.insert(0,os.path.dirname(__file__));import posemask
ROOT="/teamspace/studios/this_studio/fotevolley";OUTV=sys.argv[1];OUTCSV=sys.argv[2]
B=pd.read_csv(ROOT+"/data/boxes/player_boxes_v3video.csv")
def sq(x,y):
    if not(0<=x<=18 and 0<=y<=9):return "out"
    s="L" if x<9 else "R";c=min(int((x-(0 if s=="L" else 9))//3),2);r=min(int(y//3),2);return f"{s}{r*3+c+1}"
B["square"]=[sq(x,y) for x,y in zip(B.court_x,B.court_y)]
B[["frame","time_sec","player","team","feet_x","feet_y","court_x","court_y","court_side","square","source"]].to_csv(OUTCSV,index=False)
print("squares written",len(B),flush=True)
N=int(B.frame.max())+1
BB={f:g for f,g in B.groupby("frame")}
BALL=pd.read_csv(ROOT+"/data/boxes/ball_boxes_v3video.csv").dropna(subset=["center_x"]).set_index("frame")
t=time.time();P=posemask.load(ROOT+"/data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv");print("pose loaded",round(time.time()-t),flush=True)
Hm2px=np.linalg.inv(np.load(ROOT+"/data/mappings/homography_px_to_m.npy"))
PPM=80;X0,Y0=-1.0,-1.0;CW,CH=int(20*PPM),int(11*PPM)
Hc=Hm2px@np.array([[1/PPM,0,X0],[0,1/PPM,Y0],[0,0,1]])
def c(x,y):return int((x-X0)*PPM),int((y-Y0)*PPM)
can=np.zeros((CH,CW,3),np.uint8);alpha=np.zeros((CH,CW),np.float32)
def line(p,q,w,a):
    m=np.zeros((CH,CW),np.uint8);cv2.line(m,c(*p),c(*q),255,w,cv2.LINE_AA);np.maximum(alpha,m/255*a,out=alpha);can[m>0]=(255,255,255)
for x0 in (0,9):
    for k in range(4):
        o=k in(0,3);line((x0+3*k,0),(x0+3*k,9),9 if o else 5,0.85 if o else 0.6);line((x0,3*k),(x0+9,3*k),9 if o else 5,0.85 if o else 0.6)
tm=np.zeros((CH,CW),np.uint8)
for s,x0 in (("L",0),("R",9)):
    n=1
    for r in range(3):
        for cc in range(3):
            t_=f"{s}{n}";n+=1;(tw,th),_=cv2.getTextSize(t_,cv2.FONT_HERSHEY_DUPLEX,2.4,6)
            cx,cy=c(x0+1.5+3*cc,1.5+3*r);cv2.putText(tm,t_,(cx-tw//2,cy+th//2),cv2.FONT_HERSHEY_DUPLEX,2.4,255,6,cv2.LINE_AA)
can[tm>0]=(255,255,255);np.maximum(alpha,tm/255*0.25,out=alpha)
W,H=1920,1080
gcol=cv2.warpPerspective(can,Hc,(W,H)).astype(np.float32);ga=cv2.warpPerspective(alpha,Hc,(W,H))
POST=np.zeros((H,W),np.uint8);cv2.rectangle(POST,(989,736),(1050,1080),255,-1);cv2.rectangle(POST,(1003,350),(1025,745),255,-1)
TCOL={"BRA":(40,200,255),"ISR":(235,120,30)}
CELLQ={}
for s,x0 in (("L",0),("R",9)):
    for n in range(9):
        q=np.array([c(x0+3*(n%3),3*(n//3)),c(x0+3*(n%3)+3,3*(n//3)),c(x0+3*(n%3)+3,3*(n//3)+3),c(x0+3*(n%3),3*(n//3)+3)],np.float32)
        CELLQ[f"{s}{n+1}"]=cv2.perspectiveTransform(q[None],Hc)[0].astype(np.int32)   # cell corners in frame px
cap=cv2.VideoCapture(ROOT+"/video/main_camera_clean_v3.mp4")
p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r","25","-i","-",
   "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-crf","23","-movflags","+faststart",OUTV],stdin=subprocess.PIPE)
t0=time.time()
for i in range(N):
    ok,im=cap.read()
    if not ok:break
    rows=BB.get(i,B.iloc[:0])
    m=np.zeros((H,W),np.uint8)
    if i in P:posemask.draw(m,P[i])
    for r in rows.itertuples():          # players without a pose row: ellipse in the box
        if i not in P or r.player not in set(P[i].player):cv2.ellipse(m,(int((r.x1+r.x2)/2),int((r.y1+r.y2)/2)),(int((r.x2-r.x1)/2),int((r.y2-r.y1)/2)),0,0,360,255,-1)
    if i in BALL.index:
        b=BALL.loc[i];b=b.iloc[0] if isinstance(b,pd.DataFrame) else b
        cv2.circle(m,(int(b.center_x),int(b.center_y)),int(max(b.x2-b.x1,b.y2-b.y1)*0.6)+2,255,-1)
    m=np.maximum(m,POST);m=cv2.GaussianBlur(m,(5,5),0).astype(np.float32)/255
    tint=np.zeros((H,W,3),np.uint8);ta=np.zeros((H,W),np.uint8)
    for r in rows.itertuples():
        if r.square!="out":cv2.fillPoly(tint,[CELLQ[r.square]],TCOL[r.team],cv2.LINE_AA);cv2.fillPoly(ta,[CELLQ[r.square]],77,cv2.LINE_AA)
    a=(ta.astype(np.float32)/255*(1-m))[...,None];img=im*(1-a)+tint*a
    a=(ga*(1-m))[...,None];out=(img*(1-a)+gcol*a).astype(np.uint8)
    for r in rows.itertuples():
        col=TCOL[r.team];fx,fy=int(r.feet_x),int(r.feet_y)
        cv2.circle(out,(fx,fy),7,(0,0,0),-1);cv2.circle(out,(fx,fy),5,col,-1)
        t_=f"{r.player} {r.square}";(tw,th),_=cv2.getTextSize(t_,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        tx,ty=int((r.x1+r.x2)/2-tw/2),max(int(r.y1)-12,th+10)
        cv2.rectangle(out,(tx-8,ty-th-8),(tx+tw+8,ty+8),col,-1);cv2.putText(out,t_,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2,cv2.LINE_AA)
    cv2.rectangle(out,(250,1022),(450,1068),(0,0,0),-1);cv2.putText(out,f"frame {i}",(262,1056),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    p.stdin.write(out.tobytes())
    if i%500==0:print(i,round(time.time()-t0),"s",flush=True)
p.stdin.close();p.wait();print("done",round(time.time()-t0),flush=True)
