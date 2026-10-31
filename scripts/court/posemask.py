# Player shapes from Sapiens2 keypoints (no neural net at render time): thick limbs, torso polygon, head disc.
import numpy as np,cv2,pandas as pd
KP=["nose","left_shoulder","right_shoulder","left_elbow","right_elbow","right_wrist","left_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle","left_big_toe","right_big_toe","left_heel","right_heel","neck"]
LIMBS=[("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
       ("left_hip","left_knee"),("left_knee","left_ankle"),("right_hip","right_knee"),("right_knee","right_ankle"),
       ("left_ankle","left_big_toe"),("left_heel","left_big_toe"),("right_ankle","right_big_toe"),("right_heel","right_big_toe"),
       ("neck","nose"),("left_shoulder","right_shoulder"),("left_hip","right_hip")]
def load(path,frames=None):
    cols=["frame","player","x1","y1","x2","y2"]+[f"{k}_{a}" for k in KP for a in "xys"]
    it=pd.read_csv(path,usecols=cols,chunksize=200000)
    P=pd.concat([c[c.frame.isin(frames)] if frames is not None else c for c in it])
    return {f:g for f,g in P.groupby("frame")}
def draw(mask,g,thr=0.3):
    for r in g.itertuples(index=False):
        d=r._asdict();h=max(d["y2"]-d["y1"],30.0)
        pt=lambda k:(int(d[k+"_x"]),int(d[k+"_y"])) if d[k+"_s"]>thr else None
        w=max(int(h*0.13),6)
        for a,b in LIMBS:
            p,q=pt(a),pt(b)
            if p and q:cv2.line(mask,p,q,255,w,cv2.LINE_AA)
        T=[pt(k) for k in ("left_shoulder","right_shoulder","right_hip","left_hip")]
        if all(T):cv2.fillPoly(mask,[np.array(T,np.int32)],255)
        for k,rr in (("nose",0.085),("neck",0.06)):
            p=pt(k)
            if p:cv2.circle(mask,p,max(int(h*rr),4),255,-1,cv2.LINE_AA)
        n=sum(1 for k in KP if d[k+"_s"]>thr)
        if n<6:   # pose failed: fall back to an ellipse inside the box
            cv2.ellipse(mask,(((d["x1"]+d["x2"])/2,(d["y1"]+d["y2"])/2),(d["x2"]-d["x1"],d["y2"]-d["y1"]),0),255,-1)
    return mask
