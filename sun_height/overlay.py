import sys, pandas as pd
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw
from cam import *
t = float(sys.argv[1]); top_off, theta = float(sys.argv[2]), float(sys.argv[3]); out = sys.argv[4]
fr = int(round(t*25))
im = Image.open(f'q/f{int(t)}.png').convert('RGB'); D = ImageDraw.Draw(im)
dv = sun_vec(top_off, theta)
def pl(pts, col, w=2): D.line([tuple(p) for p in pts], fill=col, width=w)
Ys = np.linspace(-1.0, 9.7, 60)
pl([proj3((xnet(Y), Y, 2.2), KAPPA) for Y in Ys], (255,0,0))            # net top (model)
pl([proj3((xnet(Y), Y, 1.25), KAPPA) for Y in Ys], (255,128,0))         # net bottom (model)
pl([proj3((xnet(Y)+2.2*dv[0], Y+2.2*dv[1], 0), KAPPA) for Y in Ys], (0,255,255), 1)  # net-top shadow (model)
base = (xnet(POST_Y), POST_Y)
pl([proj3((*base, h), KAPPA) for h in np.linspace(0, 2.79, 20)], (255,0,255))       # post (model)
pl([proj3((base[0]+h*dv[0], base[1]+h*dv[1], 0), KAPPA) for h in np.linspace(0, 2.79, 20)], (255,0,255), 1)
# players: foot point + head-top shadow predicted with image-derived height
p = pd.read_csv('../pose/vitpose_body25.csv'); p = p[p.frame == fr]
bx = pd.read_csv('../correct_bbox/player_boxes_v9.csv'); bx = bx[bx.frame == fr]
for _, r in p.merge(bx[['player','y1']], on='player').iterrows():
    foot = np.array([(r.r_heel_x+r.l_heel_x+r.r_big_toe_x+r.l_big_toe_x)/4, (r.r_heel_y+r.l_heel_y+r.r_big_toe_y+r.l_big_toe_y)/4])
    g = to_m([foot])[0]; H = 1.70
    s = proj3((g[0]+H*dv[0], g[1]+H*dv[1], 0), KAPPA); h = proj3((g[0], g[1], H), KAPPA)
    D.line([tuple(foot), tuple(s)], fill=(0,255,0), width=1); D.ellipse([s[0]-5,s[1]-5,s[0]+5,s[1]+5], outline=(0,255,0), width=2)
    D.ellipse([h[0]-5,h[1]-5,h[0]+5,h[1]+5], outline=(255,255,0), width=2)
im.save(out)
