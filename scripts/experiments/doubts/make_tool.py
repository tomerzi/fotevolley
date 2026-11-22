"""Images and item list for the doubts tool (owner gives the final answer). Run from the repo root:
python scripts/experiments/doubts/make_tool.py OUT_DIR   -> OUT_DIR/img/*.jpg, OUT_DIR/items.json
Sections: last touch at 5 points (images from the labelling page), 10 touches whose action is flagged (rallies 2, 39,
50), 30 touches where the calibrated stance rule is borderline (speed at touch 0.8-1.8 m/s, not yet labelled), 20
frames the facing test calls 'camera' (its most frequent mistake on the labelled sample)."""
import os, sys, json, subprocess
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
OUT = sys.argv[1]; os.makedirs(OUT + '/img', exist_ok=True)
W, H = 1920, 1080
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
box = pd.read_csv('data/boxes/player_boxes_v3video.csv').drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
t = pd.read_csv('data/contacts/touches_master.csv')
body = pd.read_csv('data/analysis/touch_body.csv')
lab = pd.read_csv('data/experiments/feet_isr_labels/answers_final.csv')
done = set(lab[lab.kind == 'feet'].id.str[1:].astype(int))
# touches with a flagged action
act = t[t.action_check == True].sort_values('frame')
# stance borderline
b = body[body.in_play & ~body.touch.isin(done) & (body.stance_cal != 'jump') & body.speed_at_touch.between(0.8, 1.8)]
st = b.sample(min(30, len(b)), random_state=11).sort_values('frame')
# facing: frames called 'camera'
fc = body[body.in_play & ~body.touch.isin(done) & (body.facing == 'camera') & ~body.touch.isin(st.touch)].sample(20, random_state=5).sort_values('frame')
need = set()
for x in act.itertuples(): need |= {x.frame + d for d in (-6, -3, 0, 3, 6)}
for x in st.itertuples(): need |= {x.frame + d for d in (-4, -2, 0, 2, 4)}
for x in fc.itertuples(): need |= {x.frame}
need = sorted(need)
tmp = OUT + '/_frames'; os.makedirs(tmp, exist_ok=True)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
open(tmp + '/s.txt', 'w').write('select=' + '+'.join(f'eq(n\\,{f})' for f in need))
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', 'video/main_camera_clean_v3.mp4', '-filter_script:v', tmp + '/s.txt',
                '-vsync', '0', '-q:v', '3', tmp + '/%05d.jpg'], check=True)
fs = sorted(f for f in os.listdir(tmp) if f.endswith('.jpg')); assert len(fs) == len(need)
fm = {f: os.path.join(tmp, fs[i]) for i, f in enumerate(need)}

def crop(f, x1, y1, x2, y2, h):
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    c = Image.open(fm[f]).convert('RGB').crop((int(x1), int(y1), int(x2), int(y2)))
    return c.resize((max(1, int(c.width * h / c.height)), h))
def row(ims):
    w = sum(i.width for i in ims) + 6 * (len(ims) - 1); o = Image.new('RGB', (w, max(i.height for i in ims)), (20, 20, 20)); x = 0
    for i in ims: o.paste(i, (x, 0)); x += i.width + 6
    return o
def mark(c, k, hl=False):
    d = ImageDraw.Draw(c); d.rectangle([0, 0, 40, 30], fill=(0, 0, 0)); d.text((8, 3), str(k), fill=(255, 230, 0), font=FONT)
    if hl: d.rectangle([0, 0, c.width - 1, c.height - 1], outline=(255, 200, 0), width=5)
    return c
items = {'actions': [], 'stance': [], 'facing': []}
for x in act.itertuples():
    r = box.loc[(x.frame, x.player)]; cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
    X1 = min(max(cx - 500, 0), W - 1000); Y1 = min(max(cy - 330, 0), H - 560)
    ims = []
    for k, d in enumerate((-6, -3, 0, 3, 6)):
        c = crop(x.frame + d, X1, Y1, X1 + 1000, Y1 + 560, 250)
        if d == 0 and (x.frame, x.player) in box.index:
            rr = box.loc[(x.frame, x.player)]; s = 250 / 560
            ImageDraw.Draw(c).rectangle([(rr.x1 - X1) * s - 4, (rr.y1 - Y1) * s - 4, (rr.x2 - X1) * s + 4, (rr.y2 - Y1) * s + 4], outline=(255, 220, 0), width=4)
        ims.append(mark(c, k + 1, d == 0))
    n = f'act_{int(x.touch)}.jpg'; row(ims).save(f'{OUT}/img/{n}', quality=80)
    items['actions'].append(dict(id=f'a{int(x.touch)}', touch=int(x.touch), frame=int(x.frame), rally=int(x.rally), player=x.player,
                                 part=x.body_part, img=n, auto=dict(action=x.action, sub=x.action_subtype, nth=x.touch_in_possession)))
for x in st.itertuples():
    r = box.loc[(x.frame, x.player)]; bh = r.y2 - r.y1; pad = 0.3 * bh
    ims = [mark(crop(x.frame + d, r.x1 - pad, r.y1 - pad, r.x2 + pad, r.y2 + 0.15 * bh, 280), k + 1, d == 0) for k, d in enumerate((-4, -2, 0, 2, 4))]
    n = f'st_{int(x.touch)}.jpg'; row(ims).save(f'{OUT}/img/{n}', quality=80)
    items['stance'].append(dict(id=f's{int(x.touch)}', touch=int(x.touch), frame=int(x.frame), player=x.player, part=x.body_part, img=n,
                                auto=dict(stance_cal=x.stance_cal, speed=round(x.speed_at_touch, 2))))
for x in fc.itertuples():
    r = box.loc[(x.frame, x.player)]; bh = r.y2 - r.y1
    c = crop(x.frame, r.x1 - 0.35 * bh, r.y1 - 0.15 * bh, r.x2 + 0.35 * bh, r.y2 + 0.05 * bh, 320)
    n = f'fc_{int(x.touch)}.jpg'; c.save(f'{OUT}/img/{n}', quality=82)
    items['facing'].append(dict(id=f'c{int(x.touch)}', touch=int(x.touch), frame=int(x.frame), player=x.player, img=n, auto=dict(facing=x.facing)))
json.dump(items, open(OUT + '/items.json', 'w'), ensure_ascii=False, indent=1)
print({k: len(v) for k, v in items.items()})
