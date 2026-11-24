"""Build clips + items.json for the labelling game. Reads the repo checkout (REPO), writes only into OUT.
usage: python make_game.py REPO OUT"""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
REPO, OUT = sys.argv[1], sys.argv[2]
os.makedirs(f'{OUT}/clips', exist_ok=True)
VID = f'{REPO}/video/main_camera_clean_v3.mp4'
FPS, W, H, NF = 25, 1920, 1080, 13495
HE = {'Franklin': 'פרנקלין', 'Hiltinho': 'הילטיניו', 'Ron_Ben_Ishay': 'רון', 'Maor_Haas': 'מאור'}

t = pd.read_csv(f'{REPO}/data/contacts/touches_master.csv')
dec = pd.read_csv(f'{REPO}/data/contacts/owner_decisions.csv')
for d in dec[dec.kind == 'touch'].itertuples():
    sel = t.touch == int(d.touch)
    if d.field == 'in_play': t.loc[sel, 'in_play'] = d.value == 'True'
    elif d.field == 'action': t.loc[sel, 'action'] = d.value
t = t[(t.kind == 'player') & t.in_play.astype(bool)].sort_values(['rally', 'frame']).reset_index(drop=True)
ans = pd.read_csv(f'{REPO}/data/experiments/doubts/answers.csv')
confirmed = set(ans[ans.kind == 'actions'].touch.dropna().astype(int))
tb = pd.read_csv(f'{REPO}/data/analysis/touch_body.csv').set_index('touch')
box = pd.read_csv(f'{REPO}/data/boxes/player_boxes_v3video.csv').drop_duplicates(['frame', 'player']).set_index(['frame', 'player'])
e = pd.read_csv(f'{REPO}/data/points/point_end_verified.csv').set_index('point')
ral = pd.read_csv(f'{REPO}/data/points/rallies_v3video.csv').set_index('rally')

def box_at(f, p):
    for d in (0, -1, 1, -2, 2, -3, 3):
        if (f + d, p) in box.index: return box.loc[(f + d, p)]
    return None

jobs, touches = [], []
for r in t.itertuples():
    f = int(r.frame); b = box_at(f, r.player)
    cx, cy = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2) if b is not None else (W / 2, H / 2)
    X = int(min(max(cx - 480, 0), W - 960)); Y = int(min(max(cy - 300, 0), H - 540))
    f0 = max(0, f - 15); n = min(31, NF - f0); k = f - f0
    s = 640 / 960
    if b is not None:
        bx, by, bw, bh = int((b.x1 - X) * s) - 4, int((b.y1 - Y) * s) - 4, int((b.x2 - b.x1) * s) + 8, int((b.y2 - b.y1) * s) + 8
        draw = f",drawbox=x={bx}:y={by}:w={bw}:h={bh}:color=yellow@0.9:t=3:enable='between(n,{k-2},{k+2})'"
    else:
        draw = ''
    name = f'clips/t{int(r.touch)}.mp4'
    jobs.append(['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{f0 / FPS:.3f}', '-i', VID, '-frames:v', str(n),
                 '-vf', f'crop=960:540:{X}:{Y},scale=640:360{draw}', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                 '-crf', '30', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', f'{OUT}/{name}'])
    st = tb.stance_label.get(int(r.touch)) if int(r.touch) in tb.index else None
    touches.append(dict(id=f't{int(r.touch)}', touch=int(r.touch), rally=int(r.rally), set=int(r.set), frame=f,
                        player=r.player, he=HE[r.player], team=r.team, part=r.body_part if isinstance(r.body_part, str) else '',
                        side=r.limb_side if isinstance(r.limb_side, str) else '', action=r.action,
                        confirmed=int(r.touch) in confirmed or int(r.touch) in (244, 247, 249, 306),
                        stance=st if isinstance(st, str) else '', clip=name, k=k))

# special clips (full frame, 960x540)
def seg(name, a, b):
    a = max(0, a); b = min(NF - 1, b)
    jobs.append(['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{a / FPS:.3f}', '-i', VID, '-frames:v', str(b - a + 1),
                 '-vf', f"scale=960:540,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{{eif\\:n+{a}\\:d}}':x=12:y=12:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6",
                 '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', f'{OUT}/clips/{name}.mp4'])
    return dict(clip=f'clips/{name}.mp4', start=a, end=b)
special = {}
for p in (19, 34, 50):
    R = ral.loc[p]; special[f'p{p}'] = seg(f'p{p}', int(R.end_frame) - 125, int(R.end_frame) + 25)
special['p27'] = seg('p27', 5000, 5170)
special['p56'] = seg('p56', 12660, 12800)
R = ral.loc[39]; special['r39'] = seg('r39', int(R.start_frame), int(R.end_frame) + 10)

with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(lambda c: subprocess.run(c, capture_output=True, text=True), jobs))
bad = [(j[-1], r.stderr[-300:]) for j, r in zip(jobs, res) if r.returncode]
print('clips', len(jobs), 'failed', len(bad), bad[:3])

r39 = t[t.rally == 39][['touch', 'frame', 'player', 'team', 'body_part', 'action']]
pts = []
for p, E in e.iterrows():
    pts.append(dict(point=int(p), set=int(E.set), winner=E.winner, end=E.end_type))
json.dump(dict(touches=touches, special=special, points=pts,
               r39=[dict(touch=int(x.touch), frame=int(x.frame), he=HE[x.player], part=x.body_part if isinstance(x.body_part, str) else '?', action=x.action) for x in r39.itertuples()]),
          open(f'{OUT}/items.json', 'w'), ensure_ascii=False)
print('touches', len(touches), 'confirmed actions', sum(x['confirmed'] for x in touches), 'stance labelled', sum(bool(x['stance']) for x in touches))
