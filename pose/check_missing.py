"""Find frames where a player skeleton or the ball is missing although it is probably visible.
-> pose/suspects.csv  (frame, kind, who, detail)   and   pose/suspect_ranges.csv (merged runs)
kinds: pose_weak    player box fully inside the frame but < MIN_JOINTS joints with score >= S_MIN
       pose_edge    same, but the box touches the frame border (player partly off-screen: expected)
       player_none  fewer than 4 players in the frame (from players/player_boxes.csv)
       ball_hidden  ball missing in a short in-shot gap whose detected ends are well inside the frame
usage: python pose/check_missing.py"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, N = 1920, 1080, 15586
S_MIN, MIN_JOINTS = 0.3, 12
EDGE = 80            # px: a ball last seen this close to the border probably left the frame
MAX_GAP = 75         # longer in-shot gaps are dead time / ball gone, not a missed detection

pose = pd.read_csv(os.path.join(ROOT, 'pose', 'vitpose_body25.csv'))
done = pose.frame.max() + 1
s = pose.filter(regex='_s$').drop(columns='box_source', errors='ignore')
pose['n_ok'] = (s.values >= S_MIN).sum(1)
boxes = pd.read_csv(os.path.join(ROOT, 'players', 'player_boxes.csv'))
pose = pose.merge(boxes[['frame', 'player', 'x1', 'y1', 'x2', 'y2']], on=['frame', 'player'], how='left')
pose['edge'] = (pose.x1 <= 5) | (pose.y1 <= 5) | (pose.x2 >= W - 5) | (pose.y2 >= H - 5)
weak = pose[pose.n_ok < MIN_JOINTS]
rows = [(r.frame, 'pose_edge' if r.edge else 'pose_weak', r.player, f'{r.n_ok}/25 joints') for r in weak.itertuples()]

cnt = boxes.groupby('frame').player.apply(set).reindex(range(done), fill_value=set())
allp = {'BRA_A', 'BRA_B', 'ISR_A', 'ISR_B'}
rows += [(f, 'player_none', '+'.join(sorted(allp - ps)), f'{len(ps)} players') for f, ps in cnt.items() if len(ps) < 4]

ball = pd.read_csv(os.path.join(ROOT, 'ball', 'ball_boxes.csv')).iloc[:done]
miss = (ball.source == 'missing').values
x, y, shot = ball.center_x.values, ball.center_y.values, ball.shot.values
inside = lambda i: EDGE < x[i] < W - EDGE and EDGE < y[i] < H - EDGE


def leaves_frame(anchor, gap, step):
    """extrapolate the ball's flight from the detected frames on one side of the gap (step=-1: frames before)."""
    src = [g for g in range(anchor, anchor + 6 * step, step) if 0 <= g < done and not miss[g] and shot[g] == shot[anchor]][:5]
    if len(src) < 3:
        return False
    t = np.array(src, float); t0 = t.mean(); deg = 2 if len(src) >= 4 else 1
    gap = np.array(list(gap), float)
    gx = np.polyval(np.polyfit(t - t0, x[src], deg), gap - t0); gy = np.polyval(np.polyfit(t - t0, y[src], deg), gap - t0)
    return bool(((gx < 0) | (gx >= W) | (gy < 0) | (gy >= H)).any())


i = 0
while i < done:
    if not miss[i]:
        i += 1; continue
    j = i
    while j < done and miss[j]:
        j += 1
    a, b = i - 1, j          # anchors
    if a >= 0 and b < done and shot[a] == shot[b] and j - i <= MAX_GAP and inside(a) and inside(b) \
            and not leaves_frame(a, range(i, j), -1) and not leaves_frame(b, range(i, j), 1):
        rows += [(f, 'ball_hidden', 'ball', f'gap {i}-{j - 1}') for f in range(i, j)]
    i = j

sus = pd.DataFrame(rows, columns=['frame', 'kind', 'who', 'detail']).sort_values(['frame', 'kind'])
sus.to_csv(os.path.join(ROOT, 'pose', 'suspects.csv'), index=False)

rng = []
for (kind, who), g in sus.groupby(['kind', 'who']):
    fr = np.sort(g.frame.unique())
    for run in np.split(fr, np.where(np.diff(fr) > 1)[0] + 1):
        rng.append((kind, who, run[0], run[-1], len(run)))
rng = pd.DataFrame(rng, columns=['kind', 'who', 'start', 'end', 'n']).sort_values('start')
rng.to_csv(os.path.join(ROOT, 'pose', 'suspect_ranges.csv'), index=False)
print(f'checked frames 0-{done - 1}')
print(rng.groupby('kind').agg(runs=('n', 'size'), frames=('n', 'sum')))
print(sus[sus.kind == 'pose_weak'].who.value_counts().to_dict())
