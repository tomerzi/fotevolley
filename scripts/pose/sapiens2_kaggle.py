"""Sapiens2 308-keypoint pose on video/main_camera_clean_v3.mp4 (dead time removed, 13495 frames), using the
reviewed player boxes in v3 frame numbers (data/boxes/player_boxes_v3video.csv) widened by MARGIN on every side.
Kaggle: Accelerator = GPU T4 x2 (uses both GPUs; one GPU also works), Internet = On.
  !python fotevolley/scripts/pose/sapiens2_kaggle.py                          whole video
  !S2_START=0 S2_END=6000 python fotevolley/scripts/pose/sapiens2_kaggle.py   frames 0-5999 only (Kaggle stops a session after 12 h)
Settings by environment variable: S2_MODEL (0.4b | 0.8b | 1b), S2_FLIP (1 | 0), S2_MARGIN, S2_START, S2_END, S2_VIDEO (1 | 0).
-> /kaggle/working/sapiens2_pose_v3_<first>-<last>.csv   one row per (frame, player): v3 frame, v2 frame, box, box with margin,
                                                         308 x (x, y, score); rows are added every 250 frames
-> /kaggle/working/sapiens2_pose_v3_<first>-<last>.mp4   video with the boxes and skeletons drawn, for checking
A rerun in the same folder skips frames already in a sapiens2_pose_v3_*.csv, so a stopped run continues where it ended."""
import glob, os, subprocess, sys, time, urllib.request

MODEL = os.environ.get('S2_MODEL', '1b')      # 0.4b | 0.8b | 1b   (5b does not fit a 16 GB GPU)
MARGIN = float(os.environ.get('S2_MARGIN', 0.15))   # fraction of box width/height added on each side (0.15 -> 1.3x)
FLIP_TEST = os.environ.get('S2_FLIP', '1') == '1'   # average with the left-right flipped image (Sapiens2 default, 2x slower)
START = int(os.environ.get('S2_START', 0))
END = int(os.environ.get('S2_END', 0))              # 0 = to the end of the video
VIDEO_OUT = os.environ.get('S2_VIDEO', '1') == '1'
KPT_THR = 0.3        # only for drawing
WORK = os.environ.get('S2_WORK', '/kaggle/working')
LIMIT = int(os.environ.get('S2_LIMIT', 0))    # >0: only this many frames (quick test)
BRANCH = 'repo-reorganized'

print('started', flush=True)


def run(cmd, **kw):
    """Run a command and print its output in the cell (a notebook does not show subprocess output by itself)."""
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    if r.stdout.strip(): print(r.stdout.strip()[-3000:], flush=True)
    if r.returncode: raise RuntimeError(f'failed ({r.returncode}): {" ".join(cmd)}')


def progress(blocks, bsize, total):
    done = blocks * bsize
    if blocks % 500 == 0 or done >= total:
        print(f'  {min(done, total) // 1_000_000} / {total // 1_000_000} MB', flush=True)


# 1. repo (without LFS files) + the video straight from GitHub LFS
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
REPO = os.path.join(WORK, 'fotevolley')
if not os.path.isdir(os.path.join(REPO, 'data', 'boxes')):
    print('1/4 cloning fotevolley ...', flush=True)
    run(['rm', '-rf', REPO])
    run(['git', 'clone', '-q', '--depth', '1', '-b', BRANCH, 'https://github.com/tomerzi/fotevolley', REPO],
        env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'})
VIDEO = os.path.join(REPO, 'video', 'main_camera_clean_v3.mp4')
if os.path.getsize(VIDEO) < 1_000_000:   # still the LFS pointer file
    print('2/4 downloading video (~285 MB) ...', flush=True)
    urllib.request.urlretrieve(f'https://media.githubusercontent.com/media/tomerzi/fotevolley/{BRANCH}/video/main_camera_clean_v3.mp4',
                               VIDEO, progress)
print('video', os.path.getsize(VIDEO) // 1_000_000, 'MB', flush=True)

# 2. Sapiens2 code + checkpoint
S2 = os.path.join(WORK, 'sapiens2')
if not os.path.isdir(os.path.join(S2, 'sapiens')):
    print('3/4 cloning sapiens2 + installing packages ...', flush=True)
    run(['rm', '-rf', S2])
    run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/facebookresearch/sapiens2', S2])
    run([sys.executable, '-m', 'pip', 'install', '-q', 'iopath', 'termcolor', 'prettytable', 'accelerate', 'safetensors', 'timm'])
sys.path[:0] = [S2, os.path.join(S2, 'sapiens', 'pose', 'tools', 'vis')]
print(f'4/4 downloading sapiens2-pose-{MODEL} checkpoint ...', flush=True)
from huggingface_hub import hf_hub_download
CKPT = hf_hub_download(f'facebook/sapiens2-pose-{MODEL}', f'sapiens2_{MODEL}_pose.safetensors')
print('checkpoint', os.path.getsize(CKPT) // 1_000_000, 'MB', flush=True)

import cv2, numpy as np, pandas as pd, torch
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.models import init_model
from pose_render_utils import visualize_keypoints

from concurrent.futures import ThreadPoolExecutor
devs = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']   # GPU T4 x2 -> one model copy per GPU
print('devices', [torch.cuda.get_device_name(d) for d in devs] if devs[0] != 'cpu' else 'CPU (set Accelerator to GPU!)', flush=True)
POSE_DIR = os.path.join(S2, 'sapiens', 'pose')
os.chdir(POSE_DIR)   # Sapiens2 configs use paths relative to sapiens/pose
models = [init_model(f'configs/keypoints308/shutterstock_goliath_3po/sapiens2_{MODEL}_keypoints308_shutterstock_goliath_3po-1024x768.py', CKPT, device=d)
          for d in devs]
os.chdir(WORK)
model = models[0]   # its pipeline / preprocessor / config are used for the (CPU) crop preparation
for t in model.pipeline.transforms:   # our MARGIN replaces the model's default 1.25x box padding
    if hasattr(t, 'padding'): t.padding = 1.0
pool = ThreadPoolExecutor(len(devs))
meta = parse_pose_metainfo(dict(from_file=os.path.join(POSE_DIR, 'configs/_base_/keypoints308.py')))
codec_cfg = dict(model.cfg.codec); codec_cfg.pop('type')
codec = UDPHeatmap(**codec_cfg)
names = [meta['keypoint_id2name'][i] for i in range(meta['num_keypoints'])]
amp = devs[0] != 'cpu'   # fp16 on the T4; falls back to fp32 for a frame if fp16 gives inf/nan

# 3. frame range (skipping frames already done by an earlier run in this folder) and its boxes
cap = cv2.VideoCapture(VIDEO)
fps, W, H = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
END = min(END or 10**9, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
done = set()
for old in glob.glob(os.path.join(WORK, 'sapiens2_pose_v3_*.csv')):
    done |= set(pd.read_csv(old, usecols=['frame']).frame)
first = next((f for f in range(START, END) if f not in done), END)
if first > START: print(f'frames {START}-{first - 1} are already in earlier output, starting at {first}', flush=True)
last = min(END, first + LIMIT) if LIMIT else END
N = last - first
boxes = pd.read_csv(os.path.join(REPO, 'data', 'boxes', 'player_boxes_v3video.csv'))
boxes = boxes[(boxes.frame >= first) & (boxes.frame < last) & boxes[['x1', 'y1', 'x2', 'y2']].notna().all(1)]
by_frame = {f: g for f, g in boxes.groupby('frame')}
print(f'frames {first}-{last - 1} ({N} frames, {N / fps / 60:.1f} min of video), {len(boxes)} player boxes, margin {MARGIN:.0%} per side,'
      f' flip test {"on" if FLIP_TEST else "off"}, model sapiens2-{MODEL} on {len(devs)} device(s)', flush=True)
if N <= 0: sys.exit('nothing to do')


def predict(img, bbs, use_amp):
    ins, metas = [], []
    for bb in bbs:
        d = model.pipeline(dict(img=img, bbox=bb[None].astype(np.float32), bbox_score=np.ones(1, np.float32)))
        d = model.data_preprocessor(d)
        ins.append(d['inputs']); metas.append(d['data_samples']['meta'])
    x = torch.cat(ins)

    def infer(m, d, xi):   # one chunk of crops on one GPU
        xi = xi.to(d)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            h = m(xi)
            if FLIP_TEST:
                h = (h + m(xi.flip(-1)).flip(-1)[:, meta['flip_indices']]) / 2
        return h.float().cpu()

    parts = [(m, d, xi) for m, d, xi in zip(models, devs, x.chunk(len(devs))) if len(xi)]
    hm = torch.cat(list(pool.map(lambda a: infer(*a), parts))).numpy()
    if not np.isfinite(hm).all():
        return None
    out = []
    for h, m in zip(hm, metas):
        k, s = codec.decode(h)
        k = k / m['input_size'] * m['bbox_scale'] + m['bbox_center'] - 0.5 * m['bbox_scale']
        out.append((k[0], s[0]))
    return out


OUT = os.path.join(WORK, f'sapiens2_pose_v3_{first:05d}-{last - 1:05d}')
OUT_CSV, OUT_MP4 = OUT + '.csv', OUT + '.mp4'
cols = ['frame', 'v2_frame', 'player', 'team', 'x1', 'y1', 'x2', 'y2', 'mx1', 'my1', 'mx2', 'my2'] + [f'{n}_{c}' for n in names for c in 'xys']
rows = []


def save():   # append the rows since the last save, so the file never has to be rewritten
    global rows
    if rows: pd.DataFrame(rows, columns=cols).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
    rows = []


vw = cv2.VideoWriter(OUT_MP4 + '.tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)) if VIDEO_OUT else None
cap.set(cv2.CAP_PROP_POS_FRAMES, first)
t0 = time.time()
for f in range(first, last):
    ok, img = cap.read()
    if not ok: break
    g = by_frame.get(f)
    kps, scs = [], []
    if g is not None:
        b = g[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        bm = b + np.stack([-bw, -bh, bw, bh], 1) * MARGIN   # not clipped: the crop pads outside the image
        res = predict(img, bm, amp) or predict(img, bm, False)
        for (_, r), bb, (k, s) in zip(g.iterrows(), bm, res):
            rows.append([f, r.v2_frame, r.player, r.team, r.x1, r.y1, r.x2, r.y2, *bb.round(1), *np.c_[k.round(1), s.round(3)].ravel()])
            kps.append(k); scs.append(s)
    i = f - first
    if (i + 1) % 25 == 0 or f + 1 == last:
        el = time.time() - t0
        print(f'frame {f + 1}/{last}  {el / 60:.1f} min  ~{el / (i + 1) * (N - i - 1) / 60:.0f} min left', flush=True)
    if (i + 1) % 250 == 0: save()
    if vw is None: continue
    vis = img[:, :, ::-1].copy()
    if kps:
        vis = visualize_keypoints(image=vis, keypoints=kps, keypoints_visible=[np.ones_like(s) > 0 for s in scs],
                                  keypoint_scores=scs, radius=3, thickness=2, kpt_thr=KPT_THR,
                                  skeleton=meta['skeleton_links'], kpt_color=meta['keypoint_colors'],
                                  link_color=meta['skeleton_link_colors'])
    vis = np.ascontiguousarray(vis[:, :, ::-1])
    if g is not None:
        for (_, r), bb in zip(g.iterrows(), bm):
            cv2.rectangle(vis, tuple(bb[:2].astype(int)), tuple(bb[2:].astype(int)), (0, 255, 255), 1)
            cv2.putText(vis, r.player, (int(bb[0]), int(bb[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f'frame {f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vw.write(vis)
save()
if vw is None:
    print('saved', OUT_CSV, flush=True); sys.exit()
vw.release()
print('saved', OUT_CSV, '| encoding video ...', flush=True)
# re-encode to H.264 so the video plays in the browser / Kaggle viewer
import shutil
if shutil.which('ffmpeg') and subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', OUT_MP4 + '.tmp.mp4', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', OUT_MP4]).returncode == 0:
    os.remove(OUT_MP4 + '.tmp.mp4')
else:
    os.replace(OUT_MP4 + '.tmp.mp4', OUT_MP4)
print('video', OUT_MP4, flush=True)
