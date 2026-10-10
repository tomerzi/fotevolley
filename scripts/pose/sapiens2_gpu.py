"""Sapiens2 308-keypoint pose on one big GPU (H200 / A100 / H100): same input, settings and output as sapiens2_kaggle.py,
about 5x faster.
  python scripts/pose/sapiens2_gpu.py                       whole v3 video
  S2_START=0 S2_END=6000 python scripts/pose/sapiens2_gpu.py
Speed-ups over the Kaggle script:
  - model in bf16 + torch.compile (fuses RoPE / norm / concat kernels): ~29 ms per crop instead of ~50
  - heatmap decoding (argmax + DARK-UDP) on the GPU (same result as UDPHeatmap.decode to ~0.02 px); the CPU version took
    ~0.2 s per crop and moved 60 MB of heatmaps per crop off the GPU
  - crops of several frames go through the model together in fixed-size batches (fixed size -> no recompiling)
  - video reading + crop preparation and skeleton drawing + video writing run in background threads
Settings by environment variable: S2_MODEL (0.4b | 0.8b | 1b | 5b), S2_FLIP (1 | 0), S2_MARGIN, S2_START, S2_END, S2_VIDEO (1 | 0),
S2_LIMIT, S2_BATCH (crops per batch, default 8), S2_COMPILE (1 | 0), S2_WORK (output folder, default ../sapiens2_run next to the repo).
-> $S2_WORK/sapiens2_pose_v3_<first>-<last>.csv / .mp4, same columns as sapiens2_kaggle.py. A rerun skips frames already in a
sapiens2_pose_v3_*.csv in $S2_WORK."""
import glob, os, queue, shutil, subprocess, sys, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get('S2_MODEL', '1b')
MARGIN = float(os.environ.get('S2_MARGIN', 0.15))
FLIP_TEST = os.environ.get('S2_FLIP', '1') == '1'
START = int(os.environ.get('S2_START', 0))
END = int(os.environ.get('S2_END', 0))
VIDEO_OUT = os.environ.get('S2_VIDEO', '1') == '1'
LIMIT = int(os.environ.get('S2_LIMIT', 0))
BATCH = int(os.environ.get('S2_BATCH', 8))
COMPILE = os.environ.get('S2_COMPILE', '1') == '1'
WORK = os.environ.get('S2_WORK', os.path.join(os.path.dirname(REPO), 'sapiens2_run'))
KPT_THR = 0.3        # only for drawing

S2 = os.path.join(WORK, 'sapiens2')
if not os.path.isdir(os.path.join(S2, 'sapiens')):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/facebookresearch/sapiens2', S2], check=True)
sys.path[:0] = [S2, os.path.join(S2, 'sapiens', 'pose', 'tools', 'vis')]

import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sapiens.pose.datasets import parse_pose_metainfo
from sapiens.pose.models import init_model
from pose_render_utils import visualize_keypoints

VIDEO = os.path.join(REPO, 'video', 'main_camera_clean_v3.mp4')
CKPT = hf_hub_download(f'facebook/sapiens2-pose-{MODEL}', f'sapiens2_{MODEL}_pose.safetensors')
POSE_DIR = os.path.join(S2, 'sapiens', 'pose')
os.chdir(POSE_DIR)   # Sapiens2 configs use paths relative to sapiens/pose
model = init_model(f'configs/keypoints308/shutterstock_goliath_3po/sapiens2_{MODEL}_keypoints308_shutterstock_goliath_3po-1024x768.py',
                   CKPT, device='cuda:0')
os.chdir(WORK)
for t in model.pipeline.transforms:   # our MARGIN replaces the model's default 1.25x box padding
    if hasattr(t, 'padding'): t.padding = 1.0
net = model.to(torch.bfloat16).eval()
net = torch.compile(net) if COMPILE else net
meta = parse_pose_metainfo(dict(from_file=os.path.join(POSE_DIR, 'configs/_base_/keypoints308.py')))
names = [meta['keypoint_id2name'][i] for i in range(meta['num_keypoints'])]
flip_idx = torch.tensor(meta['flip_indices'], device='cuda')
codec = dict(model.cfg.codec)
BLUR = codec.get('blur_kernel_size', 11)
INPUT_SIZE = codec['input_size']   # (w, h)
print(f'model sapiens2-{MODEL} on {torch.cuda.get_device_name(0)}, bf16, compile {"on" if COMPILE else "off"}, '
      f'{BATCH} crops per batch', flush=True)


def gpu_decode(hm):
    """Torch port of UDPHeatmap.decode (gaussian): (B, K, H, W) heatmaps -> keypoints (B, K, 2) in input pixels, scores (B, K)."""
    hm = hm.float()
    B, K, H, W = hm.shape
    vals, idx = hm.reshape(B, K, -1).max(-1)
    x, y = (idx % W).float(), (idx // W).float()
    x[vals <= 0], y[vals <= 0] = -1, -1
    # gaussian_blur: zero padding, cv2 kernel (sigma from size), rescaled to the original maximum
    kern = torch.tensor(cv2.getGaussianKernel(BLUR, 0)[:, 0], device=hm.device, dtype=torch.float32)
    p = BLUR // 2
    b = hm.reshape(B * K, 1, H, W)
    b = F.conv2d(F.pad(b, (p, p, 0, 0)), kern.view(1, 1, 1, -1))
    b = F.conv2d(F.pad(b, (0, 0, p, p)), kern.view(1, 1, -1, 1)).reshape(B, K, H, W)
    b = b * (vals / b.reshape(B, K, -1).amax(-1))[..., None, None]
    b = b.clamp(1e-3, 50.0).log()
    b = F.pad(b.reshape(B * K, 1, H, W), (1, 1, 1, 1), mode='replicate').reshape(B, K, -1)
    W2 = W + 2
    i0 = (x + 1 + (y + 1) * W2).long()
    g = lambda o: b.gather(-1, (i0 + o).clamp(0, b.shape[-1] - 1)[..., None])[..., 0]
    i_, ix1, iy1, ix1y1, ix1_y1_, ix1_, iy1_ = g(0), g(1), g(W2), g(W2 + 1), g(-W2 - 1), g(-1), g(-W2)
    dx, dy = 0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)
    dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    eps = float(np.finfo(np.float32).eps)
    a, d = dxx + eps, dyy + eps
    det = a * d - dxy * dxy
    x = x - (d * dx - dxy * dy) / det
    y = y - (-dxy * dx + a * dy) / det
    return torch.stack([x / (W - 1) * INPUT_SIZE[0], y / (H - 1) * INPUT_SIZE[1]], -1), vals


def infer(x):
    """x: (n <= BATCH, 3, H, W) float crops -> keypoints (n, K, 2) in crop-input pixels, scores (n, K), numpy."""
    n = len(x)
    if n < BATCH: x = torch.cat([x, x.new_zeros((BATCH - n, *x.shape[1:]))])   # fixed batch size: no recompiling
    x = x.to('cuda', torch.bfloat16, non_blocking=True)
    with torch.no_grad():
        if FLIP_TEST:
            h = net(torch.cat([x, x.flip(-1)]))
            h = (h[:BATCH].float() + h[BATCH:].float().flip(-1)[:, flip_idx]) / 2
        else:
            h = net(x)
        k, s = gpu_decode(h[:n])
    if not (torch.isfinite(k).all() and torch.isfinite(s).all()):
        raise RuntimeError('non-finite keypoints')
    return k.cpu().numpy(), s.cpu().numpy()


# frame range (skipping frames already done by an earlier run in WORK) and its boxes
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
      f' flip test {"on" if FLIP_TEST else "off"}', flush=True)
if N <= 0: sys.exit('nothing to do')

OUT = os.path.join(WORK, f'sapiens2_pose_v3_{first:05d}-{last - 1:05d}')
OUT_CSV, OUT_MP4 = OUT + '.csv', OUT + '.mp4'
cols = ['frame', 'v2_frame', 'player', 'team', 'x1', 'y1', 'x2', 'y2', 'mx1', 'my1', 'mx2', 'my2'] + [f'{n}_{c}' for n in names for c in 'xys']


def reader(q):
    """Background: read frames in order and prepare the crops of their boxes -> q gets (frame, img, boxes df, margin boxes, crops, metas)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for f in range(first, last):
        ok, img = cap.read()
        if not ok: break
        g = by_frame.get(f)
        bm, crops, metas = None, [], []
        if g is not None:
            b = g[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            bm = b + np.stack([-bw, -bh, bw, bh], 1) * MARGIN   # not clipped: the crop pads outside the image
            for bb in bm:
                d = model.pipeline(dict(img=img, bbox=bb[None].astype(np.float32), bbox_score=np.ones(1, np.float32)))
                d = model.data_preprocessor(d)
                crops.append(d['inputs'].cpu()); metas.append(d['data_samples']['meta'])
        q.put((f, img, g, bm, crops, metas))
    q.put(None)


def writer(q):
    """Background: draw boxes + skeletons and write the check video."""
    vw = cv2.VideoWriter(OUT_MP4 + '.tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    while (item := q.get()) is not None:
        f, img, g, bm, kps, scs = item
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
    vw.release()


rq, wq = queue.Queue(maxsize=4 * BATCH), queue.Queue(maxsize=4 * BATCH)
threading.Thread(target=reader, args=(rq,), daemon=True).start()
wt = threading.Thread(target=writer, args=(wq,), daemon=True) if VIDEO_OUT else None
if wt: wt.start()

rows, pending, n_done, n_saved = [], [], 0, 0
t0 = time.time()


def flush():
    """Run the model on the crops of all pending frames, then emit their rows / video frames in frame order."""
    global pending, n_done
    crops = [c for p in pending for c in p[4]]
    res_k, res_s = [], []
    for i in range(0, len(crops), BATCH):
        k, s = infer(torch.cat(crops[i:i + BATCH]))
        res_k.append(k); res_s.append(s)
    k_all = np.concatenate(res_k) if res_k else None
    s_all = np.concatenate(res_s) if res_s else None
    j = 0
    for f, img, g, bm, crops_f, metas in pending:
        kps, scs = [], []
        for (_, r), bb, m in zip(g.iterrows() if g is not None else [], bm if bm is not None else [], metas):
            k = k_all[j] / m['input_size'] * m['bbox_scale'] + m['bbox_center'] - 0.5 * m['bbox_scale']
            s = s_all[j]; j += 1
            rows.append([f, r.v2_frame, r.player, r.team, r.x1, r.y1, r.x2, r.y2, *bb.round(1), *np.c_[k.round(1), s.round(3)].ravel()])
            kps.append(k); scs.append(s)
        if wt: wq.put((f, img, g, bm, kps, scs))
        n_done += 1
        if n_done % 250 == 0 or f + 1 == last:
            el = time.time() - t0
            print(f'frame {f + 1}/{last}  {el / 60:.1f} min  {n_done / el:.1f} frames/s  ~{el / n_done * (N - n_done) / 60:.0f} min left', flush=True)
            save()
    pending = []


def save():   # append the rows since the last save, so the file never has to be rewritten
    global rows
    if rows: pd.DataFrame(rows, columns=cols).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
    rows = []


n_crops = 0
while (item := rq.get()) is not None:
    pending.append(item); n_crops += len(item[4])
    if n_crops >= BATCH:   # at least one full batch: run all pending crops (a last part-batch is padded)
        flush(); n_crops = 0
flush()
save()
print('saved', OUT_CSV, flush=True)
if wt:
    wq.put(None); wt.join()
    if shutil.which('ffmpeg') and subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', OUT_MP4 + '.tmp.mp4', '-c:v', 'libx264', '-crf', '20',
                                                  '-pix_fmt', 'yuv420p', OUT_MP4]).returncode == 0:
        os.remove(OUT_MP4 + '.tmp.mp4')
    else:
        os.replace(OUT_MP4 + '.tmp.mp4', OUT_MP4)
    print('video', OUT_MP4, flush=True)
