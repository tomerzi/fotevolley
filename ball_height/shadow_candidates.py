"""Find candidate ball-shadow blobs on the sand in every frame.

The sun is strong, so the ball throws a crisp dark elliptical shadow on the sand. Camera ray + sun ray through the
shadow give a direct, physics-free triangulation of the ball ("second view"), which is used later both as an
independent check and as an extra measurement.

Here we only extract blob candidates: for each frame, D = (per-shot median background) - frame is filtered with a
ball-shadow-sized centre-surround box filter (sizes follow the perspective of the sand plane) and the strongest
isolated dark blobs are written to shadow_candidates.csv.

usage: python shadow_candidates.py camera.json out.csv [first_shot last_shot]
"""
import sys, os, subprocess, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera
from data import load_ball, ROOT
import pandas as pd

VIDEO = os.path.join(ROOT, 'main_camera_clean_v2.mp4')
W, H = 1920, 1080
V0 = 330                 # first image row that can show sand
STRIDE = 2
NCAND = 14


def read_gray(start, n):
    cmd = ['ffmpeg', '-v', 'error', '-ss', '%.3f' % max(start / 25.0 - 0.0001, 0), '-i', VIDEO, '-frames:v', str(n),
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8)
    return a[:(len(a) // (W * H)) * W * H].reshape(-1, H, W)


def geometry(cam):
    """per-pixel validity mask of the sand area and per-row shadow box half-sizes"""
    vv, uu = np.mgrid[V0:H:STRIDE, 0:W:STRIDE]
    uv = np.stack([uu.ravel(), vv.ravel()], 1).astype(float)
    Gp = cam.to_plane(uv, 0.0)
    X, Y = Gp[:, 0].reshape(uu.shape), Gp[:, 1].reshape(uu.shape)
    mask = (X > -4.5) & (X < 20.5) & (Y > -3.0) & (Y < 10.3)
    # broadcast graphics and the near net post
    mask &= ~((uu < 240) & (vv > 950)) & ~((uu > 1715) & (vv > 905)) & ~((uu > 985) & (uu < 1050) & (vv > 600))
    # local scale (px per metre) along X and Y on the sand
    e = 0.05
    px = cam.project(Gp); pX = cam.project(Gp + [e, 0, 0]); pY = cam.project(Gp + [0, e, 0])
    sx = (np.linalg.norm(pX - px, axis=1) / e).reshape(uu.shape)
    sy = (np.abs(pY[:, 1] - px[:, 1]) / e).reshape(uu.shape)
    hx = np.clip(np.round(0.13 * sx), 5, 30).astype(int)
    hy = np.clip(np.round(0.12 * sy), 3, 20).astype(int)
    return uu, vv, mask, hx, hy


def box_mean(ii, u0, v0, u1, v1):
    """mean of the image over [v0,v1) x [u0,u1) using the padded integral image ii"""
    u0 = np.clip(u0, 0, W); u1 = np.clip(u1, 0, W); v0 = np.clip(v0, 0, H); v1 = np.clip(v1, 0, H)
    s = ii[v1, u1] - ii[v0, u1] - ii[v1, u0] + ii[v0, u0]
    return s, (u1 - u0) * (v1 - v0)


def process_shot(frames_gray, geo, boxes_by_frame, frame0):
    uu, vv, mask, hx, hy = geo
    n = len(frames_gray)
    step = max(1, n // 40)
    bg = np.median(frames_gray[::step], axis=0).astype(np.float32)
    out = []
    for k in range(n):
        D = bg - frames_gray[k].astype(np.float32)
        Dp = np.clip(D, 0, 120)
        ii = np.zeros((H + 1, W + 1), np.float64)
        ii[1:, 1:] = Dp.cumsum(0).cumsum(1)
        si, ai = box_mean(ii, uu - hx, vv - hy, uu + hx + 1, vv + hy + 1)
        so, ao = box_mean(ii, uu - 2 * hx, vv - 2 * hy, uu + 2 * hx + 1, vv + 2 * hy + 1)
        inner = si / ai
        ring = (so - si) / np.maximum(ao - ai, 1)
        score = np.where(mask, inner - ring, -1e9)
        # do not look inside the players (dark hair / shorts)
        for (x1, y1, x2, y2) in boxes_by_frame.get(frame0 + k, []):
            a0 = max(0, int((y1 - V0) // STRIDE)); a1 = max(0, int((y2 - V0) // STRIDE) + 1)
            b0 = max(0, int(x1 // STRIDE)); b1 = max(0, int(x2 // STRIDE) + 1)
            score[a0:a1, b0:b1] = -1e9
        sc = score.copy()
        for _ in range(NCAND):
            idx = int(np.argmax(sc))
            r, c = divmod(idx, sc.shape[1])
            s = sc[r, c]
            if s < 10:
                break
            u, v = int(uu[r, c]), int(vv[r, c])
            bx, by = int(hx[r, c]), int(hy[r, c])
            # sub-pixel centroid of the dark blob
            y0, y1, x0, x1 = max(v - by - 2, 0), min(v + by + 3, H), max(u - bx - 2, 0), min(u + bx + 3, W)
            patch = np.clip(Dp[y0:y1, x0:x1] - 0.5 * inner[r, c], 0, None)
            if patch.sum() > 0:
                ys, xs = np.mgrid[y0:y1, x0:x1]
                cu, cv = (patch * xs).sum() / patch.sum(), (patch * ys).sum() / patch.sum()
            else:
                cu, cv = u, v
            out.append((frame0 + k, round(float(cu), 1), round(float(cv), 1), round(float(s), 1), round(float(inner[r, c]), 1), bx, by))
            r0, r1 = max(0, r - 2 * by // STRIDE - 1), r + 2 * by // STRIDE + 2
            c0, c1 = max(0, c - 2 * bx // STRIDE - 1), c + 2 * bx // STRIDE + 2
            sc[r0:r1, c0:c1] = -1e9
    return out


def main():
    cam = Camera.from_json(sys.argv[1])
    out_path = sys.argv[2]
    b = load_ball()
    shots = sorted(b.shot.unique())
    if len(sys.argv) > 4:
        shots = [s for s in shots if int(sys.argv[3]) <= s <= int(sys.argv[4])]
    pb = pd.read_csv(os.path.join(ROOT, 'correct_bbox', 'player_boxes_v9.csv'), usecols=['frame', 'x1', 'y1', 'x2', 'y2'])
    boxes = {}
    for f, x1, y1, x2, y2 in pb.itertuples(index=False):
        # keep the feet area searchable: shrink the box bottom a little, pad the sides
        boxes.setdefault(int(f), []).append((x1 - 6, y1 - 8, x2 + 6, y2 - 4))
    geo = geometry(cam)
    t0 = time.time()
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'u', 'v', 'score', 'dark', 'hx', 'hy'])
        for s in shots:
            g = b[b.shot == s]
            f0, n = int(g.frame.min()), len(g)
            fr = read_gray(f0, n)
            rows = process_shot(fr, geo, boxes, f0)
            w.writerows(rows); fh.flush()
            print('shot', s, 'frames', f0, f0 + n - 1, 'read', len(fr), 'cands', len(rows), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
