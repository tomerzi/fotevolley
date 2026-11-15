"""Verification video: the reconstructed ball height drawn on top of the original footage.

For every frame with a known height it draws
  * a vertical "plumb line" from the ball down to the point on the sand directly below it,
  * a red ellipse where the ball's shadow MUST be if the reconstructed 3-D position is right
    (in sunny shots you can check by eye that it sits on the real shadow),
  * the height in metres next to the ball and a running height chart.

usage: python render_overlay.py out.mp4 [first_frame last_frame]
"""
import sys, os, subprocess
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera
from data import load_ball, ROOT
from miniplot import font
HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(ROOT, 'main_camera_clean_v2.mp4')
OW, OH = 1280, 720
SC = OW / 1920.0
COL = {'flight+shadow': (60, 170, 255), 'flight': (255, 150, 40), 'ground': (190, 140, 90), 'held': (190, 120, 255),
       'shadow': (190, 120, 255), 'player-depth': (190, 120, 255), 'interp': (170, 170, 170)}
LABEL = {'flight+shadow': 'flight (camera + shadow)', 'flight': 'flight (camera only)', 'ground': 'on the sand',
         'held': 'carried', 'shadow': 'at a player (shadow)', 'player-depth': 'at a player', 'interp': 'interpolated'}


def main(out_path, f_first=0, f_last=None):
    cam = Camera.from_json(os.path.join(HERE, 'camera.json'))
    d = pd.read_csv(os.path.join(HERE, 'ball_height.csv'))
    b = load_ball()
    sun = pd.read_csv(os.path.join(HERE, 'sun_per_shot.csv')).set_index('shot')
    f_last = int(d.frame.max()) if f_last is None else f_last
    n = f_last - f_first + 1
    z = d.z_m.values; fr = d.frame.values
    P = d[['x_m', 'y_m', 'z_m']].values
    okP = np.isfinite(P).all(1)
    pix = np.full((len(d), 2), np.nan); nad = np.full((len(d), 2), np.nan); shp = np.full((len(d), 2), np.nan)
    pix[okP] = cam.project(P[okP])
    N = P.copy(); N[:, 2] = 0.0
    nad[okP] = cam.project(N[okP])
    s_ab = np.stack([d.shot.map(sun.a_fit).values, d.shot.map(sun.b_fit).values], 1)
    S = np.c_[P[:, :2] + P[:, 2:3] * s_ab, np.zeros(len(d))]
    oks = okP & np.isfinite(S).all(1)
    shp[oks] = cam.project(S[oks])
    # local size of the shadow ellipse (px) : 0.11 m radius on the sand
    e = 0.11
    sx = np.full(len(d), np.nan); sy = np.full(len(d), np.nan)
    sx[oks] = np.linalg.norm(cam.project(S[oks] + [e, 0, 0]) - shp[oks], axis=1) * 1.25
    sy[oks] = np.linalg.norm(cam.project(S[oks] + [0, e, 0]) - shp[oks], axis=1) * 1.25
    bx = b[['center_x', 'center_y']].values
    f_big, f_small = font(22), font(15)
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', '%.3f' % max(f_first / 25.0 - 0.0001, 0), '-i', VIDEO, '-frames:v', str(n),
                            '-vf', 'scale=%d:%d' % (OW, OH), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (OW, OH), '-r', '25', '-i', '-',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '27', '-pix_fmt', 'yuv420p', out_path], stdin=subprocess.PIPE)
    # chart geometry
    CX0, CY0, CW, CH = 470, 8, 420, 96
    ZMAX = 6.0
    for k in range(n):
        raw = dec.stdout.read(OW * OH * 3)
        if len(raw) < OW * OH * 3:
            break
        f = f_first + k
        im = Image.frombuffer('RGB', (OW, OH), raw, 'raw', 'RGB', 0, 1).copy()
        dr = ImageDraw.Draw(im, 'RGBA')
        i = f                                            # rows are frame-indexed
        me = d.method.values[i] if isinstance(d.method.values[i], str) else ''
        col = COL.get(me, (255, 255, 255))
        # chart
        dr.rectangle([CX0, CY0, CX0 + CW, CY0 + CH], fill=(0, 0, 0, 150))
        ynet = CY0 + CH - 2.2 / ZMAX * CH
        dr.line([CX0, ynet, CX0 + CW, ynet], fill=(255, 255, 255, 110))
        dr.text((CX0 + 3, ynet - 15), 'net 2.2 m', fill=(255, 255, 255, 170), font=f_small)
        lo, hi = f - 110, f + 30
        shot = d.shot.values[i]
        pts = []
        for j in range(max(lo, 0), min(hi, len(d) - 1) + 1):
            if d.shot.values[j] == shot and np.isfinite(z[j]):
                pts.append((CX0 + (j - lo) / (hi - lo) * CW, CY0 + CH - min(z[j], ZMAX) / ZMAX * CH))
            else:
                if len(pts) > 1:
                    dr.line(pts, fill=(255, 255, 255, 230), width=2)
                pts = []
        if len(pts) > 1:
            dr.line(pts, fill=(255, 255, 255, 230), width=2)
        xc = CX0 + (f - lo) / (hi - lo) * CW
        dr.line([xc, CY0, xc, CY0 + CH], fill=(255, 255, 0, 200))
        if np.isfinite(z[i]):
            bu, bv = (bx[i] if np.isfinite(bx[i, 0]) else pix[i]) * SC
            nu, nv = nad[i] * SC
            # plumb line + foot point
            dr.line([bu, bv, nu, nv], fill=col + (230,), width=2)
            dr.line([nu - 7, nv, nu + 7, nv], fill=col + (255,), width=2)
            dr.line([nu, nv - 4, nu, nv + 4], fill=col + (255,), width=2)
            # expected shadow
            if np.isfinite(shp[i, 0]) and z[i] > 0.25:
                su, sv = shp[i] * SC
                rx, ry = max(4, sx[i] * SC), max(2.5, sy[i] * SC)
                dr.ellipse([su - rx, sv - ry, su + rx, sv + ry], outline=(255, 40, 40, 255), width=2)
            dr.ellipse([bu - 12, bv - 12, bu + 12, bv + 12], outline=col + (255,), width=2)
            txt = '%.2f m' % z[i]
            tx, ty = bu + 16, bv - 12
            dr.rectangle([tx - 3, ty - 2, tx + 78, ty + 24], fill=(0, 0, 0, 140))
            dr.text((tx, ty), txt, fill=col + (255,), font=f_big)
            hud = 'frame %d   ball height %.2f m  (+-%.2f)   %s' % (f, z[i], d.z_sigma_m.values[i], LABEL.get(me, me))
        else:
            hud = 'frame %d   ball height: unknown' % f
        dr.rectangle([CX0 - 110, OH - 34, CX0 + CW + 110, OH - 8], fill=(0, 0, 0, 150))
        dr.text((CX0 - 104, OH - 31), hud, fill=(255, 255, 255, 255), font=f_small)
        enc.stdin.write(im.tobytes())
    enc.stdin.close(); enc.wait(); dec.kill()


if __name__ == '__main__':
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 0, int(a[3]) if len(a) > 3 else None)
