"""Player geometry derived from the 2-D pose: where each player stands on the sand and which player /
body part is closest to the ball in the image."""
import numpy as np
from data import JOINTS, pose_arrays

J = {n: i for i, n in enumerate(JOINTS)}
FOOT = [J[n] for n in ['l_ankle', 'r_ankle', 'l_big_toe', 'r_big_toe', 'l_small_toe', 'r_small_toe', 'l_heel', 'r_heel']]
LEG = [J[n] for n in ['l_knee', 'r_knee', 'l_hip', 'r_hip', 'mid_hip']]
UPPER = [J[n] for n in ['nose', 'neck', 'l_shoulder', 'r_shoulder', 'l_eye', 'r_eye', 'l_ear', 'r_ear']]
ARM = [J[n] for n in ['l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']]
ANKLE_Z = 0.08      # height of the ankle joint above the sand when the foot is planted


class Players:
    def __init__(self, cam, pose_df, n_frames):
        self.cam = cam
        self.arr = pose_arrays(pose_df, n_frames)
        self.names = sorted(self.arr)
        self.n = n_frames
        self.ground = {}
        for pl in self.names:
            xy, sc = self.arr[pl]
            la, ra = xy[:, J['l_ankle']], xy[:, J['r_ankle']]
            ok = ~np.isnan(la[:, 0])
            G = np.full((n_frames, 2), np.nan)
            if ok.any():
                # the planted foot is the lower one in the image only if both are at the same depth; use the mean of both
                # ankles (robust to a single raised leg being closer/farther) - jumps are handled by the robust prior later
                gl = cam.to_plane(la[ok], ANKLE_Z)[:, :2]
                gr = cam.to_plane(ra[ok], ANKLE_Z)[:, :2]
                G[ok] = 0.5 * (gl + gr)
            self.ground[pl] = G

    def nearest(self, frame, uv, radius_px=None):
        """for the ball at pixel uv in `frame`: list of (player, joint index, pixel distance) sorted by distance"""
        out = []
        for pl in self.names:
            xy, sc = self.arr[pl]
            if frame < 0 or frame >= self.n:
                continue
            p = xy[frame]
            if np.isnan(p[:, 0]).all():
                continue
            d = np.hypot(p[:, 0] - uv[0], p[:, 1] - uv[1])
            d = np.where(sc[frame] > 0.3, d, np.inf)
            j = int(np.argmin(d))
            out.append((pl, j, float(d[j])))
        out.sort(key=lambda r: r[2])
        return out

    def scale_px(self, pl, frame):
        """apparent size of the player (pixels per metre, roughly) from the neck-hip length"""
        xy, sc = self.arr[pl]
        a, b = xy[frame, J['neck']], xy[frame, J['mid_hip']]
        return float(np.hypot(*(a - b))) / 0.55 if not np.isnan(a[0]) else np.nan

    def ground_xy(self, pl, frame, win=2):
        G = self.ground[pl][max(0, frame - win):frame + win + 1]
        if np.isnan(G[:, 0]).all():
            return None
        return np.nanmedian(G, 0)

    @staticmethod
    def part(j):
        if j in FOOT:
            return 'foot'
        if j in LEG:
            return 'leg'
        if j in UPPER:
            return 'head/chest'
        return 'arm'
