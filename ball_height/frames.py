"""frame access through ffmpeg (no OpenCV in this environment)"""
import subprocess, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO = os.path.join(ROOT, 'main_camera_clean_v2.mp4')
W, H = 1920, 1080
def read_frames(start, n=1, video=VIDEO):
    """n consecutive frames starting at frame index `start` -> uint8 array (n, H, W, 3)"""
    t = start / 25.0
    cmd = ['ffmpeg', '-v', 'error', '-ss', '%.3f' % max(t - 0.0001, 0), '-i', video, '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8)
    return a[:(len(a) // (W * H * 3)) * W * H * 3].reshape(-1, H, W, 3)
