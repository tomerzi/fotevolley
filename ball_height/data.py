"""Loading of the inputs used for the ball-height reconstruction (ball boxes + player pose)."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALL_CSV = os.path.join(ROOT, 'correct_bbox', 'ball_boxes_corrected3.csv')
POSE_CSV = os.path.join(ROOT, 'pose', 'vitpose_body25.csv')
FPS = 25.0
BALL_R = 0.11          # footvolley ball (size 5): circumference 68-70 cm -> radius ~0.11 m
G = 9.81

JOINTS = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist', 'mid_hip',
          'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'l_eye', 'r_ear', 'l_ear',
          'l_big_toe', 'l_small_toe', 'l_heel', 'r_big_toe', 'r_small_toe', 'r_heel']


def load_ball():
    b = pd.read_csv(BALL_CSV)
    b['w'] = b.x2 - b.x1
    b['h'] = b.y2 - b.y1
    # real measurements only: 'interpolated' rows were synthesised by the 2-D pipeline and carry no information
    b['obs'] = b.source.isin(['detected', 'rescued', 'manual'])
    return b


def load_pose():
    p = pd.read_csv(POSE_CSV)
    return p


def pose_arrays(p, n_frames):
    """dict player -> (xy[n_frames, 25, 2], score[n_frames, 25]) with NaN where the player has no pose"""
    out = {}
    for pl, g in p.groupby('player'):
        xy = np.full((n_frames, len(JOINTS), 2), np.nan)
        sc = np.zeros((n_frames, len(JOINTS)))
        fr = g.frame.values
        for j, name in enumerate(JOINTS):
            xy[fr, j, 0] = g[name + '_x'].values
            xy[fr, j, 1] = g[name + '_y'].values
            sc[fr, j] = g[name + '_s'].values
        out[pl] = (xy, sc)
    return out
