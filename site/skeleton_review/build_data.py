"""Build the local browser data from the repository's v3 pose and touch tables."""
import base64
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def build_pose(source, output):
    with source.open(encoding="utf-8-sig") as stream:
        names = [field[:-2] for field in next(csv.reader(stream)) if field.endswith("_s")]
    if len(names) != 308:
        raise ValueError("Expected 308 Sapiens2 keypoints. Download the CSV with Git LFS first.")

    columns = ["x1", "y1", "x2", "y2"] + [
        f"{name}_{component}" for name in names for component in ("x", "y", "s")
    ]
    frames = np.full((13495, 2, len(columns)), -32768, dtype="<i2")
    players = {"ISR_A": 0, "ISR_B": 1, "Ron_Ben_Ishay": 0, "Maor_Haas": 1}
    for rows in pd.read_csv(source, usecols=["frame", "player", *columns], chunksize=3500):
        rows = rows[rows.player.isin(players)]
        values = rows[columns].to_numpy(dtype="float32", copy=True)
        values[:, :4] *= 4
        values[:, 4::3] *= 4
        values[:, 5::3] *= 4
        values[:, 6::3] *= 1000
        finite = values[np.isfinite(values)]
        if np.isinf(values).any() or (finite <= -32768).any() or (finite >= 32767).any():
            raise ValueError("Pose values exceed the viewer's packed range")
        frames[rows.frame.to_numpy(), rows.player.map(players).to_numpy()] = np.rint(
            np.nan_to_num(values, nan=-32768)
        ).astype("<i2")

    for start in range(0, len(frames), 250):
        index = start // 250
        packed = zlib.compress(frames[start:start + 250].tobytes(), level=4)
        text = base64.b64encode(packed).decode("ascii")
        (output / f"pose-{index}.js").write_text(f'window.receivePose({index},"{text}");', encoding="ascii")
    (output / "joints.js").write_text("window.jointNames=" + json.dumps(names) + ";", encoding="utf-8")


def build_touches(source, output):
    players = {"Ron_Ben_Ishay": 0, "Maor_Haas": 1}
    cases = []
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["player"] not in players or row["in_play"] != "True":
                continue
            part, side = row["body_part"], row["limb_side"]
            facing = row["facing_label"] or row["facing"]
            knee = {"left": "knee_l_ok", "right": "knee_r_ok"}.get(side)
            review = part == "foot" and (facing == "away" or row.get(knee) != "True")
            cases.append([int(row["frame"]), players[row["player"]], int(row["rally"]), int(row["touch"]),
                          part, side, row["action"], int(review)])
    (output / "touches.js").write_text("window.touchCases=" + json.dumps(cases) + ";", encoding="utf-8")


if __name__ == "__main__":
    viewer = Path(__file__).resolve().parent
    data = viewer.parents[1] / "data"
    output = viewer / "data"
    output.mkdir(exist_ok=True)
    build_pose(data / "pose/sapiens2/sapiens2_pose_v3_00000-13494.csv", output)
    build_touches(data / "analysis/touch_body.csv", output)
