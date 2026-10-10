# fotevolley

Footvolley match analysis (Israel vs Brazil, main camera): ball and player boxes,
poses, rallies and points.

## Layout

| Folder | What's in it |
|---|---|
| `video/` | `main_camera_clean_v2.mp4`: the full clean video (15,586 frames, 25 fps). Most files use its frame numbers.<br>`main_camera_clean_v3.mp4`: v2 with 32 dead-time ranges removed (13,495 frames).<br>`main_camera_rallies_only.mp4`: only the 60 rallies (13,105 frames). |
| `models/` | YOLO ball detectors `exp-2.pt`, `ballDedect.pt`. ViTPose weights go in `models/vitpose_weights/` (not tracked). |
| `data/boxes/` | **Final reviewed bounding boxes.** `ball_boxes_corrected3.csv` + `player_boxes_v9.csv` (v2 frames); `*_v3video.csv` (v3 frames, `v2_frame` keeps the old number). |
| `data/points/` | `rallies_v2.csv` / `rallies_v3video.csv`: 60 rallies with start/end frame, winner, score.<br>`point_events_corrected.csv`: points from the scoreboard, **current** (points 49-52 fixed); `point_events_v3video.csv` is the same in v3 frames.<br>`point_events_FINAL_VERIFIED.csv`: earlier version, kept for reference. |
| `data/mappings/` | Frame-number conversions: `frame_mapping.csv`, `frame_mapping_v3.csv` (v2 → v3), `frame_mapping_rallies.csv` (rallies video → v2 frame + rally). `homography_px_to_m.npy`: pixels → court metres. |
| `data/tracks/` | Earlier tracking outputs: raw/cleaned player tracks, identities, team sides, referee removal, ball tracks, `player_boxes.csv`. |
| `data/ball_pipeline/` | Outputs of `scripts/ball/` (ball candidates and filled ball boxes). |
| `data/pose/` | Outputs of `scripts/pose/` (ViTPose BODY_25 keypoints, suspect frames). |
| `scripts/ball/`, `scripts/pose/` | Detection, filling, pose and render scripts. Run them from anywhere; paths are relative to the repo root. |
| `docs/` | Notes and plans. |
| `archive/` | Old versions and check videos, not tracked by git. |

## Frame numbering

Unless a file name says `v3video`, frame numbers refer to `video/main_camera_clean_v2.mp4`.
Use `data/mappings/` to convert between videos.
