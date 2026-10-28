# fotevolley

Footvolley match analysis (Israel vs Brazil, main camera): ball and player boxes,
poses, rallies and points.

## Layout

| Folder | What's in it |
|---|---|
| `video/` | `main_camera_clean_v2.mp4`: the full clean video (15,586 frames, 25 fps). Most files use its frame numbers.<br>`main_camera_clean_v3.mp4`: v2 with 32 dead-time ranges removed (13,495 frames).<br>`main_camera_rallies_only.mp4`: only the 60 rallies (13,105 frames). |
| `models/` | YOLO ball detectors `exp-2.pt`, `ballDedect.pt`. ViTPose weights go in `models/vitpose_weights/` (not tracked). |
| `data/boxes/` | **Final reviewed bounding boxes.** `ball_boxes_corrected3.csv` + `player_boxes_v9.csv` (v2 frames); `*_v3video.csv` (v3 frames, `v2_frame` keeps the old number). |
| `data/court/` | `player_squares_v3.csv`: every player's 3×3 court square per frame (L1–L9 left side, R1–R9 right, numbered from the far sideline; `out` = outside the lines), from feet position + homography. Made by `scripts/court/court_squares.py`, which also renders `video/court_squares_v3.mp4` (whole match, grid painted on the sand, each player's square tagged and tinted). |
| `data/points/` | `rallies_v2.csv` / `rallies_v3video.csv`: 60 rallies with start/end frame, winner, score.<br>`point_events_corrected.csv`: points from the scoreboard, **current** (points 49-52 fixed); `point_events_v3video.csv` is the same in v3 frames.<br>`point_events_FINAL_VERIFIED.csv`: earlier version, kept for reference. |
| `data/mappings/` | Frame-number conversions: `frame_mapping.csv`, `frame_mapping_v3.csv` (v2 → v3), `frame_mapping_rallies.csv` (rallies video → v2 frame + rally). `homography_px_to_m.npy`: pixels → court metres. |
| `data/tracks/` | Earlier tracking outputs: raw/cleaned player tracks, identities, team sides, referee removal, ball tracks, `player_boxes.csv`. |
| `data/ball_pipeline/` | Outputs of `scripts/ball/` (ball candidates and filled ball boxes). |
| `data/pose/` | Outputs of `scripts/pose/` (ViTPose BODY_25 keypoints, suspect frames).<br>`sapiens2/sapiens2_pose_v3_00000-13494.csv`: Sapiens2-1B 308 keypoints (x, y, score) for every reviewed player box, **v3 frames** (`v2_frame` column too); `.mp4` is its check video (mp4v, open in VLC). Made by `scripts/pose/sapiens2_gpu.py`. |
| `scripts/ball/`, `scripts/pose/` | Detection, filling, pose and render scripts. Run them from anywhere; paths are relative to the repo root. |
| `data/analysis/` | Made by `scripts/analysis/` (run order in `docs/ANALYSIS_PLAN.md`). `player_positions_v3.csv`: every player's smoothed court position, velocity and speed per frame, with posture (stand / lying / air / low / hidden...), jump and airborne flags and the raw ground point. `touch_actions_v3.csv`: serve / reception / set / attack per touch. `team_spacing_v3.csv`: the defending pair's distance at every attack (conceded or returned). `player_speed_rallies_v3.csv`, `touch_speed_v3.csv`: distance and speeds per rally / before each touch. `coverage_*_v3.csv`, `coverage_heatmaps_v3.png`: area covered and time per square. `chest_stance_v3.csv`: distance between the feet at chest touches. `ball_height_v2.csv`: ball height above the sand (m) and court position for every **v2** frame; camera `camera_v2.npz` (blue lines 16 × 8 m, lens distortion, net top 2.20 m, the near antenna's 10 cm stripes; no player heights), all flights fitted together with gravity, air drag (`ball_height_settings.json`) and no jumps between flights, pinned by 94 hand-marked chest touches and 41 hand-marked sand landings, box sizes corrected from the landings. Held-out error ~0.12 m. `player_statures.json`: the four players' heights measured from the video. Made by `scripts/experiments/improve/run_final.sh` + `chain.py` (see `data/experiments/improve/README.md`); `scripts/ball_height/render_video.py` draws ball height and player speeds on the video. Lengths are in court units (court lines = 18 × 9); see `docs/ANALYSIS_PLAN.md` for the open 16 × 8 question. |
| `docs/` | Notes and plans. `ANALYSIS_PLAN.md`: match facts (event, players, rules) and the plan for ball height, actions, speeds, distances and areas. |
| `archive/` | Old versions and check videos, not tracked by git. |

## Frame numbering

Unless a file name says `v3video`, frame numbers refer to `video/main_camera_clean_v2.mp4`.
Use `data/mappings/` to convert between videos.
