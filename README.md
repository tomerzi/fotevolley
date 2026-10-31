# fotevolley

Footvolley men's final, **EFVL World Challenge Graz 2022**: Brazil (Franklin, Hiltinho) beat Israel (Maor Haas,
Ron Ben Ishay) 2-0. Players are labelled by name in every file: `Franklin`, `Hiltinho`, `Maor_Haas`, `Ron_Ben_Ishay`
(old codes BRA_A, BRA_B, ISR_A, ISR_B). Coach report: `coach_analysis/`.
Exception: the Git LFS file `data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv` still uses the old codes (its
renamed copy could not be uploaded from the cloud session); map them with `coach_analysis/graz2022_final_player_names.csv`.

Footvolley match analysis (Israel vs Brazil, main camera): ball and player boxes,
poses, rallies and points.

## Verified points (owner review, 2026-10-31)

- `data/points/landing_marks_owner.json`: the project owner's review of all 60 points, made on the review page
  (`coach_analysis/review/landing_marking_tool.html`, clips from `coach_analysis/review/make_landing_clips.py`):
  what happened, the real landing spot or the line an out ball crossed, last touch, and who won on court.
- `data/points/point_end_verified.csv` (`scripts/analysis/point_end_verified.py`): how each point ended
  (landed in / out, out over a line, net, fault, excluded) with consistency checks; 56 of 60 verified, the 4 points
  the video does not show are excluded.
- `data/points/rallies_on_court.csv`: winner of each point **on court** (owner) next to the scoreboard winner, and the
  running score. The on-screen scoreboard was wrong at point 49 (it took a point from Brazil and gave Israel one),
  so in the video Brazil reaches 18 in set 2 at point 58 and play goes on to 10-19; set 1's last point (Brazil's 18th,
  an Israeli serve out) was not filmed: on court the match is 18-14, 18-9.
- `point_positions*.csv` and `formation_points.csv` now use only reliable point moments and the on-court winner.
- `data/analysis/point_landings.csv` is the earlier automatic landing guess, kept for reference; use
  `point_end_verified.csv` instead.

## Layout

| Folder | What's in it |
|---|---|
| `video/` | `main_camera_clean_v2.mp4`: the full clean video (15,586 frames, 25 fps). Most files use its frame numbers.<br>`main_camera_clean_v3.mp4`: v2 with 32 dead-time ranges removed (13,495 frames).<br>`main_camera_rallies_only.mp4`: only the 60 rallies (13,105 frames). |
| `models/` | YOLO ball detectors `exp-2.pt`, `ballDedect.pt`. ViTPose weights go in `models/vitpose_weights/` (not tracked). |
| `data/boxes/` | **Final reviewed bounding boxes.** `ball_boxes_corrected3.csv` + `player_boxes_v9.csv` (v2 frames); `*_v3video.csv` (v3 frames, `v2_frame` keeps the old number). |
| `data/court/` | `player_squares_v3.csv`: every player's 3×3 court square per frame (L1–L9 left side, R1–R9 right, numbered from the far sideline; `out` = outside the lines), from feet position + homography. Made by `scripts/court/court_squares.py`, which also renders `video/court_squares_v3.mp4` (whole match, grid painted on the sand, each player's square tagged and tinted). |
| `data/points/` | `rallies_v2.csv` / `rallies_v3video.csv`: 60 rallies with start/end frame, winner, score.<br>`point_events_corrected.csv`: points from the scoreboard, **current** (points 49-52 fixed); `point_events_v3video.csv` is the same in v3 frames.<br>`point_events_FINAL_VERIFIED.csv`: earlier version, kept for reference. |
| `data/mappings/` | Frame-number conversions: `frame_mapping.csv`, `frame_mapping_v3.csv` (v2 → v3), `frame_mapping_rallies.csv` (rallies video → v2 frame + rally). `homography_px_to_m.npy`: pixels → metres on the 16 × 8 court (`homography_px_to_court_units_18x9.npy`: the old 18 × 9 version, used only by `scripts/experiments/`). |
| `data/tracks/` | Earlier tracking outputs: raw/cleaned player tracks, identities, team sides, referee removal, ball tracks, `player_boxes.csv`. |
| `data/ball_pipeline/` | Outputs of `scripts/ball/` (ball candidates and filled ball boxes). |
| `data/pose/` | Outputs of `scripts/pose/` (ViTPose BODY_25 keypoints, suspect frames).<br>`sapiens2/sapiens2_pose_v3_00000-13494.csv`: Sapiens2-1B 308 keypoints (x, y, score) for every reviewed player box, **v3 frames** (`v2_frame` column too); `.mp4` is its check video (mp4v, open in VLC). Made by `scripts/pose/sapiens2_gpu.py`. |
| `scripts/ball/`, `scripts/pose/` | Detection, filling, pose and render scripts. Run them from anywhere; paths are relative to the repo root. |
| `data/analysis/` | Made by `scripts/analysis/` (run order in `docs/ANALYSIS_PLAN.md`). `player_positions_v3.csv`: every player's smoothed court position, velocity and speed per frame, with posture (stand / lying / air / low / hidden...), jump and airborne flags and the raw ground point. `touch_actions_v3.csv`: serve / reception / set / attack per touch. `team_spacing_v3.csv`: the defending pair's distance at every attack (conceded or returned). `player_speed_rallies_v3.csv`, `touch_speed_v3.csv`: distance and speeds per rally / before each touch. `coverage_*_v3.csv`, `coverage_heatmaps_v3.png`: area covered and time per square. `chest_stance_v3.csv`: distance between the feet at chest touches. `chest_profile.csv` / `chest_profile_by_player.csv`: each player's body and ball at every chest touch (ball height and incoming speed, stance, feet, approach speed, distance from the net, action, rally won), made by `scripts/analysis/chest_profile.py`. `ball_height_v2.csv`: ball height above the sand (m) and court position for every **v2** frame; camera `camera_v2.npz` (blue lines 16 × 8 m, lens distortion, net top 2.20 m, the near antenna's 10 cm stripes; no player heights), all flights fitted together with gravity (drag, fitted gravity and spin were tested and did not help, `data/experiments/physics_models/`) and no jumps between flights, **the ball's shadow on the sand as a second view** (sun azimuth -112.4°, elevation 43.9°, ~5,300 shadows incl. 89 hand-marked, `data/experiments/shadow/`); outside fitted flights the ball is placed at the touching player's depth (185 frames) or by its shadow (133 frames); column `height_sigma_m` = per-frame uncertainty (0.08 m in flights with shadows ... 0.40 m size only), the ball above 2.31 m at every net crossing, pinned by the hand-marked chest touches and sand landings, box sizes corrected from the landings. Held-out errors: landings 0.06 m, chest touches 0.12 m (`ball_height_settings.json`). `player_statures.json`: the four players' heights measured from the video. `point_landings.csv`: where the ball came down at the end of each point (square as the losing team sees it; `scripts/analysis/point_landings.py`). `formation_frames.csv` / `formation_points.csv`: how each pair stands (both at the net / one up one back / side by side, partner distance, who is in front) in every rally frame and at each point (`scripts/analysis/formation.py`). `movement_by_set.csv` / `square_transitions.csv`: set 1 vs set 2 movement and moves between squares (`scripts/analysis/movement_sets.py`). `point_positions.csv` / `point_positions_summary.csv`: where every player stood (16 × 8 square, and the same square as the player sees it: lane × depth) at the moment each of the 60 points ended, won or lost, plus time share per square; made by `scripts/analysis/point_positions.py`. `conceded_partner_distance.csv`: for each of the 60 points, the distance (m, 16 × 8 court) between the two players of the team that lost it, at the moment it was lost (own error → their last touch; otherwise the ball reaching the sand; 23 moments hand-marked in `data/experiments/point_marks/`), made by `scripts/analysis/conceded_distance.py`. Made by `scripts/experiments/shadow/run_final_p3.sh` (`chain.py`, `triangulate.py`, `uncertainty.py`) (see `data/experiments/shadow/README.md`); `scripts/ball_height/render_video.py` draws ball height and player speeds on the video. All lengths, positions and speeds are **metres on the 16 × 8 m court** (net at x = 8), converted from the earlier 18 × 9 court units by `scripts/court/convert_to_16x8.py`. |
| `scripts/site/` | `build_site.py` + `template.html`: the results website (ball height, speeds, chest stance, partner distance at lost points, coverage, players). `python scripts/site/build_site.py OUT_DIR` writes `index.html`, `frames/`, `touches/` into OUT_DIR (which holds the video segments from `render_video.py`). |
| `data/experiments/`, `scripts/experiments/` | Experiments kept aside, each with a README: court scale, physics calibration, chest and sand marks (`make_tool.py` + `template.html` build the marking pages), the joint fit (`improve/`), the two-camera test from broadcast replays (`two_camera/`), and the point-moment marking page (`point_marks/`). |
| `docs/` | Notes and plans. `time_log.md`: time spent per request. `ANALYSIS_PLAN.md`: match facts (event, players, rules) and the plan for ball height, actions, speeds, distances and areas. |
| `archive/` | Old versions and check videos, not tracked by git. |

## Frame numbering

Unless a file name says `v3video`, frame numbers refer to `video/main_camera_clean_v2.mp4`.
Use `data/mappings/` to convert between videos.
