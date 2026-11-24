# Data audit: ISR vs BRA, one match (2 sets)

Scope: the single match in `tomerzi/fotevolley` at commit `64c0e79`. Source files were opened read-only. Repository code was read as text and never run or imported. Everything this analysis wrote is in `footvolley-analysis-v1/`.

## 1. Source inventory and authority

| File | Rows | Role | Used | sha256 (prefix) |
|---|---|---|---|---|
| `correct_bbox/player_boxes_v9.csv` | 60,544 | Authoritative player boxes, feet px → court m, `source`/`review_status` provenance | ✔ | `207699c698b380c9` |
| `correct_bbox/ball_boxes_corrected3.csv` | 15,586 (1 per frame) | Authoritative 2D ball, `source` ∈ detected/rescued/manual/interpolated/missing, `shot` id | ✔ | `167d5c8b292099e8` |
| `point_events_FINAL_VERIFIED.csv` | 60 | Scoreboard point events (verified) | ✔ | `ea9e89df4c9f8f82` |
| `team_side_ranges_full_video.csv` | 10 | Screen half per team colour, per range | ✔ | `764746f267b25287` |
| `homography_px_to_m.npy` | 3×3 float64 | Ground-plane homography px→m (loaded with `allow_pickle=False`) | ✔ | `9bb9474032bd22e3` |
| `frame_mapping.csv` | 16,311 | original → cleaned frame (725 frames deleted) | timebase check | `6d90e1f2162f0696` |
| `pose/vitpose_body25.csv` | 60,511 | 2D BODY_25 pose | ✘ (see §5) | — |
| `player_boxes.csv`, `player_identities.csv`, `player_tracks_*.csv`, `ball_tracks_full.csv`, `ball/ball_boxes.csv` | — | Superseded or earlier pipeline stages | ✘ duplicates | — |
| `main_camera_clean_v2.mp4`, `pose/*.mp4`, `*.pt` | — | **Git LFS pointers only (134 bytes)**. No video or model weights available | ✘ | — |

**There is no video access.** None of the evidence has been visually verified, and no stratified clip review was possible (sample count 0).

## 2. Keys, timebase and joins
- Shared timebase: `frame` of the cleaned video, 25 fps. `time_sec = frame × 0.04` was checked on consecutive kept frames. Player, ball, point and side tables all use cleaned-video frames. Their max frame (15585) agrees.
- `player_boxes_v9`: unique on (`frame`, `player`) with 0 duplicates. There are 4 roster labels (`ISR_A`, `ISR_B`, `BRA_A`, `BRA_B`), each with 14,978–15,319 rows. Not every player is present in every frame. Missing boxes stay missing and are not zero-filled.
- `ball_boxes_corrected3`: exactly 1 row per frame. Observed means detected, rescued or manual: 12,524 frames (80.4%). 548 frames are interpolated and 2,514 are missing (no_detection 1,989; trajectory_break 285; off_screen 213; …). **Interpolated ball frames are excluded from every analysis.**
- Players: 518 `interpolated` boxes (0.5–1.2% per player) are marked `observed=false` and excluded from shape snapshots. 60,266 of 60,544 boxes are `inherited_not_manually_verified`.
- Team ↔ colour ↔ half: ISR = BLUE and BRA = YELLOW. In every side range, ≥97.6% of each team's player rows lie on the half that `team_side_ranges` gives. 9 rally windows contain a side switch, so the half→team mapping is applied per ball run, not per window. (The first version mapped per window and was wrong. This was fixed.)
- Joins are exact on `frame`. No nearest-time joins were used.

## 3. Calibration and normalisation
- Homography court frame: x runs along the court from 0 to 18 m with the **net at x = 9**, and y runs across from 0 to 9 m. The 18×9 m court matches common footvolley dimensions. Re-projecting `feet_x/feet_y` reproduces `court_x/court_y` with p99 error 0.007 m, which shows internal consistency only.
- Orientation was checked by geometry. y=9 projects to the bottom of the image (px y ≈ 963–1009) and is wider in perspective, so it is the camera-side sideline. The net projects to the line (997, 420)→(1057, 985) px.
- **No landmark reprojection residuals are available** because there are no annotated court corners and no video. Metric accuracy is unverified. The near-baseline corners project outside the image (x = −276 and 2486 px), so near-corner positions are extrapolated.
- Team-relative frame: `depth_m = |x − 9|` (0 = net, 9 = baseline). `lateral_m` runs left to right while facing the net: `y` on the left half and `9 − y` on the right half. Anatomical side is never swapped.
- Feet are the bottom of the bounding box, so the ground point is wrong while a player is airborne. No jump flag exists. This adds noise to separation at contact moments.

## 4. Rally and possession segmentation
- 60 scoreboard points form 58 windows `(previous update, this update]`. Two **batch updates** (points 10–11 and 50–51) each hide 2 rallies. Those windows are excluded from rally-level work (56 eligible).
- Set boundary: point 31 is `SET_WIN` and the score resets to 0–1 at point 32. Final scores were BRA 18–13 and BRA 18–11.
- **Timing of the scoreboard relative to play is not verified.** The cleaned video has 69 camera shots, and many windows start mid-rally. The first observed ball side matched the expected server (previous point winner) in only 14/54 windows. Ball launches from the image edge near the end of windows suggest that the next serve sometimes starts before the update. **Consequence:** serves cannot be identified, and "serve receive" is not a usable phase. The first incoming ball in a window is labelled `first_incoming_in_window` only.
- Ball side of net comes from the pixel offset to the projected net line, with a ±40 px neutral band. The ball is airborne, so the homography is **not** applied to it. Runs shorter than 3 frames are dropped as noise. A *clean crossing* means the next run starts within 10 frames in the same shot.
- Possession outcome for an incoming ball (clean crossing into a team's half):
  - `returned`: clean crossing back.
  - `terminal_lost`: the window ends with the ball on that half and that team lost the point.
  - `terminal_won`: the window ends there and that team won the point (for example, an opponent ball out).
  - `returned_unclean`: shot cut or tracking gap. Excluded.
  The terminal labels depend on the unverified scoreboard alignment.
- Plausibility check: the window ended with the ball on the loser's half in 38/56 eligible windows. Balls hit out or into the net make this lower than 100% even when the logic is correct.

## 5. Contact candidates and pose
- No contact, action or body-part labels exist in the repository.
- `contact_candidates.csv` (281 rows) flags points where the 2D ball path is at its lowest image point and reverses vertically next to a player box. These are **inferred and unverified**. 62% fall in the "head" band of the box, a likely bias: low foot and chest contacts are more often occluded or missed (see `ball_hidden` and `no_detection` reasons). 90 of 237 possession runs have zero candidates. Coarse body region is therefore **not** used for coaching findings.
- Pose (`vitpose_body25.csv`) was computed on an earlier box set: box_source counts differ from v9, and `players/FIX_PLAN.md` lists phantom and missing boxes. With no verified contacts there is no valid anchor for technique angles, so pose-based technique metrics are **unavailable** in this run.

## 6. Quality gates applied
| Gate | Rule |
|---|---|
| Player position | `source ≠ interpolated`; court x ∈ [−3, 21], y ∈ [−3, 12] |
| Pair snapshot | both team players observed in the same frame (exact) |
| Ball | observed sources only; no interpolated frames |
| Rally | single-point scoreboard update; batch windows excluded |
| Outcome | returned / terminal_lost only; unclean and terminal_won excluded from failure rate |
| Units | rally window = cluster for bootstraps; frames are never treated as independent |

## 7. Missing data that blocks stronger conclusions
1. **Video.** The LFS objects for `main_camera_clean_v2.mp4` (and the rendered overlays) need to be fetched to verify any event visually.
2. **Rally start/end and serve labels.** A manual pass of serve frame, serving player and rally end frame for the 58 windows would take about 1 hour.
3. **Contact labels:** frame, player, coarse body region and action (reception/set/attack/defence) for at least 2 sets.
4. **Court landmark clicks:** 4–6 visible court points to compute homography residuals.
5. **Ball landing points**, which would allow actual seam and deep-zone analysis.
