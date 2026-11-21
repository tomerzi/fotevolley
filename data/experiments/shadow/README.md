# Experiment: the ball's shadow as a second view, precise ball size, net rule (2026-10-29)

Status: experiment only. `data/analysis/` and the site are unchanged.

## 1. Ball shadow (`scripts/experiments/shadow/`)
The match is played in sun; the ball's shadow on the sand is a second view of the ball: the point under the ball
plus height / tan(elevation) along the sun's horizontal direction must project onto the shadow.
- `detect.py`: static camera -> background = median of frames f-60..f+60; the shadow is the darkest compact blob of
  about the ball's size near the predicted spot (player boxes masked).
- `sun_fit.py`: sun direction fitted to the shadows. Pass 1 (around the site's ball positions): 5,681 shadows,
  azimuth -112.6 deg, elevation 43.9 deg, 37 % within 12 px (the misses were mostly real shadows: the site's ball
  position was off). Pass 2 (around the shadow+net solution): 6,013 shadows, azimuth -112.5 deg (drift +0.11 deg
  per 1000 frames), elevation 43.8 deg, 68 % within 12 px, median 4.5 px. Hand check from player shadows:
  azimuth -103..-114 deg, elevation 43-44 deg.
- `chain.py` SHADOW / SUN / SHADOW_SIG / SHADOW_MAXERR: one shadow residual per frame (sigma 3 px, soft-L1).

## 2. Precise ball size (`scripts/experiments/ball_size/`)
- `circle.py` (edge circle fit): locks onto the ball's inner pattern (~13 px); vs the landings corr 0.37 (box 0.89).
- `silhouette.py` (background subtraction, minor axis): 9,032 clean balls; vs the landings corr 0.86, 11 % scatter
  (box 9 %). In the joint fit it does not help (alone it is worse; with the shadow it changes nothing).

## 3. Net rule on all crossings (`NET=all`, from before)

## Results (sand landings and size fix from the even half; scored on the odd half)
| variant | sand landings (0.11 expected) median / MAE | chest touches: fit - marked height median / MAE | ball-player horizontal distance at chest touches | net crossings below the top (by) |
|---|---|---|---|---|
| site today (gravity only) | 0.18 / 0.11 | +0.13 / 0.18 | 0.43 m | 16 % (0.39 m) |
| net rule on all crossings | 0.25 / 0.14 | +0.13 / 0.18 | 0.43 m | 15 % (0.07 m) |
| silhouette size | 0.36 / 0.25 | +0.15 / 0.18 | 0.45 m | 13 % (0.39 m) |
| shadow (pass 1) | 0.07 / 0.07 | +0.08 / 0.13 | 0.28 m | 12 % (0.34 m) |
| shadow (pass 1), sigma 5 px | 0.08 / 0.11 | +0.08 / 0.13 | 0.30 m | 14 % (0.33 m) |
| shadow + net (pass 1) | 0.07 / 0.07 | +0.07 / 0.13 | 0.28 m | 14 % (0.16 m) |
| shadow + silhouette | 0.08 / 0.07 | +0.10 / 0.13 | 0.29 m | 12 % (0.34 m) |
| shadow (pass 2) | 0.10 / 0.08 | +0.07 / 0.12 | 0.25 m | 12 % (0.28 m) |
| **shadow + net (pass 2)** | **0.10 / 0.08** | **+0.07 / 0.12** | **0.25 m** | **12 % (0.11 m)** |

"fit - marked height" compares with the height read from the hand-marked ball pixel at the player's position (the
direct check); the ball-player distance is how far the fitted ball is, on the sand plane, from the player who
touched it (should be ~0.2-0.3 m: the ball is in front of the chest).
Best: shadow (pass 2) + net rule. Landings now land at the right height (0.10 vs 0.11 m), the ball sits at the
player at chest touches (0.43 -> 0.25 m), heights at chest touches improve (MAE 0.18 -> 0.12 m), and the remaining
"under the net" crossings are only 0.11 m short. `chain_p2_shadow_net.csv`.

## Hand-marked shadows (2026-10-29 evening)
Page: `scripts/experiments/shadow_marks/` (3-second clips around the 100 frames the detector was unsure about).
Marks in `data/experiments/shadow_marks/marks/shadows/`: 89 shadows, 8 no shadow, 3 unclear.
- Where the detector had also fired at the hand-marked frame (71), it is 2.1 px from the hand mark (median; 75 % within
  3 px): the detector is accurate; the misses came from frames it did not fire on or filtered out.
- The final solution WITHOUT these marks predicts the hand-marked shadows within 7.6 px / 0.12 m on the sand (median).
- Added to the fit (`run_hand.sh`, they replace the automatic shadow at the same frame): held-out landings 0.10 / MAE
  0.09 m, chest touches +0.06 / 0.13 m, ball-player 0.25 m, net 11 % (0.13 m) - the same as without them (89 marks
  next to ~4,300 automatic shadows). Promoted (`chain_final_hand.csv` -> `data/analysis/ball_height_v2.csv`).

## Options A1-A5 (2026-10-29 night) — preview only, not on the site yet
Held-out (sand/size fix from the even half, scored on the odd half):

| variant | landings median / MAE | chest touches median / MAE | ball-player | below the net |
|---|---|---|---|---|
| site now (pass 2 + hand shadows) | 0.10 / 0.09 | +0.06 / 0.13 | 0.25 m | 11 % (0.13 m) |
| A3+A4: pass-3 shadows (window 25 px, threshold 0.86) + hand | **0.09 / 0.06** | +0.07 / 0.12 | 0.27 m | 11 % (0.11 m) |
| A1: + camera and sun refined (free) | 0.05 / 0.08 | +0.00 / 0.11 | 0.26 m | 15 % (0.10 m) |
| A1: + camera refined with a small prior, + sun | 0.06 / 0.07 | +0.01 / 0.11 | 0.24 m | 15 % (0.10 m) |
| sun refined only | 0.10 / 0.09 | +0.06 / 0.12 | 0.27 m | 12 % (0.12 m) |

A1 is not used: the camera moves far (focal +105 px, principal point +69 px, 2.5 deg) and fits the court
landmarks worse (rms 2.64 -> 4.15 px); chest touches improve but landings and net crossings get worse, so the two
references disagree by ~6 cm; this needs outside data (venue measurements or a second camera) to settle.
A2 (frames outside fitted flights, `triangulate.py`): ball shadow alone fails at touches (merges with the player's
shadow); at a labelled touch (+-2 frames) the ball is put at the touching player's depth (185 frames): chest touches
outside flights MAE 0.28 -> 0.02 m (vs the hand-marked geometry, which uses the same player position - consistency,
not an independent test); 133 other frames from single-frame shadow triangulation (residual < 3 px; one landing check:
0.39 -> 0.09 m, true 0.11).
A5 (`uncertainty.py`): per-frame sigma by how the frame was measured, from the held-out errors: flights with >= 3
shadows 0.08 m (landings MAE 0.056, rms 0.073), flights with fewer shadows 0.25 m (chest MAE 0.25), touch depth 0.10,
shadow triangulation 0.15, size only 0.40 m (chest MAE 0.28). 67 % of flight frames are in flights with >= 3 shadows.
Final candidate (all marks): `final_p3/ball_height.csv` (`run_final_p3.sh`): landings 0.10 m (MAE 0.03 in-sample),
chest MAE 0.12, net 10 %.

## Ideas 1-2 (2026-11-10)
Held-out, pass-3 + hand shadows, net rule (baseline `run_p3`). Split by shadow coverage of the flight
(`score_by_shadow.py`):

| variant | landings median / MAE | chest median / MAE | chest MAE, flights with / few shadows | landings MAE, with / few | below net |
|---|---|---|---|---|---|
| baseline | 0.09 / 0.06 | +0.07 / 0.12 | 0.115 / 0.252 | 0.056 / 0.078 | 11 % (0.11 m) |
| idea 1: touch anchors (even touches, sigma 0.30 m) | 0.11 / 0.09 | +0.07 / 0.13 | 0.111 / 0.301 | 0.089 / 0.090 | 12 % (0.16 m) |
| idea 2: gravity fitted (G_FIT) | 0.08 / 0.06 | **+0.03 / 0.11** | **0.083 / 0.197** | 0.057 / 0.084 | 11 % (0.15 m) |

Idea 1 does not help: the touching player's tracked position is not close enough to the ball (feet and heads are
0.3-0.5 m from the body axis, positions +-0.15 m). Idea 2: the video's time base is clean (25.000 fps, uniform
timestamps, 38 near-duplicate frames of 13,104 in play), yet with the shadows fixing the depth the fitted gravity is
10.09 m/s^2 (+2.8 %): the reconstructed world is ~2.8 % too large (camera scale), which also explains the chest-touch
bias. Correcting it (fitted g, equivalently shrinking the world by 2.8 %) improves chest touches (bias +0.07 -> +0.03,
MAE in flights with shadows 0.115 -> 0.083) without hurting landings or the net. Not applied yet.

## Moving sun (2026-11-10), gravity fixed at 9.81
The match was in Graz around 25-26 June 2022 (EFVL news of 28 June). With the fitted sun elevation (~44 deg) the sun
moves ~10 deg per hour there, so it moved during the match; the model held the elevation fixed. `SUN_ELEV_DRIFT=1`
lets the elevation change linearly too (`run_p3.sh p3_sunmove`). Fitted: azimuth drift +0.38 deg and elevation
-0.30 deg per 1000 frames (over the match +5.9 deg / -4.6 deg): the elevation FALLS, i.e. afternoon, with the same
ratio as the astronomical sun in Graz (-0.78 vs -0.71). Matching the elevations to the astronomical sun gives
start ~16:02 and end ~16:30 local time (28 min, the length of the broadcast) - an independent confirmation.

| held-out | landings median / MAE | chest median / MAE | chest MAE with / few shadows | landings MAE with / few | below net |
|---|---|---|---|---|---|
| baseline (sun fixed) | 0.09 / 0.06 | +0.07 / 0.12 | 0.115 / 0.252 | 0.056 / 0.078 | 11 % (0.11 m) |
| **moving sun, g = 9.81** | **0.09 / 0.05** | **+0.03 / 0.11** | **0.085 / 0.193** | **0.040 / 0.079** | 11 % (0.13 m) |
| moving sun, g fitted (diagnostic) | 0.06 / 0.08 | -0.02 / 0.09 | 0.093 / 0.192 | 0.077 / 0.073 | 12 % (0.14 m) |

The moving sun gives the same gain as the fitted gravity did, with real gravity. The fitted gravity still comes out
10.1 with the moving sun, so a small scale question remains open (camera), but fixing g at 9.81 is now best overall.

## Shadow pass 4 + track search (2026-11-10) — plateau
- Pass 4 (moving sun, window 25 px, threshold 0.86): 6,084 shadows in flights, 82 % within 8 px of the prediction.
- `track_shadows.py` (flights with < 3 shadows: the flight's depth scale s = 0.80..1.20 is searched by the darkness under
  the predicted shadow path over the whole flight): 173 of 234 flights with a clear path, 611 shadows (3.4 px from the
  hand marks where they overlap).
- Held-out with 6,134 shadows (auto 5,573 + track 472 + hand 89), moving sun, g = 9.81 (`run_p4_sunmove`):
  landings 0.08 / MAE 0.06, chest +0.03 / 0.11, net 11 % — the same as `run_p3_sunmove` within the noise of the
  checks (chest with shadows MAE 0.101 vs 0.085; landings with shadows rms 0.072 vs 0.101). More shadows no longer
  change the answer: with this camera and these marks the heights have converged; the site keeps the moving-sun version.

## Focal length only (2026-11-11)
`CAM_ONLY_FOCAL=1` (everything else of the camera pinned), moving sun, g = 9.81, held-out: focal +55.6 px (+3.6 %), but
the court-line fit breaks (landmark rms 2.64 -> 9.91 px). Chest touches improve (with shadows MAE 0.060), landings move
down (0.05 m, bias -6 cm). The ball data keep asking for a ~3 % larger scale than the blue lines allow (the same
tension as the free gravity 10.09 and the free camera): either the blue lines are not exactly 16 x 8 m (the data point
to ~15.6 x 7.8 m) or the camera model is missing something. Not applied: deciding needs one outside measurement
(the real size of the blue lines, or the camera's position/height).
