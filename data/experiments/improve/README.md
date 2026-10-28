# Experiment: improvements 1-4 (2026-10-28)

Not used by the main outputs. Each variant is calibrated on the even-numbered hand-marked chest touches
(`scripts/experiments/improve/calib.py`) and scored on the odd ones (`score.py`); run with `run.sh`.
Score = ball height from the flight fits at the marked frame minus the marked height, measured either above the
player's feet or at the chest point from the pose (between hip and neck, at 0.72 x stature).

| variant | vs above the feet: median / MAE | vs chest point: median / MAE | ball-player distance | n |
|---|---|---|---|---|
| base (marks only) | -0.10 / 0.17 m | -0.22 / 0.26 m | 0.40 m | 40 |
| OCCL: no size term when the ball overlaps a player or is near a touch | -0.07 / 0.16 m | -0.24 / 0.24 m | 0.34 m | 40 |
| DRAG: quadratic drag, k fitted = 0.022 1/m | -0.34 / 0.37 m | -0.49 / 0.49 m | 0.94 m | 17 |
| CHEST: marked ball at the chest point from the pose | +0.09 / 0.19 m | **-0.07 / 0.14 m** | 0.46 m | 40 |
| CHEST + OCCL | +0.07 / 0.19 m | -0.09 / 0.17 m | 0.49 m | 40 |
| all three | -0.11 / 0.22 m | -0.27 / 0.29 m | 0.48 m | 36 |

Step 4 (camera drift, `drift.py` -> `drift.csv`): template matching of static scenery every 100 frames: the camera
moves at most ~1-2 px over the whole match, so no correction is needed.

Reading:
- The chest point from the pose is the fairer reference (in a chest trap the player leans back, the chest is not
  above the feet); against it CHEST is the best variant (MAE 0.26 -> 0.14 m).
- OCCL helps a little on its own.
- DRAG as implemented is broken, not disproved: the fitted k is plausible, but the per-frame estimator then fits
  far fewer flights (8,781 vs 11,210 frames), so most touches fall back to the size estimate. To fix before judging.

## Drag, fixed (same day)

`estimate_drag.py` keeps the no-drag flights and refits each one with drag from its no-drag solution (RK4, 8 steps
per frame), so both versions cover the same 825 flights. With k = 0 it reproduces the CHEST result exactly.
Scored on the odd marks, CHEST camera, against the chest point (median / MAE):

| drag k (1/m) | vs chest point | ball-player distance |
|---|---|---|
| 0 | -0.07 / 0.14 m | 0.46 m |
| 0.01 | -0.08 / 0.15 m | 0.46 m |
| 0.02 | -0.08 / 0.16 m | 0.41 m |
| 0.03 | -0.10 / 0.19 m | 0.46 m |
| 0.013, fitted with the camera (DRAG + CHEST) | -0.06 / 0.14 m | 0.51 m |

Drag is physically plausible (k ~ 0.013-0.022) but does not change the chest touches: those flights are short.

## Sand landings as a check

Heights at the 37 sand landings of data/contacts/ball_contacts_v3.csv (should be 0.11 m): 0.38 m (no drag),
0.45-0.49 m with drag. Anchoring half of them (`ANCHORS=sand SAND_USE=even` in chest_marks/estimate_marked.py)
brings those to 0.12 m but the held-out half only from 0.45 to 0.39 m. The lowest fitted height within +-3 frames
of a landing is often 1 frame away and some events are wrong (6 m up), so the landing labels themselves are noisy:
at landing speed (~8 m/s) one frame is ~0.3 m. Hand-checked landings would make them a usable reference.

Best so far: CHEST (with or without the fitted drag): chest touches -0.06 / 0.14 m on held-out marks.

## Sand-landing marks, continuity, box-size fix (2026-10-28 evening)

`data/experiments/sand_marks/marks/`: 49 landings checked by hand: 41 landings (37 at the automatic frame, 4 one
frame later), 2 not a landing, 6 unclear.

1. Heights at the marked landings (should be 0.11 m): site 0.45 m, CHEST camera 0.23 m.
2. Pinning landings only inside their own flight barely helps the others (held-out 0.35 -> 0.33 m).
3. `chain.py`: all flights fitted together, the ball may not jump between consecutive flights (553 links).
   Jump at the joins 0.74 -> 0.13 m; chest touches vs chest point MAE 0.14 -> 0.11 m. Landings unchanged.
4. The misses are the far landings (boxes 18-22 px): the detector's boxes are squeezed towards ~25 px, so small
   (far) balls look too big, too close, too high. `SIZE_FIX` maps box size to ball size with a line fitted on the
   landings (even half: ball = 1.30 x box - 8.3 px). Held-out landings 0.37 -> 0.18 m (MAE 0.26 -> 0.11 m); chest
   touches -0.04 / 0.11 m.
5. `final/` (`run_final.sh`): everything with all marks (ball = 1.198 x box - 5.7 px). Landings 0.13 m (MAE 0.03,
   in-sample); chest touches -0.07 / 0.13 m vs chest point; ball in flight median 2.55 m, 95th percentile 4.21 m,
   max 6.29 m; statures BRA_A 1.79, BRA_B 1.69, ISR_A 1.74, ISR_B 1.72 m.
   Note: at chest touches the ball centre sits at ~0.80 x stature (marked geometry, 1.37 m), not 0.72 x stature:
   it is in front of the chest and one radius out, so "0.72 x stature" checks read ~0.2 m high.
