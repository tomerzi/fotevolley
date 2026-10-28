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
