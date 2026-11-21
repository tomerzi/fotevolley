# Experiment: size of the blue lines (2026-10-27)

Not used by the main outputs. Made by `scripts/experiments/court_scale/run.py`; one row per assumed line size in
`scan.csv` (the per-run folders `s*/` are not tracked).

For each size the camera is calibrated to the blue lines, the net top (2.20 m at the near antenna) and standing
Israeli players (1.86 m). The ball is then re-fitted with gravity 9.81 m/s^2 and anchors at the labelled touches
(ball at the touching body part) and sand landings.

| lines (m) | g fitted freely | ball vs touching body part (height / distance) | chest touches | head touches | net (px) | principal point y (px) |
|---|---|---|---|---|---|---|
| 16 x 8 | 9.65 | +0.43 / 0.72 m | 1.66 m | 2.52 m | 8 | 262 |
| 17 x 8.5 | 10.24 | +0.54 / 0.88 m | 1.77 m | 2.64 m | 18 | 250 |
| 18 x 9 | 11.83 | +0.20 / 0.53 m | 1.42 m | 2.18 m | 21 | 533 |
| 19 x 9.5 | 10.81 | +0.24 / 0.57 m | 1.45 m | 2.22 m | 18 | 537 |

Expected: chest ~1.3 m, head ~1.8 m plus any jump, principal point near 540 (image centre).

Reading:
- No size makes every check agree.
- The free-gravity value does not change steadily with the size (9.65, 10.24, 11.83, 10.81), so a single camera
  does not pin the depth well enough for it to decide the question.
- The touches and the camera geometry favour 18 x 9 or larger: the ball comes closest to the touching body part,
  and only there does the principal point land at the image centre (with 16 x 8 it is pushed ~280 px up).
- The net check favours 16 x 8, but it rests on one hand-picked pixel (1008, 622).
- Even at the best size the ball stays ~0.2 m above the body part at touches, so a second error source remains
  (the ball partly hidden by the body at the touch, the flight fit near its ends).
