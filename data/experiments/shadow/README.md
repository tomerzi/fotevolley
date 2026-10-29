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
