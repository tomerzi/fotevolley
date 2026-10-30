# Experiment: camera from physics only (2026-10-27)

Not used by the main outputs. `scripts/experiments/physics_calib/calib.py` fits the camera and 72 ball flights
together using only known quantities: blue lines 16 x 8 m, net top 2.20 m along its whole length (white top tape,
x ~ 1011 px, plus the near antenna foot), gravity 9.81 m/s^2, ball 0.22 m, 46 sand landings (ball centre 0.11 m).
No player heights. `check.py` then re-estimates the ball and tests it on what the fit never saw.

Calibration: lines <= 11 cm, net 2-4 cm along the tape and 0.5 px at the antenna, principal point (1026, 519)
(near the centre), sand landings agree with the ball size within 2 %, flights 1.28 px. Camera 5.64 m above the sand.

Independent tests:
| | 16 x 8 + Israeli heights (site) | physics only (this) | 18 x 9 + net + Israeli heights |
|---|---|---|---|
| ball to touching player, horizontally (median) | 0.95 m | 0.73 m | 0.47 m |
| ball nearer the camera than the player by | +0.87 m | -0.64 m | +0.23 m |
| player statures from the video | ISR 1.86 (input), BRA 1.84 / 1.76 | BRA_A 1.69, BRA_B 1.60, ISR_A 1.67, ISR_B 1.63 | ISR 1.86 (input), BRA 1.80 / 1.72 |
| ball at chest touches | 1.74 m | 0.96 m (0.58 x stature) | 1.42 m |

At a touch the ball should be ~0.2-0.3 m from the player and at ~0.72 x stature for the chest. The physics-only
camera puts the ball behind the player and the players at 1.60-1.69 m; the version with the players' heights puts
it in front. None is right yet; the 18 x 9 variant is the closest on these tests.

## Update: antenna stripes (same day)

External rules (EFVL / footvolley): antennas 1.80 m, top 0.80 m above the net in 10 cm red / white stripes;
posts 2.55 m, 0.5-1 m outside the sidelines; net 1 m wide, 2.20 m at the centre, ends at most +2 cm.
The near antenna's stripe edges (y 486 ... 617 px at x ~ 1018) give 8 points of known height (3.00 ... 2.30 m).
They show the net top above the near sideline at y ~ 636 px (hidden by the post cap), not 622 as used before.
The post cap edge (1017, 628) does not fit 2.55 m (57 px off) and is only reported.

With the antenna instead of the single net point (no player heights):
lines <= 10 cm, antenna stripes <= 1.5 px, net tape 3-5 cm, principal point (1018, 498), sand landings 0.98, flights 1.28 px,
camera 6.04 m high.
Independent tests: statures BRA_A 1.73, BRA_B 1.65, ISR_A 1.71, ISR_B 1.66 m; chest touches 1.08 m (expected
0.72 x stature = 1.23, error -0.11 m), shoulder -0.02 m; ball 0.56 m from the touching player, 0.40 m behind him.
Best result so far on every test, with the blue lines at 16 x 8.
