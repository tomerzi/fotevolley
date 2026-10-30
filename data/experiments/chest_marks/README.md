# Experiment: hand-marked chest touches (2026-10-28)

`marks/marks/*.json`: the 99 chest touches marked on the marking page (contact frame, feet; the ball position is
the automatic box unless marked by hand). 95 with both feet on the sand. 88 at the labelled frame, 11 moved by 1-2 frames.

1. `analyze.py` -> `result.csv`: the ball's height at each marked touch from geometry only (ball pixel at the
   touching player's horizontal position) is ~1.35-1.43 m whatever the calibration: a robust reference.
2. The ballistic flight fits miss it: site camera +0.36 m, physics + antenna camera -0.24 m.
3. `calib_marked.py`: the physics + antenna calibration plus the marks (ball at the player's horizontal position in
   the flight leaving / reaching the chest). Hold-out (calibrated on even touch ids, scored on odd ones, `score.py`):
   error -0.24 -> -0.10 m, MAE 0.24 -> 0.17 m, ball-player distance 0.53 -> 0.40 m.
4. `calib_all/`: the same with all 94 usable marks. Lines <= 12 cm, antenna <= 0.9 px, sand landings 1.00,
   flights 1.43 px. Ball at chest touches 1.23 m vs expected 0.72 x stature = 1.24 m; shoulder +0.04 m.
   Statures from the video: BRA_A 1.76, BRA_B 1.66, ISR_A 1.72, ISR_B 1.69 m. Ball in flight: median 2.36 m,
   95th percentile 4.06 m, max 6.17 m. Blue lines 16 x 8 m, no player heights used.
