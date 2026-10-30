# Experiment: second camera (broadcast replay) — one synchronized point

Status: experiment only. Nothing in `data/analysis/` or the site was changed.

## Inputs
- `youtube_main_1800.webp` — YouTube broadcast at 18:00, main camera. Same moment as v2 frame **9638**
  (rally 36, set 2, BRA 4–0, bicycle kick by BRA_B at the net).
- `replay_cam2.png` — broadcast replay of the same kick from a second camera behind the ISR baseline
  (554×313 screenshot). The ball is at the kicker's foot = kick contact (v2 frame 9638–9639).
- `cam2_points.json` — pixel readings: both blue sidelines (sub-pixel ridge points), net top tape upper edge,
  BRA_A (standing) feet/head with his main-camera ground point and measured stature 1.791 m, ball centre.

## Method (`scripts/experiments/two_camera/calib_cam2.py`)
1. Calibrate the replay camera (pinhole, principal point at the image centre, 7 unknowns) from
   sidelines Y=0 / Y=8, net top Z=-2.2 at X=8, and BRA_A. Fit rms 0.85 (normalized).
   Camera: X ≈ 0.0 (on the baseline line, ±1 m), Y ≈ 7.3, height ≈ 2.85 m, f ≈ 690 px.
2. Triangulate: point on the main-camera ray of the ball box that best projects onto the ball in the replay.
3. Monte Carlo (300 runs): pixel noise, BRA_A ground position ±0.15 m, ball size.

## Result
| frame | site (main camera only) X / Y / height | two cameras X / Y / height (MC ±) |
|---|---|---|
| 9637 | 8.72 / 5.30 / 2.84 | 8.74 / 5.09±0.06 / 2.77±0.02 |
| 9638 | 8.64 / 5.38 / 2.65 | 8.66 / 5.19±0.06 / 2.57±0.02 |
| 9639 | 8.21 / 5.63 / 2.77 | 8.24 / 5.28±0.06 / 2.61±0.03 |

At the contact the main camera placed the ball 0.2–0.35 m too far from itself (Y) and therefore
7–15 cm too high. This matches the ~11 cm typical error found with the hand marks.
The replay instant is known only to ±1 frame, which adds about ±0.1 m.

`replay_overlay.jpg`: projected court (cyan), net top (yellow), ball from the site (red) and from two cameras (green).

## Limits / next step
One still = one 3D point: it validates, it cannot correct the whole match.
Replay video segments (many frames per replay) would give hundreds of such points and allow
correcting the main camera's depth directly.

## Update: several replay screenshots (`calib_multi.py`, `match.py`, `shots.json`)
Synchronized (frame chosen where the ball reprojects best in the replay):

| screenshot | rally / v2 frame | moment | site − two cameras (height) |
|---|---|---|---|
| replay_cam2.png | 36 / 9638 | bicycle-kick contact | +0.01 … +0.17 m |
| img5.webp | 18 / 2998–2999 | bicycle-kick contact | −0.10 … +0.06 m |
| img4.webp | 34 / 8802 | header at the net (frame match by player layout, medium confidence) | +0.07 … +0.09 m |
| img6.webp | 18 / 3007 | ball passing ISR_A after the kick | −0.13 … −0.02 m |

Ranges = results for replay-camera positions between 0.7 m in front of and 8 m behind the ISR baseline:
along the camera's viewing axis, its distance trades off against zoom and cannot be determined from
these lines alone (lines parallel to X do not constrain the position along X).
Mean difference ≈ 0 to +0.05 m, spread ≈ 0.1 m → no clear systematic bias; consistent with the ~11 cm
typical error. img7/img8 could not be synchronized reliably (player layout matches no labelled touch) and were not used.
