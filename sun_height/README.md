# Ball height from its shadow

Match: EFVL World Challenge Graz 2022 (Freiheitsplatz, Graz), ISR vs BRA, 26 Jun 2022 (assumed).

## Pipeline (run in this order)
1. `lines.py`, `calib.py`: court lines -> ground homography + 2-term radial distortion (`calib.npy`, ~1.5 cm RMS).
2. `measure.py`: every 5 s, net-top shadow offset from the net (2.20 m top) and post-shadow angle (`meas.json`).
   Needs crops: `ffmpeg -i ../main_camera_clean_v2.mp4 -vf "fps=1/5,crop=360:400:760:620" frames_5s/c%04d.png`.
3. `sunfit.py`: fits those to the astronomical sun path (`sun.py`) -> court bearing 191 deg, clock time per clip second (`sunfit.json`).
   `sunclock.sunstate(t)` returns clock, elevation, azimuth and the ground shadow vector per metre of height.
4. `vvp.py`: vertical vanishing point from upright players in `pose/vitpose_body25.csv` (`vvp.npy`).
   Vertical scale `KAPPA` in `cam.py` comes from the near post (2.79 m tall from its shadow).
5. `scan2.py`: per frame, scores transient dark blobs (vs per-shot median background) along the curve of possible shadow
   positions for heights 0-15 m; masks ball, players and player shadows; sunlit sand only (~26 min).
6. `track2.py`: Viterbi over height with smoothness -> `ball_height_raw.csv`; gaps <= 10 frames filled by parabola into `ball_height.csv`.

## Checks
- post-shadow line passes through the same post base in every frame (Y 10.02-10.07 m)
- predicted player head-shadow tips land on the real ones (`overlay.py`)
- sun-path fit chi2/n 0.35; predicted shadow angle within 0.1-0.8 deg of measured
- free-flight parabolas give g = 9.35 m/s^2 (IQR 8.9-9.9): absolute scale may be ~5% low;
  players come out ~1.70 m. Real player heights would pin the scale.

## Output `ball_height.csv`
`height_m` only where the shadow was found (`shadow_found`), `height_filled_m` adds parabola fills, plus clock time (CEST),
sun elevation/azimuth and the shadow's court position. No shadow -> empty (ball over the crowd, in shade, or shadow hidden).
