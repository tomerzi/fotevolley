# Experiment: flight physics models (after the ball3d benchmark, CVPRW 2026)

Status: experiment only. `data/analysis/` and the site are unchanged.

`scripts/experiments/improve/run_models.sh` runs the joint flight fit (`chain.py`) with different flight models.
Same camera (`improve/chest`), sand landings and box-size fix from the even half; scored on the odd half
(21 sand landings, 40 chest touches, ~65 net crossings) that the fit never saw.

| model | chain.py options | fitted | sand landings (0.11 expected) median / MAE | chest touches vs chest point median / MAE | net crossings below the top |
|---|---|---|---|---|---|
| **gravity only (parabola)** | – | – | 0.18 / **0.11** | **-0.04 / 0.11** | **16 %** |
| + air drag | DRAG=1 | k 0.0139 1/m | 0.23 / 0.12 | -0.06 / 0.12 | 18 % |
| fitted gravity (ball3d winner) | G_FIT=1 | g 10.41 m/s² | 0.00 / 0.11 | -0.28 / 0.32 | 19 % |
| spin (constant extra acceleration per flight, ±1.5) | ACC=1 | median 1.4 m/s² | 0.06 / 0.11 | -0.26 / 0.29 | 27 % |
| spin, tighter (±0.7) | ACC=1 ACC_SIG=0.7 | median 0.5 m/s² | 0.03 / 0.11 | -0.24 / 0.26 | 23 % |
| fitted gravity + spin | G_FIT=1 ACC=1 | g 9.60, 1.5 m/s² | 0.07 / 0.11 | -0.25 / 0.28 | 23 % |
| drag + spin | DRAG=1 ACC=1 | k 0.0026, 1.4 m/s² | 0.07 / 0.11 | -0.27 / 0.28 | 27 % |

Reading: every extra degree of freedom makes it worse on the held-out checks. From one camera the extra
parameters do not find real spin or gravity; they absorb pixel noise and the depth (distance-from-camera)
ambiguity, which moves balls down by ~0.2 m at chest touches and puts more net crossings under the net.
ball3d's "fitted gravity wins" holds for their setting (no metric anchors); here the camera is metric (court lines,
net, antenna) and the marks anchor the depth, so the plain parabola stays best. The limit is the depth from one view,
not the physics model.
