#!/bin/bash
# EXPERIMENT (improve): the best combination with ALL marks: chest-point calibration (all chest marks), flights refit
# jointly with continuity, all sand landings pinned, box sizes corrected from the landings.
cd "$(dirname "$0")/../../.."
O=data/experiments/improve/final
OUT=$O MARK_USE=all CHEST=1 DRAG=0 OCCL=0 python3 -W ignore scripts/experiments/improve/calib.py > $O.calib.log 2>&1 && grep -E "^f=|line max|stripe|sand land|reprojection" $O.calib.log
K_DRAG=0 python3 -W ignore scripts/experiments/improve/estimate_drag.py $O/camera.npz $O/est
SAND_USE=all SIZE_FIX=all python3 -W ignore scripts/experiments/improve/chain.py $O/camera.npz $O/est
python3 scripts/experiments/sand_marks/score_sand.py $O/est/chain.csv all
python3 -W ignore scripts/experiments/improve/score_csv.py $O/camera.npz $O/est/chain.csv all
