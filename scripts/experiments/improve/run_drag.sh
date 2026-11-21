#!/bin/bash
# EXPERIMENT (improve, drag fixed): same flights as without drag, drag strength swept and fitted; scored on the odd marks.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz
for k in 0.01 0.02 0.03; do
  O=data/experiments/improve/dfix_k$k
  K_DRAG=$k python3 -W ignore scripts/experiments/improve/estimate_drag.py $C $O && echo "== chest camera, k=$k" && python3 -W ignore scripts/experiments/improve/score_csv.py $C $O/ball_height_v2.csv odd
done
# camera and k fitted together (chest point, drag)
O=data/experiments/improve/drag_chest
OUT=$O MARK_USE=even DRAG=1 OCCL=0 CHEST=1 python3 -W ignore scripts/experiments/improve/calib.py > $O.calib.log 2>&1
grep -E "drag k|^f=" $O.calib.log
python3 -W ignore scripts/experiments/improve/estimate_drag.py $O/camera.npz $O/est && echo "== drag+chest camera, fitted k" && python3 -W ignore scripts/experiments/improve/score_csv.py $O/camera.npz $O/est/ball_height_v2.csv odd
