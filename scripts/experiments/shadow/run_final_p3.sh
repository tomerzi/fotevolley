#!/bin/bash
# FINAL candidate: pass-3 shadows + hand-marked shadows, net rule, all marks, no camera refinement; then
# frames outside flights (touch depth / shadow triangulation) and the per-frame uncertainty.
cd "$(dirname "$0")/../../.."
F=data/experiments/improve/final; R=data/experiments/shadow; O=$R/final_p3; mkdir -p $O; cp $F/est/nodrag.csv $O/
SHADOW=$R/shadows_p3_hand.csv SUN=$R/pass3/sun.json NET=all CHEST_MARKS=all SAND_USE=all SIZE_FIX=all \
  python3 -W ignore scripts/experiments/improve/chain.py $F/camera.npz $O > $O/chain.log 2>&1 || { tail -5 $O/chain.log; exit 1; }
python3 scripts/experiments/shadow/triangulate.py $O/chain.csv $R/pass3/shadows_fit.csv $R/pass3/sun.json $F/camera.npz $O
python3 scripts/experiments/shadow/uncertainty.py $O/chain_tri.csv $R/shadows_p3_hand.csv $O/ball_height.csv
python3 scripts/experiments/sand_marks/score_sand.py $O/ball_height.csv all
python3 -W ignore scripts/experiments/improve/score_csv.py $F/camera.npz $O/ball_height.csv all
python3 scripts/experiments/improve/score_net.py $O/ball_height.csv all
