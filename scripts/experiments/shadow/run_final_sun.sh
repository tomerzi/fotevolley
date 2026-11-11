#!/bin/bash
# FINAL candidate: pass-3 + hand shadows with the MOVING sun (azimuth and elevation drift fitted), real gravity 9.81,
# net rule, all marks; then frames outside flights (touch depth / shadow) and the per-frame uncertainty.
cd "$(dirname "$0")/../../.."
F=data/experiments/improve/final; R=data/experiments/shadow; O=$R/final_sun; mkdir -p $O; cp $R/final_p3/chain.csv $O/nodrag.csv  # warm start from the converged fixed-sun solution
MAX_NFEV=200 SHADOW=$R/shadows_p3_hand.csv SUN=$R/pass3/sun.json SUN_FIT=1 SUN_ELEV_DRIFT=1 NET=all CHEST_MARKS=all SAND_USE=all SIZE_FIX=all \
  python3 -W ignore scripts/experiments/improve/chain.py $F/camera.npz $O > $O/chain.log 2>&1 || { tail -5 $O/chain.log; exit 1; }
grep -E "status|sun:" $O/chain.log
python3 scripts/experiments/shadow/triangulate.py $O/chain.csv $R/pass3/shadows_fit.csv $O/sun_fit.json $F/camera.npz $O
python3 scripts/experiments/shadow/uncertainty.py $O/chain_tri.csv $R/shadows_p3_hand.csv $O/ball_height.csv
python3 scripts/experiments/sand_marks/score_sand.py $O/ball_height.csv all
python3 -W ignore scripts/experiments/improve/score_csv.py $F/camera.npz $O/ball_height.csv all
python3 scripts/experiments/improve/score_net.py $O/ball_height.csv all
