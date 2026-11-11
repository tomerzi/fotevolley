#!/bin/bash
# EXPERIMENT (physics models, after the ball3d benchmark): flight models in the joint fit, held-out scoring.
# Sand landings and box-size fix from the even half; scored on the odd half (landings, chest touches, net crossings).
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0; R=data/experiments/physics_models
for v in "parabola::" "drag:DRAG=1:" "fitg:G_FIT=1:" "spin:ACC=1:" "spin_tight:ACC=1 ACC_SIG=0.7:" "fitg_spin:G_FIT=1 ACC=1:" "drag_spin:DRAG=1 ACC=1:"; do
  IFS=: read name opts _ <<< "$v"
  env $opts SAND_USE=even SIZE_FIX=even MAX_NFEV=${MAX_NFEV:-80} python3 -W ignore scripts/experiments/improve/chain.py $C $O > $R/$name.log 2>&1 || { echo "$name failed"; tail -3 $R/$name.log; continue; }
  cp $O/chain.csv $R/chain_$name.csv; cp $O/chain_settings.json $R/settings_$name.json
  echo "== $name"; grep -E "status|drag k|extra acc|jump" $R/$name.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_$name.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $C $R/chain_$name.csv odd
  python3 scripts/experiments/improve/score_net.py $R/chain_$name.csv odd
done
