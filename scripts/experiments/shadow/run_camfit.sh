#!/bin/bash
# EXPERIMENT (option 1): camera + sun refined together with all flights (CAM_FIT, SUN_FIT), shadows pass 2 + hand marks,
# net rule. Held-out (even/odd); scored with the refined camera.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0; R=data/experiments/shadow
SHD=${SHD:-$R/shadows_with_hand.csv}; NAME=${NAME:-camfit}
for v in "$NAME:CAM_FIT=1 SUN_FIT=1" "${NAME}_sun:SUN_FIT=1"; do
  name=${v%%:*}; opts=${v#*:}
  env $opts SHADOW=$SHD SUN=$R/pass2/sun.json NET=all SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $R/$name.log 2>&1 || { echo "$name failed"; tail -5 $R/$name.log; continue; }
  cp $O/chain.csv $R/chain_$name.csv; CS=$C; [ -f $O/camera_fit.npz ] && [ "${opts#*CAM_FIT}" != "$opts" ] && cp $O/camera_fit.npz $R/camera_$name.npz && CS=$R/camera_$name.npz
  echo "== $name"; grep -E "status|camera change|sun:" $R/$name.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_$name.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $CS $R/chain_$name.csv odd
  python3 scripts/experiments/improve/score_net.py $R/chain_$name.csv odd
done
