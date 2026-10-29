#!/bin/bash
# EXPERIMENT (shadow / net): joint flight fit with the ball's shadow as a second view, and with the net rule on all
# crossings. Sand landings and size fix from the even half; scored on the odd half.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0; R=data/experiments/shadow
SH="SHADOW=$R/pass1/shadows_fit.csv SUN=$R/pass1/sun.json"
for v in "shadow:$SH" "shadow_sig5:$SH SHADOW_SIG=5" "net_all:NET=all" "shadow_net:$SH NET=all"; do
  name=${v%%:*}; opts=${v#*:}
  env $opts SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $R/$name.log 2>&1 || { echo "$name failed"; tail -3 $R/$name.log; continue; }
  cp $O/chain.csv $R/chain_$name.csv
  echo "== $name"; grep -E "flights,|status|jump" $R/$name.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_$name.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $C $R/chain_$name.csv odd
  python3 scripts/experiments/improve/score_net.py $R/chain_$name.csv odd
done
