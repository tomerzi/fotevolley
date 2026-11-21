#!/bin/bash
# EXPERIMENT (shadow, 2nd pass): shadows re-detected around the shadow+net solution with the fitted sun, then refit.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0; R=data/experiments/shadow
for v in "p2_shadow_net:SHADOW=$R/pass2/shadows_fit.csv SUN=$R/pass2/sun.json NET=all SHADOW_MAXERR=15" "p2_shadow:SHADOW=$R/pass2/shadows_fit.csv SUN=$R/pass2/sun.json SHADOW_MAXERR=15"; do
  name=${v%%:*}; opts=${v#*:}
  env $opts SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $R/$name.log 2>&1 || { echo "$name failed"; tail -3 $R/$name.log; continue; }
  cp $O/chain.csv $R/chain_$name.csv
  echo "== $name"; grep -E "flights,|status|jump" $R/$name.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_$name.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $C $R/chain_$name.csv odd
  python3 scripts/experiments/improve/score_net.py $R/chain_$name.csv odd
done
