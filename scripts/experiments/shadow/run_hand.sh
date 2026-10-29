#!/bin/bash
# EXPERIMENT (shadow + hand-marked shadows): pass-2 shadows plus the 89 hand-marked shadows (they replace the
# automatic one at the same frame). Held-out check (even/odd) and the final run with all marks.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0; F=data/experiments/improve/final; R=data/experiments/shadow
SH="SHADOW=$R/shadows_with_hand.csv SUN=$R/pass2/sun.json NET=all"
if [ "$1" = final ]; then
  env $SH CHEST_MARKS=all SAND_USE=all SIZE_FIX=all python3 -W ignore scripts/experiments/improve/chain.py $F/camera.npz $F/est > $R/final_hand.log 2>&1 || { tail -5 $R/final_hand.log; exit 1; }
  cp $F/est/chain.csv $R/chain_final_hand.csv
  echo "== final_hand"; grep -E "flights,|status" $R/final_hand.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_final_hand.csv all
  python3 -W ignore scripts/experiments/improve/score_csv.py $F/camera.npz $R/chain_final_hand.csv all
  python3 scripts/experiments/improve/score_net.py $R/chain_final_hand.csv all
else
  env $SH SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $R/hand_shadow_net.log 2>&1 || { tail -5 $R/hand_shadow_net.log; exit 1; }
  cp $O/chain.csv $R/chain_hand_shadow_net.csv
  echo "== hand_shadow_net"; grep -E "flights,|status" $R/hand_shadow_net.log
  python3 scripts/experiments/sand_marks/score_sand.py $R/chain_hand_shadow_net.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $C $R/chain_hand_shadow_net.csv odd
  python3 scripts/experiments/improve/score_net.py $R/chain_hand_shadow_net.csv odd
fi
