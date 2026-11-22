#!/bin/bash
# EXPERIMENT (improve): net constraints and chest points from ViTPose / Sapiens2 in the joint flight fit.
# Constraints from the even half, scored on the odd half (net crossings, sand landings, chest marks).
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; O=data/experiments/improve/dfix_k0
for v in "net:even:none:vitpose" "chest_vit:none:even:vitpose" "chest_sap:none:even:sapiens" "net_chest_sap:even:even:sapiens"; do
  IFS=: read name net cm pose <<< "$v"
  NET=$net CHEST_MARKS=$cm POSE=$pose SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $O/$name.log 2>&1 || { echo "$name failed"; tail -3 $O/$name.log; continue; }
  cp $O/chain.csv $O/chain_$name.csv
  echo "== $name"; grep -E "flights,|jump" $O/$name.log
  python3 scripts/experiments/improve/score_net.py $O/chain_$name.csv odd
  python3 scripts/experiments/sand_marks/score_sand.py $O/chain_$name.csv odd
  python3 -W ignore scripts/experiments/improve/score_csv.py $C $O/chain_$name.csv odd
done
