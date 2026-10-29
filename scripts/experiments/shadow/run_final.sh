#!/bin/bash
# FINAL (promoted to data/analysis): all marks + the ball's shadow (pass 2) + the net rule on all crossings, no drag.
cd "$(dirname "$0")/../../.."
F=data/experiments/improve/final; R=data/experiments/shadow
SHADOW=$R/pass2/shadows_fit.csv SUN=$R/pass2/sun.json SHADOW_MAXERR=15 NET=all CHEST_MARKS=all SAND_USE=all SIZE_FIX=all \
  python3 -W ignore scripts/experiments/improve/chain.py $F/camera.npz $F/est > $R/final.log 2>&1 || { tail -5 $R/final.log; exit 1; }
cp $F/est/chain.csv $R/chain_final.csv; cp $F/est/chain_settings.json $R/final_settings.json
grep -E "flights,|status|jump" $R/final.log
python3 scripts/experiments/sand_marks/score_sand.py $R/chain_final.csv all
python3 -W ignore scripts/experiments/improve/score_csv.py $F/camera.npz $R/chain_final.csv all
python3 scripts/experiments/improve/score_net.py $R/chain_final.csv all
