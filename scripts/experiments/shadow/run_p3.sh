#!/bin/bash
# EXPERIMENT: pass-3 shadows + hand marks; (a) no camera refinement, (b) small camera refinement (CAM_PRIOR) + sun.
# Held-out (even/odd), with per-frame uncertainty (UNC=1). Each variant in its own output folder.
cd "$(dirname "$0")/../../.."
C=data/experiments/improve/chest/camera.npz; R=data/experiments/shadow; SH="SHADOW=$R/shadows_p3_hand.csv SUN=$R/pass3/sun.json NET=all UNC=1"
v=$1; case $v in p3) opts="";; p3_camtight) opts="CAM_FIT=1 CAM_PRIOR=1 SUN_FIT=1";; esac
O=$R/run_$v; mkdir -p $O; cp data/experiments/improve/dfix_k0/nodrag.csv $O/
env $SH $opts SAND_USE=even SIZE_FIX=even python3 -W ignore scripts/experiments/improve/chain.py $C $O > $O/chain.log 2>&1 || { tail -5 $O/chain.log; exit 1; }
CS=$C; [ -f $O/camera_fit.npz ] && CS=$O/camera_fit.npz
echo "== $v"; grep -E "status|camera change|sun:|sigma" $O/chain.log
python3 scripts/experiments/sand_marks/score_sand.py $O/chain.csv odd
python3 -W ignore scripts/experiments/improve/score_csv.py $CS $O/chain.csv odd
python3 scripts/experiments/improve/score_net.py $O/chain.csv odd
