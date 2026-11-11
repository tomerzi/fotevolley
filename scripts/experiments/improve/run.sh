#!/bin/bash
# EXPERIMENT (improve): calibrate on the even-numbered marked chest touches with each switch, score on the odd ones.
cd "$(dirname "$0")/../../.."
for v in "base:0:0:0" "occl:0:1:0" "drag:1:0:0" "chest:0:0:1" "all:1:1:1"; do
  IFS=: read name drag occl chest <<< "$v"
  out=data/experiments/improve/$name
  echo "== $name (drag=$drag occl=$occl chest=$chest)"
  OUT=$out MARK_USE=even DRAG=$drag OCCL=$occl CHEST=$chest python3 -W ignore scripts/experiments/improve/calib.py > $out.calib.log 2>&1 || { echo "calib failed"; tail -3 $out.calib.log; continue; }
  grep -E "drag k|^f=|sand|reprojection" $out.calib.log
  OCCL=$occl python3 -W ignore scripts/experiments/improve/score.py $out/camera.npz $out/score odd
done
