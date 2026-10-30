# Next: finish + verify boxes/pose on GPU (written 2026-10-09)

State: fixes from review/fix/ are applied to player_boxes.csv and pose/vitpose_body25.csv (uncommitted).
CPU is too slow (Faster R-CNN 5.5 s/frame -> 24 h full video; ViTPose ~1 s/box -> 17 h). Run on the T4.

## 1. Leftover ranges (looked at by eye on CPU)
- 1755-1780: two BRA cross; 1765-1772 one hides the other -> one box is OK. Check BRA_A/BRA_B names stay right after 1773.
- 12572-12586: only one BRA on screen (diving) -> one box OK. Decide its name (BRA_A or BRA_B) from 12566-12571 / 12587-12592 by shirt/position.
- Ball 747-754: in shadow at a player's feet, both models miss it -> try `python ball/fill_ball.py --track` or label by hand.

## 2. Full-video verification (GPU)
1. GPU part runs on Kaggle (studio can't switch to GPU): review/kaggle/detect_all.py on the video
   (Kaggle notebook, Accelerator GPU T4, Internet ON) -> download people_full.csv -> put it at review/full/people_full.csv.
   Everything after this runs on the studio CPU.
2. Compare with player_boxes.csv per frame:
   - on-court detection (score>0.7, feet inside COURT) with no player box -> "missing player"
   - player box with no detection IoU>=0.3 -> "box on nothing" (except edge / occluded)
   - two player boxes IoU>0.5 -> "two boxes one person"
   - team-name jumps (centre jump > GATE px between frames within a shot)
3. Re-run pose/check_missing.py; pose inside its box; joints count.
4. Show the new suspects in an HTML page like review/wrong_frames.html, fix with review/fix/repair_boxes.py, re-pose changed boxes with review/fix/repose.py (cuda).
5. Then: add pose/weights/ to .gitignore, commit, push (needs `gh auth login`).
