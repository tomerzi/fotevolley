# Ball contacts (Sapiens2 + v3 ball): progress notes

Run: `python scripts/contacts/detect_contacts.py` (~2 min, CPU, 4 cores)
Check by eye: `python scripts/contacts/review_sheet.py OUT_PREFIX --filter "in_play and confidence < 0.9"` (or `--frames 459,2860`)

## Outputs (data/contacts/)
- `ball_contacts_v3.csv`: one row per ball event: frame, rally, in_play, player, team, body_part, limb_side, confidence, runner-up player/part, notes (2D overlap, depth, push)
- `contact_frames_v3.csv`: one row per video frame (13,495): who touches the ball with which part (empty = nobody)
- `contact_candidates_v3.csv`: every (event, player) pair with all costs, for debugging

## Current result (2026-10-11)
519 ball events, 365 touches (330 in play), 154 with no player (sand, net, tracking).
In play: head 107, chest 97, foot 94, shoulder 9, shin 8, thigh 7, arm 4, torso 3, knee 1.

## Checked against the video so far
- Rallies 4-8 in full, every risky touch (2D overlap, confidence < 0.9, unusual parts), and in-play
  "no player" events near a player (2 of 4 sheets).
- Who touched: almost always right, including near/far overlaps (depth + rally rules settle them).
- Fixed after review: arm/hand chosen over head/chest (arm penalty 0.10 m); shin over foot (larger foot capsule,
  shin penalty); hand touch = catching = dead ball; pre-serve ball bouncing = dead; rallies starting after a camera cut.

## Done 2026-10-11
- Net crossing (Viterbi): crossed = ball's side at the touch != last toucher's court side; the path test (accumulated
  since the last real touch, through "nobody" events) only when the ball is at the net. Recovers 2860 ISR_A, 11846 BRA_A
  foot, and 9235 ISR_B hand (catch, dead ball).
- Ball sent back up (v_in down, v_out up) with a player on its side within 0.30 m: "nobody" costs +1.5 (LIFT_NONE).
  Recovers 2278 ISR_A foot, 2606 and 6336 ISR_B. Velocities alone cannot tell this from a sand bounce (cf. 383), and
  the sand test says "on sand" for these, so the nearby player is what decides it.
- Ball at face height: a shoulder less than 0.10 m nearer than the head becomes head (side-of-head header);
  relabel only, the pose frame/player choice is unchanged. 2113, 2606, 6336, 7382 -> head (checked on video).
  (Tried a gap penalty on shoulders above the neck instead: too broad, changed the player at 10061. Reverted.)
- manual_overrides.csv: 2860 ISR_A right foot (Sapiens2 loses the foot on that bicycle kick, auto said thigh).
- Reviewed: in-play "nobody" events with a player within 0.35 m (13), random sample of 8 confident touches (all right
  player; 841 chest/head and 5716 shin/foot borderline).

- Catch: the ball stops in the air (speed out < 4 px/frame, not on sand) on an arm/hand -> dead; later events of the
  same rally row get their own serve search and rally rules (segment `seg`). Rally 34 holds a replayed point (score
  unchanged at 7630-7779): 7623 BRA_A catch, 7678/7685 pre-serve juggling now dead; the real point starts 7713.
- Unclear "nobody" events checked frame by frame: 0 and 1627 are the first frame after a camera cut (ball just falling;
  the touch is the next event), 540 is the ball landing on the sand and rolling to a foot -> all correctly nobody.
  1619 = nobody (BRA_B header 1601, net 1606, sand; user watched it) and 7618 = nobody,
  ball lands on the sand (user checked) -> manual_overrides.csv (player "-" = nobody).

## Next
0. (done) 1619 = nobody: BRA_B header 1601 -> net 1606 -> sand ~1619 (user watched the video).
1. Ball reaching the sand right next to a player (7617) does not end the rally, because "a player is near" blocks
   the stop in in_play(); a touch that comes after a ground event with the ball barely rising may need a check.
2. Shin/foot and chest/head borderline labels: maybe a sample focused on them.
3. Commit when happy (nothing committed yet).

## Touch Point labels done (2026-10-11)
All 240 remaining touches labeled on the Touch Point page (db "labels") -> data/contacts/friend_part_labels.csv
(one labeler; columns: label part/side/special + detector auto_player/auto_part/auto_side/auto_sure).
- 20 no_touch (incl. pre-serve ball adjustments 176/196/214), 5419 and 5744 first cant_tell, then right foot after the video check; 2288 = arm, out of play.
- Body part agrees with detector on 189/218 (87%): sure 148/152, not sure 41/66. Side agrees 60/65.
- Main misses: chest->shoulder (5), shin->foot (4), thigh->foot/knee/arm.
- Not yet merged into manual_overrides.csv (same as video_reviews.csv).
- Last 3 open touches resolved from clips: 570 chest, 3659 left foot, 4341 chest (video_reviews.csv). Every touch now has a label.
- video_reviews.csv got a correct_side column: 1109 R shoulder, 1198 L foot (was thigh), 1247 L foot (was knee), 3264 R shoulder, 3513 R foot, plus sides from comments (85 R, 1773 L, 2694 R, 3659 L). Every sided touch now has a side.

## Final labels (2026-10-11)
data/contacts/touch_labels_final.csv = the one ground-truth file, one row per touch (363):
touch, frame, time_sec, rally, in_play, player, body_part, side, kind
(kind: player / no_touch / pre_serve / sand / net). Side only for shoulder/arm/hand/thigh/knee/shin/foot.
Built from video_reviews.csv + friend_part_labels.csv, now moved to archive/labels_raw/.
Next: merge into manual_overrides.csv, then retune body-part picking.
