# Owner checks, 2026-11-24

Contact sheets made from `video/main_camera_clean_v3.mp4` (v3 frames) and shown to the owner.

## Ron Ben Ishay identity (`ron_identity_frames.csv`)
32 frames with the box the data calls `Ron_Ben_Ishay` drawn on it: 24 spread evenly over the frames where Ron's box
overlaps no other player, and 8 picked at random among the first clean frames 10-60 frames after Ron's and Maor's
boxes overlap (where a tracker switch would show). Owner: **all 32 are Ron**. Together with the 6 blind identity frames
of 2026-11-22 this supports the name swap and no Ron/Maor identity switch along the match.

## Touch labels (`touch_check.csv`)
Each touch shown as three frames (3 before, the touch frame with the ball circled, 3 after), the labelled player boxed.
- `sample`: 40 in-play touches: 24 stratified (2 per player x chest / foot / head) and 16 drawn from the risky ones.
- `uncertain`: the other 57 touches where the label differs from the detector (part or side), a second player was
  within 0.5 cost, the detector's confidence was below 0.6, or the action was flagged.

Owner: every player label right; two corrections, both applied in `data/contacts/touch_labels_final.csv` and recorded
in `data/contacts/owner_decisions.csv`: touch 40 (Hiltinho, frame 1527) right foot, not left; touch 246 (Ron, frame
8966) right foot (body part was empty). Note that touch 40's wrong side agreed with the detector, so left / right
labels outside these 97 touches are not proven by this check.

## Player positions where the tracking was unsure (`position_squares.csv`)
134 on-court box frames had no smoothed position (box cut by the image border, player hidden behind the teammate,
short tracks). They were shown in 32 groups (Q1-Q32) with the 3x3 squares drawn on the court. The owner named the
square for 13 groups (58 frames); `scripts/analysis/player_positions.py` moves those points to the nearest point
inside the named square (src `owner_square`). The other groups keep the automatic point (box bottom); Q12 (Maor_Haas,
frame 4638) stays without a position.
