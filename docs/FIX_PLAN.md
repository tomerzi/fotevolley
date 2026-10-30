# Player-box fix plan (written 2026-10-06, to run next session)

Found by `pose/check_missing.py` -> `pose/suspect_ranges.csv` and checked by eye on rendered frames.

## Problems

| frames | player | problem | box source now |
|---|---|---|---|
| 305–375 | BRA_B | visible (standing next to BRA_A) but no box | none (only 10 boxes in 300–380) |
| 5311 | BRA_B, ISR_B | BRA_B visible mid-court, ISR_B partly visible at right edge; no boxes | none |
| 6734 | BRA_A | box on empty sand near the net | `redetect` |
| 8251–8289 | BRA_A | box on empty sand (player actually off-screen) | `interpolated` |
| 9651 | BRA_B | box on empty sand near the net | `interpolated` |

## Likely causes (verify first)
- **Missing BRA_B 305–375.** `players/redetect.csv` has ~5 detections per frame there, so the detector sees BRA_B. `fill_gaps.py` does not assign them, probably because:
  - the seed overlaps BRA_A too much (`SEED_MAX_IOU = 0.2`), or
  - the OSNet re-id (`reid.name`) gives the crop to BRA_A.

  Check which condition rejects them.
- **Phantom `interpolated` boxes.** Interpolation bridges a gap where the player actually left the frame, or the end box is wrong. Interpolated boxes need to be validated.

## Steps
1. **Debug the missing BRA_B.** In `players/fill_gaps.py`, print why the seed candidates for BRA_B in frames 305–375 and 5311 are rejected. Relax the right rule only for the case where the other team member is already boxed and the candidate is the remaining unassigned same-team person. Don't relax it globally.
2. **Validate interpolated and redetect boxes.** Keep an `interpolated` or single-frame `redetect` box only if `players/redetect.csv` has a person detection overlapping it (IoU ≥ 0.3) in that frame, or the gap is short (≤ 5 frames). Drop it otherwise. This should remove 6734, 8251–8289 and 9651. Count how many boxes this drops overall and spot-check a sample, so real occluded players aren't lost.
3. **Re-run the box pipeline.** Run `python players/fill_gaps.py`, then compare the new `players/player_boxes.csv` against the old one: count of added and removed boxes per player.
4. **Re-run pose only on frames whose boxes changed.** Add a frame-list option to `pose/vitpose.py`, e.g. env `POSE_FRAMES=file.csv`, writing to a temp CSV. Then replace those (frame, player) rows in `pose/vitpose_body25.csv` and drop rows for removed boxes. The full re-run takes ~70 min on the T4, so avoid it.
5. **Verify.**
   - Run `python pose/check_missing.py`. The ranges above should be gone from `pose/suspect_ranges.csv`.
   - Run `python pose/suspect_sheet.py pose_weak <out>` and `python pose/suspect_sheet.py player_none <out>` and look at the sheets.
   - Re-render with `python pose/render_pose.py --suspects pose/suspects_only.mp4` and the full `pose/pose_ball_full.mp4`. Re-encode to H.264 with the imageio-ffmpeg binary; system ffmpeg is missing.
6. **Push to GitHub.** Copy the changed files into the repo clone `ball/repo` and push. This needs the user logged in (`gh auth login`); the studio has no GitHub credentials.

## Ball items left open (lower priority)
- **Visible but undetected:** 747–754 (ball in shadow at a player's feet) and 5213–5217. Both models miss it even on zoomed crops. The only options are template tracking (`python ball/fill_ball.py --track`, ~65% precise) or manual labels.
- **To confirm by eye** in `pose/suspects_only.mp4`: 378–397, 6662–6674, 10061–10073, 10861–10875, 13134–13140, 14313–14318.
