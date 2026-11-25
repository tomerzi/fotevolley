# Coach metrics (2026-11-22)

Status of each metric the coach asked for. ✅ hand-checked data, 🟡 automatic estimate, 👁 needs a human check.
All in v3 frames unless noted; lengths in metres on the 16 × 8 court.

| Metric | File | Status |
|---|---|---|
| Coverage / heat maps by phase (serve, reception, build, after own attack, defence) | `data/analysis/phase_heatmaps.*` (`scripts/analysis/phase_heatmaps.py`, png by `render_phase_heatmaps.py`) | 🟡 measured frames only (89%): pose ground point, not occluded, not border-cut, not bridged |
| Partner overlap per phase | `phase_heatmaps_pairs.csv` | 🟡 |
| Touch success, reception quality, winning / losing touch, error chain | `touch_outcomes.csv`, `touch_success_summary.csv`, `point_touches.csv` (`touch_outcomes.py`) | ✅ player + body part (labels), ✅ point end (owner), 🟡 action (inferred) |
| Set quality (air time, ball apex, partner movement) | `set_quality.csv` | air time ✅, apex 🟡 (`height_sigma_m`), movement 🟡 |
| Joint angles, facing, feet / stance at every touch | `touch_body.csv`, `touch_body_summary.csv` (`touch_body.py`) | 🟡 image-plane angles with `*_ok` flags (foreshortening, scores); 👁 stance |
| Return to base (base = mean position) | `return_to_base*.csv` (`return_to_base.py`) | 🟡 |
| Facing the camera / back to it (orientation only where `orientation_ok`) | `body_orientation_facing.csv` (`orientation_facing.py`, v2 frames) | 🟡 👁 |

Definitions (coach): a touch succeeds unless it lost the ball (last touch of the point by the team that lost it);
base = the player's mean position in his own view.

## ISR names corrected (2026-11-22)

The owner's labelling page (`data/experiments/feet_isr_labels/`) showed that Maor_Haas and Ron_Ben_Ishay were swapped
in every file derived from the player boxes: 5 of 5 blind identity frames, 5 of 5 marked touches and 23 of 23
disputed ISR points. Ron Ben Ishay wears the white armband (box code ISR_A). The two names were swapped in 181 text
files (data, scripts, docs, reports, site pages); not swapped: the owner's own marks (`landing_marks_owner.json`,
`point_end_verified.csv`), `docs/time_log.md` and the labelling folder. Images with names drawn on them (contact
sheets, site stills) still show the old names until they are re-rendered. The owner review now agrees with the touch
table on 48 of 56 points.

## Owner decisions (doubts tool, 2026-11-24)

`data/contacts/owner_decisions.csv`, applied by `touch_outcomes.py` (answers in `data/experiments/doubts/answers.csv`):
- Point 2: Hiltinho's touch 4 is a dead ball; Franklin (reception, foot) touched last. Point 13: Maor_Haas (touch
  table right, the earlier review wrong). Point 56: Ron_Ben_Ishay touched last after Franklin's attack, a touch the
  table missed (added, body part unknown). Point 27 (checked on a half-speed clip): Maor_Haas touched last and the
  ball went out; the ball is hidden at v3 5092-5124, so one touch there is missing from the table. Point 28:
  Franklin's attack touched last, the Israeli defender did not touch it.
- Rally 39 (settled on the rally sheet, 2026-11-24): Hiltinho's 244 is a set and Franklin attacks at 245; Ron's
  untagged touch 246 (right foot) returns it; Franklin's 247 is the reception (247-249 reception / set / attack).
  The earlier "244 and 247 are attacks" is withdrawn.
- Rally 50: reception / set / attack confirmed.
- The last toucher in the table now matches the owner on 53 of 56 shown points; the other 3 are net faults with no
  last touch in the review (points 19, 34, 50).

## Stance and facing: what to trust

- Stance (stood set / arrived moving / jumped) is **not** measurable reliably from this one camera: on the owner's
  69 labelled touches the best rule (jump test + body speed at the touch) scores 52%, and the jump test finds about
  half the jumps. Use `stance_label` (the owner's 69 labels) for any statement about stance.
- Facing: shoulder-width threshold raised to 0.4 (`SIDE_W`), 65% on the 60 labelled frames; `facing_label` holds
  the owner's labels.
- Foot-by-foot planted / sliding / air from the pose: about 45%, not used.
- Position from heel + toes instead of ankles was tested (`POS_GROUND=sole` in `player_positions.py`): noisier in
  the sand (outliers 3.1% vs 2.2%), shift 6 cm median, so the ankle stays the default.

## Labelling game (2026-11-24, evening)

`data/experiments/labelling_game/` (answers, page link), applied as "labelling game" rows in
`data/contacts/owner_decisions.csv`; metrics recomputed in `coach_analysis/recomputed_2026-11-24/`; data/analysis rerun the same evening
(all scripts that read the actions except touch_body.py):
- Net faults: point 19 Ron_Ben_Ishay, 34 Maor_Haas, 50 Hiltinho. Point 56: Ron's added touch is a foot, about v3 12700.
- Point 27: no touch is missing; Maor_Haas's set (foot, v3 5049) is the last touch. Replaces the note above.
- Rally 19: touch 80 is a reception (Ron sets, then the net fault).
- Actions: on a 60-touch stratified sample the possession rule was right on 20/20 attacks and 20/20 sets but 14/20
  receptions. 32 of the 53 rallies whose first filmed touch was called a serve reception actually start with the
  set (the camera picked the rally up late): serve-reception rates now use only the 21 filmed serve receptions.
- Stance: 60 more owner labels (128 in all).

## Coach summary page (2026-11-25)

`coach_analysis/match_summary_he.html` (Hebrew, built by `scripts/site/build_summary.py` + `summary_template.html`):
every metric above in one page. Before building it, the branch `claude/tender-babbage-0996wx` (filled positions,
missing-touch review, owner's position squares) was merged with the labelling-game line and the whole pipeline was
rerun, `touch_body.py` included (the Sapiens2 file was pulled from LFS). New in `data/analysis/` from
`scripts/analysis/summary_extras.py`: `ball_after_touch.csv` (ball apex and time to the next touch after every
touch), `set_partner.csv` (partner distance, movement and facing at every set), `set_foot_side.csv` (ball/foot left,
middle or right of the setter's body at every foot set, 0.15 m middle band; 19 sets).
