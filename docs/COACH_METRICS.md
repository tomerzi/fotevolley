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

## Open

1. Points 13 and 27 (ISR, ball out): the touch table now says Maor_Haas touched last, the owner review Ron_Ben_Ishay.
   Points 28 and 56: the owner review says Ron_Ben_Ishay, the touch table Franklin (the owner chose Franklin for 28 on
   the labelling page). Point 2: Hiltinho (table) vs Franklin (review), not visible.
2. Stance: calibrated on the 40 labelled touches (`stance_cal` in `touch_body.csv`): jump from the jump test (10 of
   12), else set when the body moves < 1.3 m/s at the touch; 67% leave-one-out over set / moving / jump. Foot-by-foot
   planted / sliding / air from the pose stays weak (about 45%) and is not used for conclusions.
3. Facing: 25 of 40 against the labels; side-on players are often called 'camera'.
4. Position from heel + toes instead of ankles was tested (`POS_GROUND=sole` in `player_positions.py`): noisier in
   the sand (outliers 3.1% vs 2.2%), shift 6 cm median, so the ankle stays the default.
