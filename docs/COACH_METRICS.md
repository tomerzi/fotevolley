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

## Open: needs a human decision / check

1. **ISR names in the point review.** `point_end_verified.csv` `last_touch` disagrees with `touches_master.csv` in
   28 of 56 points; for Israel it is a swap (owner says Maor_Haas where the touch table says Ron_Ben_Ishay in 22 of 26
   ISR points, and the reverse in 8 of 8). Frames 528 and 887 (v3) show the touch table matches the player boxes and
   `graz2022_final_player_names.csv` (Maor_Haas = white armband). Point outcomes here use the touch table; the owner
   should confirm which player is which before the review's `last_touch` is used anywhere.
2. **Stance at the touch.** Against the 99 hand-marked chest touches (`data/experiments/chest_marks`, feet on the
   sand: 95 both, 4 air) the automatic `air` finds 1 of the 4. "Planted" (foot still ±2 frames) is a different
   question from "on the sand". Needs hand labels per foot (planted / sliding / in the air) on a sample (~40 touches)
   to set the thresholds.
3. **Facing.** Spot-checked on 16 chest touches only; a labelled sample of ~50 frames would give its error rate.
4. Position from heel + toes instead of ankles was tested (`POS_GROUND=sole` in `player_positions.py`): noisier in
   the sand (outliers 3.1% vs 2.2%), shift 6 cm median, so the ankle stays the default.
