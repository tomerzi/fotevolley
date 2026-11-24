# Metrics catalog

All metrics are for one match. Unit is the rally window unless stated otherwise. The evidence class follows `references/statistics.md` of the skill.

| Metric | Formula / units | Window / anchor | Eligible population, exclusions | Class | Status |
|---|---|---|---|---|---|
| Points won after winning / after losing previous point | count / count, per team | scoreboard sequence | 58 points with a known previous winner (the first point of each set is excluded). Batch updates are kept because both of their points went to the same team | measured | used (F1) |
| Side-out proxy | P(win \| lost previous point) | as above | as above; equals receive-side win rate **only under the "winner serves" rule (not observed)** | measured + rule assumption | used (F1) |
| Pair separation | Euclidean distance between partners, metres | frame of the last ball observation on the opponent half before a clean crossing into the team's half | both partners observed and valid; 121/122 incoming events | measured (geometry unverified) | used (F2, F3) |
| Width / depth stagger | \|Δlateral\|, \|Δdepth\| (m) | same anchor | same | measured | used (F2) |
| Front / back depth | min / max partner depth from the net (m) | same anchor | same | measured | used (F2) |
| Lateral role | which partner has the lower `lateral_m` (left facing the net) | same anchor | same | measured | used (F2) |
| Occupancy 3×3 | fraction of active frames per cell; thirds of 3 m in lateral and depth | frames inside ball possession runs | 51,952 observed player-frames; duration weighted at 25 fps | measured | used (F2) |
| Defensive failure proxy | terminal_lost / (returned + terminal_lost) | incoming possession | 78 labelled possessions, 39 rally clusters; unclean (33) and terminal_won (10) excluded | exploratory (alignment unverified) | used (F3) |
| Modeled union / uncovered | `scripts/coverage.py`, 30×30 grid on a 9×9 m half | same anchor | 128 eligible, 1 excluded; 4 scenarios R = 2.65–5.27 m | modeled | used as a null check (F3) |
| Contact candidate | 2D ball low-point reversal within a box ±35% width / ±15% height | per frame | 281 candidates, 9 with an ambiguous player | exploratory | **not retained** (bias, unverified) |
| Coarse contact region | ball y relative to box: <0.18 head, <0.45 chest, <0.70 thigh, else lower leg/foot | candidate frame | as above | exploratory | **not retained** |
| Pose contact region | nearest keypoint group (head / chest / thigh / foot, score ≥ 0.5) to the ball centre; kept if the distance is ≤ 15% of box height | candidate frame | 220/281 candidates | exploratory | used (F4) |
| Head yaw proxy | sign of nose_x − mean visible ear_x vs the direction to the ball; \|yaw\| < 2% of box height counts as unknown | incoming anchor, offsets 0/5/10 frames | pairs with both players known | exploratory | not retained (no variance) |
| 2D knee angle / hip height | hip-knee-ankle angle in the image (°); (box bottom − mid_hip_y) / box height | same | keypoint score ≥ 0.5 | exploratory, 2D proxy only | not retained |
| Net-player recovery | depth at the next incoming crossing − depth at own crossing-out, for the partner nearer the net (m) | own attack → opponent return | 121 transitions, 33 labelled | measured geometry | descriptive only |
| Reception control, set attackability, attack conversion | — | — | require contact and action labels | unavailable | missing labels |
| Preparation / recovery latency, contact angles | — | — | require verified contacts, pose on v9 boxes, and a jump flag | unavailable | missing labels |
| Seam vulnerability (actual) | — | — | requires ball landing points | unavailable | missing labels |

## Hypothesis log (declared before comparison)
| ID | Type | Hypothesis | Result | Kept? |
|---|---|---|---|---|
| T1 | technical | Players differ in coarse contact region mix | Distributions are similar (head 63–70% for everyone). Likely detector bias | No: unverified measure |
| T2 | technical | First-touch region predicts whether the ball is returned | chest 1/16, head 2/31, foot 2/13, thigh 0/8 failures. Too few failures (5) | No |
| T3 | technical | Pose-based reception posture predicts control | No verified contacts, and pose comes from an older box set | Unavailable |
| A1 | tactical | ISR side-out conversion fell in set 2 | 10/17 in set 1 vs 5/17 in set 2 | **F1** |
| A2 | tactical | Teams differ in defensive formation (stagger, separation, roles) | Fixed left/right roles in both teams. ISR shows more variable stagger and separation | **F2** |
| A3 | tactical | Modeled uncovered area differs by team or outcome | No difference in any of 4 scenarios | Null, reported in F3 |
| L1 | linked | Wide pair separation at the incoming crossing goes with a failed defence | ISR 6/8 vs 3/31 at offset 0. Not robust to a 0.2 s anchor shift | **F3** (exploratory) |
| H6 | orientation | Defenders' head yaw (nose vs ears, 2D) toward the ball at the incoming crossing goes with a returned ball | 97–100% are toward the ball: no variance | No |
| H7 | technical | Readiness stance (min 2D knee angle, hip height in box) differs for returned vs failed balls | ISR direction flips between offsets 0, 5 and 10 frames | No |
| H8 | transition | Recovery of the net player after own attack (depth at attack → depth at next incoming) goes with the defence outcome | Both teams 1.3 → 3.4–3.6 m (recovery ~1.9 m); only 4 labelled failures | No (descriptive) |
| H9 | technical + linked | Teams differ in first-touch body region (nearest pose keypoint), and the region links to the possession structure | BRA foot 29% vs ISR 16% (CI +0.6 to +24 pp); first touch with the foot → 3+ touches 39% vs head 20%; opponent fail 0/15 | **F4** (exploratory) |
| L2 | linked | Number of touch candidates goes with a returned ball | 0 candidates: 8/11 failed; 3 or more: 0/30 failed. Largely tautological (a missed ball has no touch) | No |

Tests run: H2 used 3 predeclared shape splits × 2 teams = 6 comparisons. Bonferroni α = 0.0083. The ISR event-level Fisher p = 0.0006 passes nominally, but events within a rally are not independent and the effect fails the anchor-shift test. It is therefore treated as exploratory.
