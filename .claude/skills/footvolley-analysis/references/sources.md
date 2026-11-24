# Sources and version checks

From the repo root:

```sh
git status --short -b
git for-each-ref --format='%(objectname) %(refname)' refs/heads refs/remotes
git ls-remote --heads origin
GIT_LFS_SKIP_SMUDGE=1 git fetch --no-tags origin '+refs/heads/*:refs/remotes/origin/*'
```

Fetch when relevant branches are missing or freshness matters and network access is available. If offline, use local refs and state their coverage. The fetch updates remote-tracking refs. Read individual objects with `git show "$sha:$path"`; use `git ls-tree -r --name-only "$sha"` to locate files and `git rev-parse "$sha:$path"` to spot identical blobs across refs. A Git LFS pointer contains no observations. For needed LFS data, obtain just the required objects if available or report the affected metrics. No checkout or branch merge is needed.

## Branches inspected

| Ref | Tip inspected | Main use |
| --- | --- | --- |
| `origin/claude/ball-handling-positioning-info-jspaw5` | `f900acf1494a75528e7a193aaaff525704073ca2` | Primary coaching bundle. Read `README.md`, `docs/COACH_METRICS.md`, `docs/ANALYSIS_PLAN.md` and `docs/CONTACTS_PROGRESS.md`. |
| `origin/claude/match-analysis` | `493b13815d922632cea0a1bf6b5826ba1b5cef19` | Earlier reviewed point endings and positioning inputs. Israel's names differ from the later identity correction. |
| `origin/repo-reorganized`, `origin/dead-time-removed-v3`, `origin/bbox-final-v9`, `origin/main` | Resolve on use | Source lineage, video mappings, reviewed boxes and baseline tracks. |
| `origin/claude/ball-contact-detection-vlmfcn` | Resolve on use | Detector candidates and unassigned events. |
| `origin/ball-height-3d-shadow`, `origin/ball-height-review-300-frames`, `origin/sun-shadow-ball-height` | Resolve on use | Flight reconstruction and limited-scope validation. |
| `origin/claude/determined-volta-tgcy59`, `origin/claude/optimistic-tesla-s8w2yu` | Resolve on use | Alternative analysis definitions for corroboration and sensitivity. |

## Join and provenance checks

- Match is the EFVL World Challenge Graz 2022 men's final, BRA versus ISR. Use the later branch's identity CSV at `coach_analysis/graz2022_final_player_names.csv` and correction note. On the inspected coaching tip, ISR_A is Ron_Ben_Ishay and ISR_B is Maor_Haas; BRA_A is Franklin and BRA_B is Hiltinho. The older branch gives ISR names in the opposite order. Owner-supplied marks retained their original names in the later correction. Treat rendered labels on old images as stale.
- `data/points/rallies_on_court.csv` records on-court and scoreboard outcomes; prefer on-court winners. Read `data/points/point_end_verified.csv` and `data/points/landing_marks_owner.json` for observed endings. The inspected bundle has 60 points, including 4 excluded endings, and documents a scoreboard error at point 49. Count and validate current records instead of assuming fixed totals.
- `data/contacts/touches_master.csv` gives reviewed touch IDs and parallel auto fields. Read `data/contacts/owner_decisions.csv` and `data/experiments/doubts/answers.csv` for later field-level decisions. Some derived metrics predate those decisions. Apply corrections once or verify they are already present. Retain uncertainty independently for winner, last player, body part, exact frame, and possession order. Point 27 has a confirmed last player but a hidden missing touch. Point 56 has a confirmed additional last player with an estimated frame and unknown body part. Point 2 includes an explicitly dead touch. Rally 39 contains possible missed contacts. Recheck each claim against current data.
- Match IDs, rally/point IDs, touch IDs and frame/player keys have different grains. Verify rally-to-point alignment and one-row uniqueness before joining. `frame` in contacts, rally windows and player positions is usually v3. Ball height and full-match facing use v2. Use `data/mappings/frame_mapping_v3.csv` and `frame_mapping_rallies.csv` for conversions, and inspect producer code for unclear frame fields such as terminal review frames. Deleted frames may have no v3 equivalent. The inspected videos run at 25 fps; use the correct time base and check continuity before subtracting timestamps.
- Court is calibrated on a 16 × 8 m plane with net at x=8. Legacy experiments use 18 × 9 units. Validate calibration and orientation before combining results. To align sides in the current convention use (x,y) for the team on the left and (16−x,8−y) for the team on the right, so own baseline is x′=0 and the net x′=8. Determine side by rally and team. Feet locations, ball projections and actual sand contacts are different kinds of positions.

## Design references

The skill uses Anthropic's short entry point with linked references from [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator) and explicit calculation and validation guidance from [xlsx](https://github.com/anthropics/skills/tree/main/skills/xlsx). Installation and invocation follow [Claude Code skills](https://code.claude.com/docs/en/skills). Match-specific rules and geometry come from the repo's analysis plan and reviewed data.
