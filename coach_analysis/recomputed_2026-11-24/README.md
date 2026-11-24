# Coach metrics after the labelling game (2026-11-24)

Made by `scripts/analysis/round2_recompute.py` from `data/contacts/touches_master.csv` + every row of
`data/contacts/owner_decisions.csv`. Nothing in `data/analysis/` was changed: the files there (touch_outcomes.csv,
set_quality.csv, team_spacing_v3.csv, return_to_base.csv, phase_heatmaps.*) still predate these decisions and need the
pipeline rerun (touch_actions.py -> build_touch_table.py -> the analysis scripts) to absorb them.

Definitions:
- Action = the owner's decision when there is one, else the possession rule (3 touches: reception, set, attack;
  2: reception, attack; 1: attack; a losing team's last 1-2 touch possession: failed reception).
- `reception_unfilmed`: the rally's first filmed touch is a set, so the serve reception was not filmed. The set is
  touch 2 of its possession and the attack after it touch 3. These rallies are left out of serve-reception rates.
- Reception stability = a filmed reception (touch 1 of its possession) whose possession reaches an attack.
  serve = first possession of the rally, attack = any later possession.
- Touch success = the point was not lost on that touch (coach's definition).
- Stance: owner labels only (68 earlier + 60 from the game = 128); no automatic stance.

| file | content |
|---|---|
| touches.csv | one row per in-play touch with final action, possession, touch number, success, stance |
| touches_by_action_part.csv | player x action x body part |
| touch_success.csv | success / eligible per player x action, Wilson 95% interval |
| reception_stability.csv | filmed receptions by team, player and body part, serve vs attack |
| unfilmed_reception_rallies.csv | rallies starting with a set, per team |
| set_quality_sets.csv | sets only: outcome, air time, apex, partner movement (from set_quality.csv rows) |
| attacks.csv | outcome and touch number in possession |
| points.csv | how each point ended and who lost it (net faults by the owner's labels) |
| stance.csv | owner stance labels per player x action |
