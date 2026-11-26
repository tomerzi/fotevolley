# Labelling game (2026-11-24, evening)

Page: https://claude.ai/artifact/Uj1UMHUR7fvzkzxsrdmBL3 (answers in its database, collection `answers`).
Page code: `scripts/experiments/labelling_game/index.html`; clips and item list: `scripts/experiments/labelling_game/make_game.py`
(1.2 s clip per touch, the touching player boxed in yellow at the touch frame; full-frame clips with the v3 frame number
for the open points).

`answers.csv` holds all 112 answers:
- `net_fault` (points 19, 34, 50): who touched the net. Ron_Ben_Ishay, Maor_Haas, Hiltinho; winners as recorded.
- `hidden_touch` (point 27): the last touch is Maor_Haas's set (foot) at v3 5049; no touch is missing.
- `added_touch` (point 56): Ron_Ben_Ishay, foot, about v3 12700.
- `missing_touches` (rally 39): the moment marked is touch 251; no touch is missing (owner confirmed in chat).
- `rally_check` (rally 19): touch 80 (Maor_Haas, shin) is a reception; Ron sets and touches the net.
- `sample_touch` (60): stratified sample, 5 per player x action (reception / set / attack), random within the cell,
  preferring touches with no earlier stance label. Action + stance (set / moving / jump) per touch.
  "dead ball" answers mean an attack that failed and went out (owner, in chat).
- `first_touch` (44): every other first filmed touch of a rally labelled as reception: reception or set.
- `approval`: the owner approved this update.

Findings:
- The automatic action was right on all 20 sampled attacks and 20 sampled sets, but only 14 of 20 receptions.
- 32 of the 53 first filmed touches of a rally are sets: the camera picked the rally up after the serve reception.
  These are now `action=set` in `data/contacts/owner_decisions.csv`.
- Touch 23 (Maor_Haas, rally 7) is a first-touch attack that went out, not a reception.

All decisions are in `data/contacts/owner_decisions.csv` (rows noted "labelling game"). Metrics recomputed with them:
`coach_analysis/recomputed_2026-11-24/` (`scripts/analysis/round2_recompute.py`).
