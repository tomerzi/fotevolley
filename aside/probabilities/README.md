# Probabilities (set aside)

Moved out of the main project at the user's request: the "chances" metrics are not relevant to the project now.

- `build_analysis_ui.py` - computes set win probability (score and server only), touch value (change in the
  rally-win chance from the touch's state) and attack quality (ridge logistic on body part, angle, contact height,
  length; barely better than the base rate: Brier 0.213 vs 0.224, in-sample).
- `analysis_ui/` - the match-analysis console page that showed them (published copy:
  https://claude.ai/artifact/9Q4zSRStRm5GB8ViSKm2DZ).

Run from the repo root: `python aside/probabilities/build_analysis_ui.py`. It reads `site/court_story/data.json`.
