"""One-off conversion (2026-10-31): court units (homography mapped the court lines to 18 x 9) -> metres on the real
EFVL 16 x 8 m court. Every length / position / speed is multiplied by 8/9 (both axes scale the same), areas by 64/81,
pixels-per-unit by 9/8. The old homography is kept as data/mappings/homography_px_to_court_units_18x9.npy for the
experiments in scripts/experiments/ that were built on it. Run once from the repo root; it refuses to run twice."""
import os, shutil, numpy as np, pandas as pd

S, A = 8 / 9, 64 / 81
M = 'data/mappings/'
if os.path.exists(M + 'homography_px_to_court_units_18x9.npy'):
    raise SystemExit('already converted')
H = np.load(M + 'homography_px_to_m.npy')
shutil.copy(M + 'homography_px_to_m.npy', M + 'homography_px_to_court_units_18x9.npy')
np.save(M + 'homography_px_to_m.npy', np.diag([S, S, 1.0]) @ H)

LEN = {
    'data/boxes/player_boxes_v9.csv': ['court_x', 'court_y'],
    'data/boxes/player_boxes_v3video.csv': ['court_x', 'court_y'],
    'data/tracks/player_boxes.csv': ['court_x', 'court_y'],
    'data/tracks/player_identities.csv': ['court_x', 'court_y'],
    'data/court/player_squares_v3.csv': ['court_x', 'court_y'],
    'data/analysis/player_positions_v3.csv': ['raw_x', 'raw_y', 'bump', 'x', 'y', 'vx', 'vy', 'speed'],
    'data/analysis/player_speed_rallies_v3.csv': ['distance', 'mean_speed', 'top_speed'],
    'data/analysis/team_spacing_v3.csv': ['distance', 'towards_net', 'along_net', 'p1_x', 'p1_y', 'p2_x', 'p2_y'],
    'data/analysis/touch_speed_v3.csv': ['approach_speed', 'speed_at_touch'],
    'data/analysis/chest_stance_v3.csv': ['left_x', 'left_y', 'right_x', 'right_y', 'stance', 'towards_net',
                                          'along_net', 'side_to_side', 'front_back'],   # *_move are box heights
    'data/analysis/coverage_rallies_v3.csv': ['max_from_mean'],
    'data/contacts/ball_contacts_v3.csv': ['gap_m', 'part_speed_ms'],
    'data/contacts/contact_candidates_v3.csv': ['gap_m', 'gap2_m', 'part_speed'],
}
AREA = {'data/analysis/coverage_match_v3.csv': ['hull_area', 'area50', 'area90'],
        'data/analysis/coverage_rallies_v3.csv': ['hull_area']}
INV = {'data/contacts/contact_candidates_v3.csv': ['scale']}   # px per unit -> px per m

for f in sorted(set(LEN) | set(AREA) | set(INV)):
    d = pd.read_csv(f, keep_default_na=False, na_values=[''], dtype=str)
    for k, cols in ((S, LEN.get(f, [])), (A, AREA.get(f, [])), (1 / S, INV.get(f, []))):
        for c in cols:
            v = pd.to_numeric(d[c], errors='coerce')
            dec = d[c].str.split('.').str[1].str.len().max()
            dec = int(dec) if dec == dec else 3
            d[c] = np.where(v.notna(), (v * k).round(max(dec, 2)).map(lambda x: f'{x:.{max(dec, 2)}f}'.rstrip('0').rstrip('.') if x == x else ''), d[c])
    d.to_csv(f, index=False)
    print('converted', f)
