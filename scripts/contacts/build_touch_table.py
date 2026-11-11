"""One table for every touch: the hand labels (truth) next to what the detector said, the action and the ball.

One row per labelled ball event (363). The label columns are the truth; `auto_*` is the detector's own answer for
the same event, kept beside it so the detector can be scored and retuned against the labels.

inputs : data/contacts/touch_labels_final.csv   hand labels: kind, player, body part, side
         data/contacts/ball_contacts_v3.csv     detector output (player, part, confidence, runner-up)
         data/analysis/touch_actions_v3.csv     serve / reception / set / attack
         data/analysis/ball_height_v2.csv       ball height and court position (v2 frames)
         data/mappings/frame_mapping_v3.csv     v3 -> v2 frame
         data/points/rallies_v3video.csv        set of each rally
output : data/contacts/touches_master.csv
usage: python scripts/contacts/build_touch_table.py"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FPS = 25
TEAM = {'Maor_Haas': 'ISR', 'Ron_Ben_Ishay': 'ISR', 'Franklin': 'BRA', 'Hiltinho': 'BRA'}


def read(*p, **k):
    return pd.read_csv(os.path.join(ROOT, *p), **k)


def main():
    lab = read('data', 'contacts', 'touch_labels_final.csv')
    det = read('data', 'contacts', 'ball_contacts_v3.csv')
    act = read('data', 'analysis', 'touch_actions_v3.csv')
    ball = read('data', 'analysis', 'ball_height_v2.csv')
    fmap = read('data', 'mappings', 'frame_mapping_v3.csv')
    ral = read('data', 'points', 'rallies_v3video.csv')

    t = lab.rename(columns={'side': 'limb_side'}).copy()
    t['time_sec'] = (t.frame / FPS).round(2)
    v2 = fmap[~fmap.deleted].set_index('v3_frame').v2_frame
    t.insert(2, 'v2_frame', t.frame.map(v2).astype(int))
    t.insert(5, 'set', t.rally.map(ral.set_index('rally').set))
    t['team'] = t.player.map(TEAM)
    unknown = set(t.player.dropna()) - set(TEAM)
    assert not unknown, f'players without a team: {unknown}'

    d = det[det.event_type == 'touch'].set_index('frame')
    assert d.index.is_unique and t.frame.isin(d.index).all(), 'every label must sit on one detector touch event'
    for new, old in [('auto_player', 'player'), ('auto_part', 'body_part'), ('auto_side', 'limb_side'),
                     ('auto_in_play', 'in_play'), ('auto_confidence', 'confidence'), ('auto_source', 'source'),
                     ('auto_second_player', 'second_player'), ('auto_second_part', 'second_part'),
                     ('auto_cost_margin', 'cost_margin')]:
        t[new] = t.frame.map(d[old])
    real = t.kind == 'player'
    # the detector called all of these a touch, so a label that is not a player touch is a false detection
    t['auto_false_touch'] = ~real
    t['auto_player_ok'] = (t.auto_player == t.player).where(real)
    t['auto_part_ok'] = (t.auto_part == t.body_part).where(real)

    a = act.dropna(subset=['touch']).set_index('touch')
    assert a.index.is_unique
    for new, old in [('action', 'action'), ('action_subtype', 'subtype'), ('possession', 'possession'),
                     ('touch_in_possession', 'touch_in_possession'), ('possession_len', 'possession_len'),
                     ('action_check', 'check')]:
        t[new] = t.touch.map(a[old])

    b = ball.set_index('frame')
    for new, old in [('ball_height_m', 'height_m'), ('ball_height_sigma_m', 'height_sigma_m'),
                     ('ball_court_x_m', 'court_x_m'), ('ball_court_y_m', 'court_y_m'), ('ball_method', 'method')]:
        t[new] = t.v2_frame.map(b[old])

    cols = ['touch', 'frame', 'v2_frame', 'time_sec', 'rally', 'set', 'in_play', 'kind', 'player', 'team',
            'body_part', 'limb_side', 'action', 'action_subtype', 'possession', 'touch_in_possession',
            'possession_len', 'action_check', 'ball_height_m', 'ball_height_sigma_m', 'ball_court_x_m',
            'ball_court_y_m', 'ball_method', 'auto_player', 'auto_part', 'auto_side', 'auto_in_play',
            'auto_confidence', 'auto_source', 'auto_second_player', 'auto_second_part', 'auto_cost_margin',
            'auto_false_touch', 'auto_player_ok', 'auto_part_ok']
    out = os.path.join(ROOT, 'data', 'contacts', 'touches_master.csv')
    t[cols].to_csv(out, index=False)

    r = t[real]
    print(f'{len(t)} labelled events -> {out}')
    print(f'  real touches {len(r)} (in play {int(r.in_play.sum())}), false detections {int((~real).sum())}: '
          + ', '.join(f'{k} {v}' for k, v in t[~real].kind.value_counts().items()))
    print(f'  detector body part right {int(r.auto_part_ok.sum())}/{len(r)}, player same as label '
          f'{int(r.auto_player_ok.sum())}/{len(r)} (the label inherited the player, not an independent check)')
    print(f'  with an action {int(r.action.notna().sum())}, with ball height {int(r.ball_height_m.notna().sum())}, '
          f'action rows to check {int((r.action_check == True).sum())}')  # noqa: E712


if __name__ == '__main__':
    main()
