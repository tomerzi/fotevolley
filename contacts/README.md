# Ball touches: who touched the ball, with which body part, and in which frame

Run: `python contacts/detect_contacts.py` (about 1.5 minutes, CPU)

## Output files
| file | contents |
|---|---|
| `ball_contacts.csv` | **one row per touch**: `frame`, `time_sec`, `player` (BRA_A/BRA_B/ISR_A/ISR_B), `team`, `body_part` (head/chest/shoulder/torso/thigh/knee/shin/foot/arm/hand), `limb_side` (right/left), `confidence` (0–1), `second_player`/`second_part` (the runner-up candidate), `note` (2D overlap, part ambiguity) |
| `unassigned_ball_events.csv` | sharp changes in the ball's direction with no player within reach (sand, net, tracking errors) |
| `contact_candidates.csv` | every (touch, player) pair that was considered, with every cost, for debugging |

## How it works
1. **Touch frame**: a quadratic is fitted to the ball's path before and after each frame. A touch is a peak of the velocity jump `|v_after - v_before|` (> 7 px/frame).
2. **Body part**: the ViTPose skeleton is turned into capsules (head, chest, thigh, shin, foot…). We measure the gap, in metres at the player's depth, between the ball and each capsule.
3. **Two players close to the ball in 2D** (e.g. one player's head and another player's foot): resolved by **depth**. The ball's size in pixels shows how far it is from the camera. Expected size = `BALL_K × px/m` at the player's court position (from the homography). The player whose depth matches the ball's size wins. When this happens, the `note` column reports both players and the ratio for each.
   Two more costs: a player on the other side of the net from the ball, and the same player touching twice in a row (a Viterbi pass over each rally).

## Checking by eye
- `python contacts/contact_sheet.py OUT_PREFIX START END`: a tile for each touch
- `python contacts/strip.py FRAME OUT.jpg`: a strip of frames around one touch

## Known limitations
- Dead-ball touches (catching the ball, juggling between rallies) are also detected. `hand` usually means a caught ball.
- A touch while the ball is hidden behind a body is detected only if there are enough frames on both sides of the gap.
- The head vs. chest/shoulder boundary is fuzzy when the ball arrives from above. See `second_part` and the note `part close to`.
