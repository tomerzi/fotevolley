# Match analysis plan (written 2026-10-22)

Goal: on top of what exists (player boxes and Sapiens2 pose, ball track, points/rallies, touches with player and body
part, 3x3 squares), compute: ball height (and ball position on the court), stance width at chest touches, action of
every touch (serve / reception / set / attack), teammate distance when a point is lost, player speed, area covered.
All work is in v3 frames (`*_v3video.csv`, Sapiens2, contacts); every output keeps `v2_frame` and `time_sec`.

## Match facts

- Event: **EFVL World Challenge Graz 2022** (banner in the video, bolao branding). EFVL report: final, Brazil beat
  Israel 2-0 (footvolleyeurope.com/news/28/6/2022/brazil-still-on-top).
- Players (scoreboard): ISR **R.B. Ishay** (Ron Ben Yishai, רון בן ישי) and **M. Haas** (Maor Haas, מאור האס);
  BRA **Hiltinho** (Hilton Joaquim de Melo Junior) and **Franklin** (Franklin Henrique Freitas dos Santos).
  ISR_B and BRA_B wear shirt number 2 (seen in 4954, 117, 7374, 13456). If number 1 is the first name on the
  scoreboard (usual convention), ISR_A = Ben Yishai, ISR_B = Haas, BRA_A = Hiltinho (yellow long sleeves, cap),
  BRA_B = Franklin. To be confirmed by someone who knows the players.
- Score: the EFVL report says 18:14, 18:11. The on-screen scoreboard shows ISR 13 - BRA 17 at the end of set 1
  (the last point is never shown) and our files follow the scoreboard (18:13, 18:11).
- Player heights: no published height found for any of the four (web search in English, Hebrew, Portuguese;
  pages could not be opened from the cloud environment, network policy). oGol/Sofascore heights for "Hiltinho"
  (169-172 cm) belong to a namesake footballer.
- Rules: men's net **2.20 m** (FIFV, CBFv, EFVL agree), posts 2.55 m, ball 68-70 cm circumference (~0.22 m).
- Court size: **open**. FIFV/CBFv say 18 x 9 m, EFVL (this event's league) says 16 x 8 m. The homography maps the
  court lines to 18 x 9, so if the court is 16 x 8 every length/speed/area is 12.5% too big (x 8/9).
  Ball-on-sand check (9 frames runs, ball 0.22 m): detector box width gives 0.246 court units (fits 16 x 8),
  colour segmentation 0.20-0.22 (fits 18 x 9) - the ball is ~20 px with blurred edges and an attached shadow, so
  inconclusive. To settle in step 1: gravity in the ballistic fits and the net height.

## Steps

### 0. Setup + smoothed player positions - done
`scripts/analysis/player_positions.py` -> `data/analysis/player_positions_v3.csv` (one row per frame and player).
- Ground point by posture: standing -> ankle midpoint (lowered to the sole); the planted ankle only when the other
  foot is higher AND swinging while the planted one stays still (image height alone confuses a raised foot with
  feet apart in depth); lying on the sand -> hips; upside down (bicycle kick) -> no measurement, bridged;
  sitting / kneeling / no pose -> box bottom. Ankle keypoints outside the player's box or on the teammate's ankle are
  dropped. Occluded = teammate's box overlapping (IoU > 0.4) or in front of the feet: counts less, and a hidden box
  bottom gives no measurement. Boxes cut by the image border count less; very short ones give no measurement.
- Jumps with an upright pose: the point slides away from the camera while the feet are off the sand. Bumps in the
  distance from the camera's ground point (>= 0.7 units, 1.0 with no own touch near; <= 15 frames wide at half
  height not counting upside-down frames; more than running could do) are taken out over the whole bump, each row
  by its own excess; `airborne` = jump or upside down.
- Constant-velocity Kalman + RTS per player and camera segment; pixel noise through the homography Jacobian.
  After 4 outliers in a row the filter is reset in place from a line fit of them (unless the next measurements are
  back on the old track: a glitch); the RTS pass joins the two sides of a reset.
- Result: speed median 1.3, p99 5.8, p99.9 7.3 court units/s, the same at every depth (raw box bottoms: p99 14);
  largest frame-to-frame step 0.39 units; summed steps = integrated speed (3404 vs 3412 units).
- Verification (2026-10-22): two rounds of independent checks (code review, jumps / speeds / postures / occlusion by
  eye in the video, data integrity) found ~20 then 8 problems; all fixed except the limits below. On the
  verifiers' labelled cases: 13 of 14 seen jumps flagged and 1 of 18 non-jumps (ISR_A 10800-10812, standing on one
  leg - the position correction is still right there); the bicycle kick BRA_B 4196-4228 now rises smoothly
  2.8 -> 4.3; no reset makes a position jump.
- Limits: small hops (< 0.7 units of bump) are not flagged; a jump hidden behind the teammate is not flagged
  (BRA_B 4820-4838, its upside-down frames are still airborne); sitting down with the legs toward the camera can
  push the ankles ahead of the body for a few frames (ISR_A 11263, top speed 9.1 vs ~5 from the hips).

### 1. Ball height and ball position on the court
1. Full camera calibration: the homography alone does not give it (decomposition is inconsistent: orthogonality
   gives f^2 < 0). Decomposing H in court units gives f, R and the camera position in court units, so one known
   height in metres then gives the court scale. Vertical references seen in the median background (players
   removed): the near post top (~ (1017, 627) px; its base is below the image), the striped antenna above the near
   end of the net (red bands ~37.5 px apart around y 490-617; volleyball antennas are 1.8 m with 10 cm bands - to
   confirm for this event), the net top at 2.20 m. The camera is ~1 m from the net plane (its ground point is
   (8.1, 20.6) court units), so the net is seen almost edge-on and its top tape is nearly a vertical line.
   Fit f, principal point (and maybe k1), R, t. Check: reprojection < 3 px, standing players' head heights.
   Tried (2026-10-22): vertical vanishing point from 7,560 standing-still players (ankle-mid -> shoulder-mid lines,
   robust LS) = (1163, 5428) px; solving K algebraically from it + H is ill-conditioned (principal point
   y = 241, the camera then misses H by ~40 px at (9, 9)). Next: a joint nonlinear fit (court line points, the
   players' vertical lines, antenna, ball parabolas) with a prior keeping the principal point near the centre.
   Once calibrated, standing players' heights in metres come out too (the web had none).
2. 3D ballistic fit per free-flight piece (`scripts/contacts/ballflight.py` pieces): position + velocity, gravity
   9.81 m/s^2, minimising pixel error + ball-size term (0.22 m) + anchors (sand events z ~ 0.11 m; at a touch the
   ball is at the touching body part of that player, whose court position is known from step 0).
3. Fit g freely on long clean flights first: g in court units/s^2 gives the court scale (9.81 -> 18 x 9,
   11.04 -> 16 x 8).
4. Output `data/analysis/ball_3d_v3.csv` (frame, x, y, z, speed, piece, quality) + per touch: contact height, apex
   after it, height over the net, landing point. Checks: z at bounces ~0.1, headers ~1.8, net crossings > 2.2,
   video with the height drawn, compare with `data/contacts/ball_height_f0-99.csv`.

### 2. Stance width at chest touches - first version done
`scripts/analysis/chest_stance.py [SHEET.jpg]` -> `data/analysis/chest_stance_v3.csv`. 99 chest touches; heels/toes
of each foot (median of t-1..t+1) -> court units; stance, along / towards the net, side-to-side / front-back of the
hip line (the hip line projected as if on the sand keeps its direction: a homothety about the camera's foot).
One camera cannot tell a raised foot from one set further back, so each foot gets a status from its motion over
t-2..t+2: planted / moving (a step) / raised (moving and higher). reliable = no raised foot, no jump (90 in play).
Feet points checked on a 12-touch sheet.

### 3. Serve / reception / set / attack - basic version done
`scripts/analysis/touch_actions.py` -> `data/analysis/touch_actions_v3.csv`. Possession = consecutive in-play
touches of one team: 3 -> reception, set, attack; 2 -> reception, attack; 1 -> attack. The losing team's last
possession of 1-2 touches is a failed defence (reception/set, subtype failed) unless the ball then landed at the net
or on the other side. Attack subtypes: kill / failed_defence / error / returned.
The serve is never in this video (the main camera picks the rally up at the reception): 60 inferred serve rows
(team = winner of the previous rally, frame = rally start, player unknown).
Result: 31 kills, 12 failed defences, 15 attack errors; 10 rows flagged to check (a 5-touch possession in rally 39,
first touch by the serving team in rally 2...).
Next: refine with the ball's 3D path from step 1 (set goes up and stays on its side; attack crosses the net).

### 4. Teammate distance when a point is lost - done
`scripts/analysis/team_spacing.py` -> `data/analysis/team_spacing_v3.csv`. At every attack (step 3), the defending
pair's distance, along / towards the net, squares; conceded = kill or failed_defence, baseline = attacks they
returned. A second row at the ball landing for conceded points.
Median distance at the attack: ISR conceded 3.95 (kill) / 4.1 (failed defence) vs returned 3.6; BRA 3.25 / 3.3
vs 3.1 (court units).

### 5. Player speed - done
`scripts/analysis/player_speed.py` -> `data/analysis/player_speed_rallies_v3.csv` (per rally and player: distance,
mean, top speed = best 0.5 s average, time above 4 units/s, jump frames) and `touch_speed_v3.csv` (approach speed
over the 0.5 s before each touch).

### 6. Area covered - done except the team-level part
`scripts/analysis/coverage.py [HEATMAP.png]` -> `coverage_rallies_v3.csv` (per rally: hull, squares),
`coverage_match_v3.csv` (hull, 50% / 90% areas, squares), `coverage_heatmaps_v3.png`. Match-level numbers are in the
team's view (distance from the net, left / right facing the net), since the teams switch sides: each player then has
one clear zone - BRA_A and ISR_A play the right side, BRA_B and ISR_B the left; 90% areas 32-36 units^2.
Next: at each opponent attack, the part of the half closer to each defender (Voronoi).

## Order
0 -> 5 -> 2, 4, 6 -> 3 (basic) -> 1 -> 3 refined. The court scale (step 1) applies to every length at the end.
Run order: player_positions.py, touch_actions.py, then team_spacing.py, player_speed.py, coverage.py,
chest_stance.py (all in scripts/analysis/, each reads the outputs of the first two).
