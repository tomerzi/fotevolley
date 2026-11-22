# User notes, round 1 (doubt-marking page)
- 3, 5: the landing was a few frames before the clip started -> clips now start 5 s before the point ends.
- 12: the player's back hides the ball after it hits the net and then the sand; landing frame estimated.
- 13: ball hits the net but play continues -> clip must continue (now +4 s).
- 14: a Brazil player touches the ball and sends it out of play -> point ISR (matches the data).
- 19: unclear, need to see what came before; ball went out, impact not visible; frame and spot estimated.
- 27: an Israeli player heads the ball out; the ball hits the net and then the sand (data: point BRA).
- 32: unclear moment and place of impact -> check against the project's tags.
- 33: an Israeli player kicks the ball out -> point BRA (matches).
- 34: play continues, no point to anyone in the clip -> clip must continue.
- 35, 42: end of a rally, need a few seconds before.
- 37: a Brazil player kicks the ball out -> point ISR (matches).
- 40: Israel had already touched 3 times -> point BRA (matches).
- 41: camera cuts away; by the ball's height probably point BRA -> check the tags (data: BRA).
- 46: an Israel player touches the ball and it lands on his own side -> point BRA (matches).
- 47: impact not visible, by the shadow the ball went out -> point BRA (matches).
- 50: a Brazil player crosses the net line; impact not visible, point ISR by the ball's direction (data: ISR).
- 55: a Brazil player touches the ball and sends it out; the moment of that touch is marked (point ISR, matches).
- Gaps: 2 clicks are not enough, each player has two feet -> page now takes 4 clicks.
- 39: a Maor Haas touch between touches 7 and 8 resets Brazil's touches (applied: frame 10506).
- 49 (question): the ball hits the net and falls on Israel's side -> Brazil won; the final scoreboard gives the point to Israel.

# Round 2
- 13: an Israeli player kicks the ball out -> point BRA (matches); kind set to out, no landing seen.
- 19: the point is cut in the middle -> go by the scoreboard (BRA, matches).
- 21: ignore the "out" click -> landing back to unknown.
- 34: a Brazil player crossed the net line (scoreboard still gives BRA) -> conflict noted, scoreboard kept.
- 35: the clip still did not reach the ground contact -> clip now covers the whole rally 9110-9400; the tags have no in-play touch in this rally.
- 42: ball hits the net and Israel had already touched 3 times -> point BRA (matches).
- 27 (question): point Brazil (matches).
- Gaps: 48 marked with 4 feet, 6 "cannot tell". vs tracking median |diff| 0.21 m; 6 crossed the 3.5 m line.
  ISR concede above 3.5 m 53% (17/32), at or below 31% (10/32), p = 0.076 (was 58% vs 27%, p = 0.013).

# Round 3
- 35: the mark (frame 9152) is where the Brazilian's foot crosses the net line, i.e. the point ended on a net-line
  fault by Brazil; the scoreboard still gives the point to Brazil (0-4) -> conflict noted, scoreboard kept.
- 41 gap (frame 10996): "cannot tell".
- Gaps: the user asked to go by the pose joints ("the most certain there is"). All 126 attack moments now use the
  feet joints (heel + big toe, conf >= 0.5; grounded frames around it when in the air). Pose vs the user's 4-click
  marks: median |diff| 0.26 m (pose 0.18 m smaller on average); pose vs tracking 0.11 m.
