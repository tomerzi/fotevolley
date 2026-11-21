# Self-marking experiment (no hand marks)

Can the two weak layers of the court page be fixed without a person marking frames?

## Where points ended (22 points with only a rally-end fallback)
Three automatic methods, each scored first on the 24 hand-marked landings (held out):

| method | found | median error | within 0.5 m | within 1 m |
|---|---|---|---|---|
| M1 flight extrapolated to the sand (g = 9.81) | 24/24 | 0.16 m | 16 | 18 |
| M2 first ball point <= 0.3 m after the last touch | 24/24 | 0.04 m | 15 | 18 |
| M3 bounce in the image, camera ray to the sand | 24/24 | 0.22 m | 15 | 16 |
| M2 and M3 agree within 0.5 m | 17/24 | 0.14 m | | |

Single methods fail badly on about a quarter of points (errors of 2-11 m), so only agreement is trusted.
On the 22 open points: 5 resolved by agreement, 4 more by looking at the frames (`landings_self.csv`), 13 stay
undecided - the ball flies toward the far stands or out of the frame, and from one camera a ball high above the far
court looks the same as a ball already out. Three existing labels look wrong (rallies 1, 46, 56).

## Partner gap at the opponent's attack (60 doubtful moments)
Feet re-measured from the pose (heels and toes on the sand; grounded frames around it when a defender is in the air).
On the 65 clean moments the pose gap agrees with the tracked gap: median difference 0.11 m, 90% within 0.30 m.
Of the 60 doubtful moments: 31 clearly below 3.5 m, 4 clearly above, 25 within 2 sigma of the line; 5 changed side.
ISR concede rate: above 3.5 m 54% (was 57%), at or below 32% (was 27%). Threshold-free, by gap terciles (both teams):
close 19% (2.5 m), middle 29% (3.1 m), wide 54% (3.7 m). The finding holds and does not depend on the 3.5 m line.

Scripts: `scripts/experiments/self_marks/` (landings_auto.py, landings_methods.py, gap_selfcheck.py).
