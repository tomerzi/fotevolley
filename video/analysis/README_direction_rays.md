# Body direction rays (eyes, shoulders, knees, feet)

| file | what | span |
|---|---|---|
| `video/analysis/body_direction_rays_full_match.mp4` | laser rays from eyes, shoulders, knees and feet of all four players, with a top-down inset | whole video, 623 s |
| `data/analysis/body_direction_full_match.csv` | per frame, player, body part and side: direction and angle to the net | frames 0-15499 (13,495 frames with players tracked) |
| `video/analysis/body_direction_rays_first_30s.mp4`, `data/analysis/body_direction_first_30s.csv` | the same, first 30 s | frames 0-749 |
| `video/analysis/shoulder_rays_to_net_first_30s.mp4`, `data/analysis/shoulder_direction_first_30s.csv` | first version: dashed shoulder rays that run to the net | frames 0-749 |

Made by `scripts/analysis/body_rays.py` (all parts) and `scripts/analysis/shoulder_rays.py` (shoulders to the net).
Angle to the net: 0 = square to the net, 90 = side-on, 180 = back to it.
