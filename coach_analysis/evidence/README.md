# Evidence for the coach summary

Made by `scripts/evidence/render_evidence.py` from `video/main_camera_clean_v3.mp4` (Git LFS) and the analysis tables.

- `touches/t{touch}.jpg`: every in-play touch at its contact frame: zoom on the player with the Sapiens2 skeleton
  (feet line at chest touches) and every value measured on that touch.
- `points/p{point}.jpg`: the moment each point was decided (the frame the partner distance is measured at), landing marked.
- `clips/r{rally}.mp4`: the 60 annotated rally clips. Not in git (Git LFS uploads fail from the cloud session); they are
  published with the page and are rebuilt by the script (about 15 min).
- `index.json`: clip start/end frame per rally, and where each touch / point is in its clip.
