# Results website

`built/` is the published site (https://claude.ai/artifact/HoJCNw9TMzDPrr8p7yCaPv) as of the last update:
`index.html` with all data embedded, example frames, chest-touch crops and the overlay video in 60 s segments
(`video/partNN.mp4`, from `scripts/ball_height/render_video.py`). Open `built/index.html` through a local web server
(e.g. `python3 -m http.server` in `built/`) so the video segments load.

Rebuild: `python3 scripts/ball_height/render_video.py site/built/video` then `python3 scripts/site/build_site.py site/built`.

`marking_pages/`: the data behind the hand-marking pages (chest touches, sand landings, point moments, ball
shadows); their builders and templates are in `scripts/experiments/*_marks/`, the marks themselves in
`data/experiments/*_marks/marks/`.
