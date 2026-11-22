# Feet and ISR identity labels (2026-11-22)

Labelling page: https://claude.ai/artifact/TLca5Y2Nuz7WJHcyRLFfBR (answers in its database, collection `labels`).
`items.json`: every item on the page with the automatic answer (`auto` for feet; `touch_table` / `owner_review` for
points) and, for the identity frames, the hidden key `_answer` (which crop, A or B, is which player by our boxes).
Images: `scripts/experiments/feet_isr_labels/make_images.py` (ffmpeg + Pillow, v3 video); page: `tagging.html`.

1. Identity: 6 frames, both ISR players, "which one is Maor Haas?"
2. Points: the 28 points where touches_master.csv and point_end_verified.csv disagree on the last toucher.
3. Feet: 40 in-play touches (16 chest, 12 foot, 12 head; spread over players): each foot planted / sliding / air /
   unseen, stance (set / moving / jump), facing the camera.
