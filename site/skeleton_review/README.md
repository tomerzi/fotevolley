# Skeleton review

Review the two Israeli players frame by frame in the Graz 2022 final. The original v3 video and the unfiltered Sapiens2 pose appear side by side. The viewer includes three owner-reviewed approvals, keyed by player and v3 frame, recorded in [skeleton_review.csv](../../data/experiments/owner_checks_2026-11-24/skeleton_review.csv).

From the root of a local clone, with Python, NumPy, pandas and Git LFS installed, run:

```sh
git lfs pull --include="video/main_camera_clean_v3.mp4,data/pose/sapiens2/sapiens2_pose_v3_00000-13494.csv"
python site/skeleton_review/build_data.py
```

Open `site/skeleton_review/index.html` in a modern browser. The builder derives the viewer's local data from the files in the clone. Subsequent frame decisions are saved in that browser and can be exported as CSV.
