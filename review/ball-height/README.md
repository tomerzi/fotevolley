# 300-frame ball-height review

Download and play `review_300_frames.mp4` (approximately 6 MB).

This video contains exactly 300 consecutive frames (6340–6639) from `main_camera_clean_v2.mp4`, corresponding to 04:13.60–04:25.56 at 25 fps. There are 199 experimental height estimates and 101 unknown heights.

The circle marks the observed ball. The line and cross represent the model's predicted vertical and ground position, not an independent measurement. Height is the ball center above the assumed sand plane in metres. UNKNOWN does not mean zero.

These monocular ballistic estimates are unvalidated. Camera calibration includes extrapolated court/net reference points. A small image fitting error does not establish accurate 3D height. Do not treat these numbers as verified measurements.

## Revised review (v2)

`review_300_frames_v2.mp4` shows the SAME frames 6340–6639. Coverage is 270/300 (90%), versus 199/300 previously. The original 199 heights are retained unchanged. The 71 added heights use observed inliers at the edges of accepted ballistic fitting windows and are labeled EDGE ESTIMATE. No fitting acceptance thresholds were relaxed and no heights were extrapolated outside accepted windows. The remaining 30 frames stay UNKNOWN. All 300 frames have ball image positions (one of the 30 unresolved-height positions is interpolated); unresolved heights should not be described as missing detections.

More coverage is not evidence of greater metric accuracy. Calibration and all heights remain unvalidated, and edge estimates are more sensitive to contacts/model mismatch. Frame-level data are in `review_300_frames_v2.csv`.
