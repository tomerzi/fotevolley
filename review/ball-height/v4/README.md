# Camera calibration audit and conditional height sensitivity (v4)

This is a diagnostic release, NOT an improved validated metric-height release. The attempted calibration did not pass the independent net consistency check. Do not treat v1–v3 or these ranges as measured heights.

## What was executed
1. Extract visible court boundary pixels from frame 0 using local intensity valleys and robust line fitting. Use the known 16 x 8 m ground rectangle. Near corners remain outside the image; optimize directly against visible line interiors rather than pretending extrapolated corners are observed control points.
2. Fit a physical perspective camera using OpenCV solvePnP/projectPoints and SciPy least_squares. Assume square pixels, zero skew, and no lens distortion. Scan 35 assumed principal points (cx 900–1080, cy 360–720). Of these, 25 pass positive camera-height, focal-length, and <2 px held-out line RMS gates.
3. Exclude every third sampled line pixel from fitting. Held-out RMS is 0.57–1.93 px across accepted cameras/lines. This checks image consistency on the same ropes in the same reference frame, NOT independent metric ground truth or net height accuracy. Rope curvature/slack and off-image corners remain limitations.
4. Try a manually located presumed upper net tape point at (1012,600), using the fact that the net-top line lies at x=8 m,z=2.20 m. The exact physical correspondence is NOT verified. Ground-compatible cameras miss it by 23–45 px. No camera passes the 6 px net-point gate. A separate presumed point (1002,220) is held out. Trying to fit more intrinsic parameters drives the principal point to a search boundary rather than giving a trustworthy calibration. Possible causes include an incorrect net annotation, camera-model assumptions, or inaccurate ground constraints. This does not establish that the court/net dimensions are wrong.
5. Audit image stability at six widely separated frames using ORB. A general background homography can move an extrapolated near-ground control by ~15 px in the review segment; this is not a direct physical displacement measurement.
6. Register each of the 300 review frames to frame 0 using a partial affine image transform fitted to static feature matches, excluding player boxes and broadcast overlays. Use held-out matches to check the transform. All 300 pass the chosen gates; median held-out inlier RMS ~1.04 px. This is a modest image motion correction, NOT new camera intrinsics or a 3D calibration. Affine registration cannot perfectly account for all parallax or projective motion.
7. Refit contact-bounded ballistic flights using corrected ball coordinates in reference-image space under each of the 25 ground-compatible camera scenarios. Pose is used to gate contact labels as in v3. No extrapolation beyond the review's selected contacts and no legacy height fallback enter the sensitivity envelope.

## Results and interpretation
- Same review: frames 6340–6639, 300 frames / 12 seconds.
- 142 frames have accepted flight estimates under at least five camera scenarios.
- Median min-to-max scenario height width: ~0.377 m; maximum ~0.736 m.
- Ranges are CONDITIONAL SENSITIVITY ENVELOPES, NOT statistical confidence intervals and NOT guaranteed error bounds. They omit many other sources of error, including contact errors, unknown distortion, uncertain net correspondence, and model inadequacy. Accepted-camera subsets can differ by frame.
- No net-validated camera was found. We therefore do not select a new single trustworthy metric height or ground projection. The diagnostic video deliberately omits the predicted ground footpoint.
- Previous v3 height is shown for comparison only, not as truth. Skeletal bone lengths, measured player heights, depth neural models, and ball shadows are not used by this calibration audit.

## Files
- `review_300_calibration_diagnostic.mp4`: same 300 frames, observed ball circle, previous v3 height and exploratory model range. All heights remain unvalidated.
- `calibration_audit.jpg`: reference-frame projected net ground line (cyan), predicted 2.20 m line (magenta), and uncertain manual net annotation points (red). The cyan/magenta curves are model outputs, not detected lines.
- `review_300_calibration_sensitivity.csv`: all 300 rows. Range cells are empty if fewer than five scenarios give an accepted fit. Empty is not zero. The previous-v3 columns retain their historical metadata; they do not describe the new scenario fits.
- `camera_ensemble.json`: all candidate camera parameters, ground residuals and net checks. The field selected_model_ids denotes ground-only sensitivity scenarios because no net-compatible models exist, not validated calibrations.
- `sensitivity_summary.json`, `review_registration.json`, `camera_stability.json`, `ground_lines.json`: audit data.

## Remaining prerequisite for metric validation
Identify an unambiguous non-ground control feature of known height and known court location, or supply independently calibrated camera intrinsics. The near pole padding and antenna must not be assumed to be the 2.20 m net top. Review the red annotation points before using them as controls. Once the correspondence is established, rerun calibration and reserve a different height feature for validation.
