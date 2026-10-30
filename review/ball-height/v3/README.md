# Contact-aware height review (v3)

Video: ../review_300_frames_v3.mp4. Same frames 6340–6639, 300 frames at 25 fps.

Coverage: 280/300 versus 270/300 in v2. Of these, 116 frames use the new contact-aware fit; 164 retain legacy unvalidated estimates; 20 remain unknown. Across the full video, 8,244 frames have estimates; 2,894 use the new model. Other estimates are retained fallback results, not revalidated by pose.

Source contacts: branch claude/ball-contact-detection-vlmfcn, commit 412b38fea7fbf052293fdae0aa960f4b1477820d, contacts/ball_contacts.csv. Pose and video objects matched our checkout. The contact detector is automatic, not ground truth.

Of 405 contacts, 310 are usable proposed flight boundaries. We reject low-score events, events whose ball location disagrees with corrected boxes, weak pose regions, and head/chest labels incompatible with 2D pose. For other body parts, the existing contact label is used without a new limb-region gate. Contact audit records all decisions.

Fit single flights between consecutive selected contacts, sometimes jointly with their neighbor. Shared contact positions impose position continuity in joint fits while allowing velocities to change. Constant gravity is assumed. Do not fit through selected contacts, across shots, or removed time. Apply the previous physical and 4-pixel RMS fit acceptance limits. Flights with unresolved contacts may remain rejected or retain legacy fallback estimates. Automatic missed/false contacts remain possible.

Green skeletons show observed 2D pose. CONTACT + POSE means the fit uses pose-gated contact boundaries; it does NOT mean a metric 3D skeleton was reconstructed. LEGACY ESTIMATE identifies fallback results. Height is the ball center above the assumed sand plane. Contact region labels are automatic candidates.

No measured player heights or metric bone lengths were supplied. Previous rough player heights are not treated as facts. We do not assume feet are on the sand when players jump/bend; pose is used to gate contact labels, not to assign fixed head/chest heights. This version does not reconstruct metric 3D skeletons.

All metric heights remain UNVALIDATED because the approximate camera calibration was retained. More coverage and small pixel residuals are not proof of height accuracy. Changes on review frames with both old and new estimates: median absolute difference ~0.034 m, 90th percentile ~0.114 m, maximum ~0.419 m; these are model comparisons, not errors against truth.

Checks: 15,586 aligned frame rows, no generated estimates for missing observations, nonnegative estimated ball bottoms, known synthetic-flight solver recovery, and all 161 accepted flight fits bounded by selected contacts without crossing selected contacts, shots, or removed time. These checks do not establish real-world metric accuracy.
