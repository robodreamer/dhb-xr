# 2026-04-27 Affordance-Gripper Retrain Summary

## What Changed

The regenerated affordance-labeled dataset was used for two follow-up training
runs:

- `dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_fromxy500`
  trained the side-affordance action residual from the reach/transport
  checkpoint.
- `dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_grip_fromaff500`
  froze the action path and trained the generic learned gripper/contact heads.

The policy path still uses no object-distance gripper gate at rollout. Demo
geometry is used only offline for labels and post-hoc scoring.

## Results

The action residual preserved the corrected reach behavior. The step-500 reach
sweep over `0:0`, `0:+3`, `0:-3`, `1:+3`, and `1:-3` reached `5/5`, with
average minimum bowl XY `38.2 mm` and `0` IK failures:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_fromxy500/live_reach_corrected_raw_y135_z115_max140_5case_step250_500_commanded.json`

The action-only affordance checkpoint still did not close under the learned
gripper-state decoder at threshold `0.55`, even though it reached close to the
bowl. Its best close-vector affordance error was `36.1 mm` around step `93`,
but no close command fired.

The gripper-only continuation made the learned gripper-state head actionable:
at step `800` with `close_threshold=0.55`, it closed at step `83`, had `0` IK
failures, and reached minimum bowl XY `6.7 mm`. However, close happened high
and early relative to the demo side-grasp vector: close-vector error was
`51.8 mm`, with best possible close-vector error `25.0 mm` later at step
`113`. It never reopened and ended about `116 mm` bowl-plate XY from target.

A generic confidence calibration `close_threshold=0.70` delayed close to step
`87` and improved the close-vector error to `42.3 mm`, still outside the
affordance-pass threshold. Higher thresholds suppressed closure in the actual
closed-loop trajectory.

Demo gripper replay remains a bad control under non-zero offset: it closed at
step `42`, with `194.8 mm` close-vector error and `112` IK failures.

The learned contact-phase event decoder exposed a real implementation bug:
3-class event decoding was using argmax and ignoring the configured threshold.
That is now fixed so close/release require thresholded event confidence. After
the fix, the current contact-phase head still closes too early (`step 39`,
`205.6 mm` close-vector error), so the decoder was necessary but not sufficient.

An event-BCE continuation was then trained from the gripper step-800 checkpoint
to test whether treating the contact phase as independent thresholded close/open
events would fix the timing:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_eventbce.yaml`

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_eventbce_fromgrip800/checkpoints/steps_800_pytorch_model.pt`

The training run completed, but the live no-gate contact-phase rollouts remain
negative. With `close_threshold=0.55`, the model closes at step `3`, with
`339.3 mm` close-vector error. Higher generic thresholds delay closure and
remove IK failures, but still miss the side-grasp affordance: `0.80` closes at
step `74` with `79.7 mm` close-vector error, and `0.90` closes at step `75`
with `76.3 mm` close-vector error. None of these runs reopens or reaches
object-target placement. The trace shows close probability ramping inside
nearly every chunk, so the event head is still learning a relative-in-chunk
close prior rather than a sufficiently perception/state-conditioned contact
event.

A release-focused dense-gripper continuation was also tested:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip.yaml`

This run oversampled terminal/release chunks while freezing the action path.
It gives a useful partial signal but not a full-task solution. At step `300`,
the no-gate dense gripper decoder learns to reopen (`close_step=137`,
`reopen_step=165`), but the close is center-contact rather than the demo
side-grasp affordance (`54.4 mm` close-vector error), release is still far from
the plate (`119.9 mm` bowl-plate XY), and post-release IK failures rise to
`124`. Lowering the generic close threshold to `0.45` improves close-vector
error to `41.9 mm`, but causes an immediate close/open pair
(`close_step=87`, `reopen_step=89`) and still fails placement. By step `600`,
the same branch overfits to chunk-relative gripper pulses and no longer
reopens, although it keeps `0` IK failures and reaches `6.5 mm` minimum bowl
XY. This suggests release supervision is present and learnable, but the current
head needs phase persistence and transport coupling rather than more
release-heavy sampling alone.

To address the flicker directly, a target-aware gripper-state smoothness loss
was added and tested in:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip_smooth.yaml`

The loss penalizes adjacent closed-probability changes only when the supervised
gripper state does not change. The implementation is useful, but this first
configuration is still negative for full pick-place. Step `200` and step `400`
both keep `0` IK failures and reach extremely close center contact
(`0.2-0.6 mm` minimum bowl XY), but neither reopens. Close-vector error remains
`46.8-49.0 mm`, so the branch stabilizes gripper state but does not solve the
side-affordance grasp or object-target release.

## Interpretation

The current evidence is positive for the specific claim that learned gripper
state is better than demo replay under delayed offset rollout. It is not yet
positive for full pick-place. The active blocker is now more specific:

- pre-contact reach is stable and smooth enough for paper evidence;
- learned gripper-state can close without object-distance heuristics;
- contact-phase event logits are still poorly calibrated and overfit to
  chunk-relative timing even under BCE event training;
- release-heavy dense-gripper training can produce a no-gate reopen at early
  checkpoints, but not a stable close-hold-transport-release sequence;
- target-aware gripper-state smoothness suppresses flicker and IK collapse, but
  in the tested weighting it also suppresses release;
- the policy does not yet learn the demo side-grasp affordance tightly enough;
- post-close transport and release remain unvalidated.

The next most useful work is not more scalar threshold sweeping. It should
train a better event representation for side-affordance close and object-target
release, with a reach-preservation gate and an affordance-score gate after each
run.
