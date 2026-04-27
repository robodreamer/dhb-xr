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

A coupled post-contact continuation was then trained:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_coupled_postcontact.yaml`

This branch starts from the affordance-gripper `steps_800` checkpoint and trains
the closed-state residual together with the generic contact/gripper heads while
keeping the open/reach backbone frozen. It improves the learned close geometry
but still does not solve full pick-place. With the evaluator gates still enabled
and `close/open=0.55/0.005`, step `600` closes at step `93`, keeps `0` IK
failures, and improves close-vector error to `36.0 mm`, but never reopens.

The run also exposed a harness issue: `scripts/starvla_dhb_live_checkpoint_sweep.py`
was not forwarding `--disable-gripper-gates`, so several files labeled
`nogates` were still using object-specific evaluator gates. That is now fixed
and covered by `tests/test_live_checkpoint_sweep.py`. With actual gripper gates
disabled and generic dense-state hysteresis (`close=0.55`, `open=0.40`,
debounce `2`), the same step `600` checkpoint closes at step `81` and reopens
at step `93`, with `57.6 mm` close-vector error and `129.2 mm` reopen
bowl-plate XY. This confirms the model can emit a release transition without
object gates, but the phase is not persistent enough and releases before
transport.

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
- the coupled post-contact branch improves gated close geometry to near the
  affordance gate, but true no-gate decoding still flickers from close to early
  release;
- the policy does not yet learn the demo side-grasp affordance tightly enough;
- post-close transport and release remain unvalidated.

The next most useful work is not more scalar threshold sweeping. It should
train a better event representation for side-affordance close and object-target
release, with a reach-preservation gate and an affordance-score gate after each
run.

## Phase-Persistence And Pre-Contact-Open Follow-Up

A phase-persistence continuation was trained from the coupled post-contact
checkpoint:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_phasepersistent.yaml`

The implementation adds a dense gripper-state margin loss on top of the
target-aware smoothness loss. With true gripper gates disabled and generic
`close=0.55/open=0.40` hysteresis, step `300` gives the best learned-gripper
sequence so far: close at step `80`, reopen at step `197`, `0` IK failures,
and minimum bowl XY `2.3 mm`. The score is still not a full-task pass:
close-vector error is `60.2 mm`, close XY is `64.7 mm`, reopen target XY is
`118.1 mm`, and final target XY is `116.8 mm`.

The diagnostic trace explains the remaining failure. The arm physically reaches
the bowl around steps `136-148`, but the dense closed probability becomes high
from step `80` onward. Raising the generic close threshold to `0.85` only
delays close to step `83` and still misses the side-grasp affordance
(`51.6 mm` close-vector error). Decoding from the contact-phase head is worse:
it closes at step `2` with `343.3 mm` close-vector error. Therefore the
current blocker is not gripper polarity, missing release labels, or another
execution threshold. It is learned contact-readiness calibration.

A follow-up pre-contact-open branch tested whether close-phase oversampling at
the end of the action horizon would provide enough hard open-before-contact
negatives:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_precontactopen.yaml`

This is a useful negative result. It improves the scored close-vector error to
`42.1 mm` and reaches `9.3 mm` minimum bowl XY, but immediately reopens
(`87 -> 89`) and introduces `72` IK failures. The heavy checkpoint/final-model
artifacts were removed to preserve disk; the lightweight rollout summaries and
score JSON remain for provenance.

Updated interpretation: the phase-persistence branch is the current best
learned-gripper evidence because it demonstrates a no-gate close-hold-reopen
sequence. The next model change should not be a scalar threshold or a sampler
only. It should create a contact-ready target that keeps approach open until
the predicted/observed state is truly at the grasp affordance, while separately
maintaining post-close hold/transport persistence.

A sparse generic contact-event branch was also tested after adding
`--model-gripper-decode-source` passthrough to the checkpoint sweep:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_contactevent.yaml`

This branch trains the existing `contact_head` with 3-way sparse event labels
and evaluates with `--model-gripper-decode-source contact`. It is also
negative. The contact decoder closes at step `3`, never reopens, and has
`339.3 mm` close-vector error. This confirms that the early-close problem is
not specific to the dense gripper-state head or the 3-class contact-phase head.
Current sparse event supervision still learns a chunk-position prior when the
rollout timing stretches under offset adaptation.

The working conclusion is now stricter: the next breakthrough likely requires
explicit contact-ready supervision or hard-negative calibration tied to the
current visual/proprioceptive state, not another head type trained on the same
short event windows. Offline simulator geometry can provide the labels, but
deployment must remain object-free.

A follow-up trace-scored run of the phase-persistence checkpoint makes the
same conclusion more quantitative. With true gripper gates disabled, generic
`close=0.55/open=0.40` hysteresis, and per-step trace recording, the rollout
closed at step `79`, reopened at step `233`, kept `0` IK failures, and reached
minimum bowl XY `4.5 mm`. The actual close-vector error was `63.2 mm`, but the
same trajectory later reached a best close-affordance error of `26.1 mm` at
step `123`. Therefore the arm trajectory already contains a substantially
better grasp opportunity than the learned gripper uses. The immediate model
gap is contact-readiness calibration from current perception/proprioception,
not another execution threshold or object-distance deployment gate.
