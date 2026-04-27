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

## Current-Aligned Gripper Target Follow-Up

The StarVLA wrapper now supports explicit temporal alignment for contact and
gripper targets:

- `contact_target_alignment`
- `contact_phase_target_alignment`
- `gripper_state_target_alignment`

The default remains `future`, matching the action chunk horizon. A new
`current` mode broadcasts the first label in the sampled window across the
chunk, which tests whether the learned gripper/contact heads should classify
the current observation phase instead of learning future demo phase timing.
This implementation is covered by `tests/test_qwendhb_target_alignment.py`.

The fully current-aligned branch:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentgrip.yaml`

delayed closure to the real bowl-contact region. With true gripper gates
disabled, it closed at step `127`, reached `6.9 mm` minimum bowl XY, and the
close snapshot was only about `[14.9, -3.1, 17.0] mm` EE-minus-bowl. However,
it immediately reopened at step `129`, introduced `81` IK failures, and still
failed the affordance score (`50.5 mm` close-vector error, `118.1 mm` reopen
target XY). This is an informative negative result: current-state labels fix
the early-close prior, but they convert the dense gripper state into a
short-lived close pulse instead of a persistent hold signal.

The hybrid current-phase branch:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentphase.yaml`

keeps dense gripper-state supervision future-aligned while making only the
contact/contact-phase targets current-aligned. This preserves no-gate
stability better than full current alignment: the rollout closes at step `77`,
reopens at step `233`, has `0` IK failures, reaches `4.7 mm` minimum bowl XY,
and has path efficiency `0.75`. It still closes too early for the
side-affordance metric: actual close-vector error is `69.6 mm`, while the same
trajectory later reaches a `25.9 mm` best close-affordance opportunity at step
`122`.

Updated conclusion: current observation conditioning is necessary, but the
target design must decouple "open now because we are not contact-ready" from
"stay closed after contact through transport." The next training change should
keep future-aligned dense gripper-state labels for hold/release persistence
while adding a current-aligned open/contact-ready margin over the actually
executed prefix of each chunk. Full current broadcast is too destructive; pure
future labels fire too early.

## Current-Prefix And Contact-Ready Split Follow-Up

The `current_prefix` alignment branch tested a softer version of full current
broadcast:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentprefix.yaml`

It repeats the current label over only the first four supervised prefix steps,
then keeps the remainder future-aligned. This is also negative. With true
gripper gates disabled, dense gripper-state decoding closes at step `80`,
reaches `0.34 mm` minimum bowl XY, and has `0` IK failures, but the close
vector error remains `60.2 mm` and final target XY remains `112.1 mm`.
Contact-phase decoding from the same branch closes even earlier at step `74`
and never reopens. The prefix target alone suppresses neither the early-close
prior nor the missing release/transport behavior enough to close the claim.

The next branch explicitly decouples close-readiness from hold/release state:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready.yaml`

Here the generic contact head is trained as a current-observation
contact-readiness signal from `dhb.gripper_state`, while the dense
`gripper_state_head` remains future-aligned for persistence. Rollout uses the
object-free hybrid decoder `contact_gripper_state`: the contact-readiness head
controls only the open-to-closed transition, and the dense gripper-state head
controls closed-state persistence and reopening.

This is the first clean learned no-gate close result near physical contact.
With generic `close=0.55/open=0.40` hysteresis, true gripper gates disabled,
and `0` IK failures, the rollout closes at step `103`, reaches `10.2 mm`
minimum bowl XY, and scores `37.1 mm` close-vector error with `19.0 mm` close
XY. The close snapshot is approximately `[-12.2, -14.6, 21.9] mm`
EE-minus-bowl. This beats the previous learned no-gate actual-close score and
shows that current-readiness supervision fixes the major early-close failure.

The result is still not a full pick-place success. The model never reopens,
and final bowl-plate XY is `118.0 mm`. The remaining gap is therefore no
longer generic "gripper timing"; it is post-close transport/release under a
persistent learned closed state. The next useful branch should preserve the
currentready close-readiness head and focus release/transport supervision on
the dense gripper-state/contact-phase heads, with rollout decisions still made
only from learned model outputs rather than object-distance gates.

## Seed-Controlled Evaluator Follow-Up

A follow-up exposed an evaluation-control issue: the live rollout script used
the training config's `seed` directly for the LIBERO environment, and previous
continuation configs used different seeds. That made single-run comparisons
between `currentready`, `currentready_release`, and `stateonly_release`
partially seed-confounded. The evaluator now supports `--rollout-seed`, seeds
Python/NumPy/Torch/CUDA plus the LIBERO environment, and records both
`config_seed` and `rollout_seed` in every summary.

Under the controlled `rollout_seed=70` contract, the currentready checkpoint
with generic `close=0.55/open=0.40` closes at step `91`, never reopens, keeps
`0` IK failures, and reaches a best close-affordance opportunity of `3.3 mm`
at step `110`, but the actual close-vector error is `36.4 mm`. Raising the
generic close threshold to `0.58` delays close to step `101` and improves the
close snapshot to about `16.9 mm` XY from the demo close affordance, but it
reopens prematurely at step `121`, gives `9` IK failures, and remains about
`141 mm` bowl-plate XY from target. This is useful calibration evidence, not a
full-task result.

The state-only release continuation:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_stateonly_release.yaml`

changes only the four `gripper_state_head` tensors relative to currentready.
With controlled `rollout_seed=70`, it closes at step `87`, never reopens,
keeps `0` IK failures, and reaches a best close-affordance opportunity of
`3.0 mm` at step `115`, but actual close-vector error is `42.0 mm` and final
target XY is `129.4 mm`. A higher-threshold diagnostic
(`close=0.60/open=0.50`, debounce `4`) suppresses closure entirely and causes
late IK failures, so scalar hysteresis is not the breakthrough.

Updated interpretation: the arm trajectory repeatedly contains near-perfect
side-grasp opportunities under the deterministic evaluator, but the learned
contact/release heads do not choose the right temporal phase. The strongest
next model direction is a seed-controlled, reach-preserving training branch
that preserves currentready contact-readiness while adding explicit held-object
transport and release supervision. Full pick-place should remain unclaimed
until it passes close-affordance and object-target release metrics under a
fixed rollout-seed matrix, not a single uncontrolled video.

## Currentready Transport-Release Continuation

The next branch followed that direction:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_transport_release.yaml`

It warm-starts from currentready, freezes Qwen, the base action path, the
currentready contact-readiness head, and the phase head, then trains only the
closed-state residual adapter plus dense `gripper_state_head`. The trainer
confirmed the intended scope with `4.464M` trainable parameters.

This branch is a useful partial but not a breakthrough. At `steps_200` under
controlled `rollout_seed=70`, true no-gate hybrid decoding preserves the
currentready close behavior and learns late release: close step `103`, reopen
step `221`, `0` IK failures, minimum bowl XY `10.2 mm`, close-vector error
`37.1 mm`, and final target XY `114.4 mm`. This is better than no release, but
object-target transport remains essentially unsolved.

By `steps_400`, the same branch regresses into the known flicker mode:
close step `85`, reopen step `89`, `12` IK failures, `46.6 mm` close-vector
error, and `136.6 mm` final target XY. Therefore the branch identifies the
right coupling direction but overtraining quickly destabilizes the gripper
phase. The next attempt should either early-stop around `steps_200` and focus
on a stronger closed-transport target, or add an explicit hold-through-transport
loss/gate in training so release cannot become a short local pulse.

## Closed-Transport Action-Only Follow-Up

An action-only closed-transport continuation was trained from the useful
transport-release `steps_200` checkpoint:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_closedtransport_actiononly.yaml`

This branch freezes all gripper/contact heads and trains only the
`action_closed_residual_adapter` (`0.267M` trainable parameters). The goal was
to isolate whether the remaining failure was simply weak post-close motion.

The controlled `rollout_seed=70` results are negative. Step `150` closes at
step `91`, never reopens, keeps `0` IK failures, reaches a best
close-affordance opportunity of `6.9 mm`, but ends `123.9 mm` from the target.
Step `300` restores the same close/reopen timing as the transport-release
partial (`close_step=103`, `reopen_step=237`, `0` IK failures) and scores
`37.1 mm` close-vector error with `19.0 mm` close XY, but final target XY is
still `112.9 mm`.

This rules out pure closed-residual action MSE as the breakthrough. The
transport failure is not caused by lack of post-close actuator motion alone;
the model needs a more explicit object-level transport/release supervision
signal during training. That signal can be derived offline from simulator/demo
geometry, but rollout should continue to use only learned predictions from
images, language, state history, and DHB/action heads, with no object-distance
deployment gate.

## Object-Geometry Transport Follow-Up

The LIBERO-to-LeRobot converter now also exports generic offline geometry
features for transport supervision:

- `dhb.object_to_eef_pos`
- `dhb.target_to_eef_pos`
- `dhb.object_minus_target_pos`
- `dhb.transport_phase`

These labels are simulator/demo-derived training targets only. The live policy
still receives images, language, state history, and learned head outputs; it
does not consume object-distance gates at deployment.

The first object-transport branch used a learned
`dhb.object_minus_target_pos` auxiliary head to condition the closed residual:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objecttransport.yaml`

This branch is a useful negative result. Step `150` preserves smooth motion
and the currentready close timing (`close_step=103`, `0` IK failures,
`37.1 mm` close-vector error), but still ends over `110 mm` from the plate and
does not reliably release across repeated traces. Step `300` regresses into an
early close/open pulse (`87 -> 89`) and late IK failures. Rollout diagnostics
show why: the learned object-target vector is almost constant across chunks,
so it is not giving the closed residual meaningful placement information.

A second branch supervised the richer object-pair target
`[object_to_eef_pos, target_to_eef_pos]`. Reading this target from the frozen
action-slot features also collapsed to near-constant predictions and worsened
the live result: step `150` never reopens and finishes `128.6 mm` from target;
step `300` reopens at step `189` but still releases/finalizes around
`133 mm` from the plate.

The follow-up implementation added an opt-in separate cross-attention geometry
readout:

`object_target_readout: cross_attention`

The context projection into the closed residual is zero-initialized so the
warm-start action behavior is preserved until the geometry path learns. This
readout does produce non-constant object-pair predictions in live rollout, but
the task result is still negative. Step `150` overfits into the close/open
pulse (`87 -> 89`), and step `300` returns to stable close without release
(`close_step=103`, `0` IK failures, `117.9 mm` final target XY).

Updated interpretation: object-level geometry supervision is now plumbed and
diagnosable, but adding it as an auxiliary action context does not by itself
solve held-object transport/release. The next breakthrough should decouple the
phase decision more explicitly: current-ready close, persistent hold/transport,
and a learned release-ready signal trained from offline object-target geometry.
Do not continue scalar object-target loss or residual-weight sweeps without a
release-ready/persistence objective.

## Object-Target Release Label Implementation

The next branch now implements that release-ready/persistence objective. The
converter exports generic object-target release labels computed offline from
the object-minus-target vector:

- `dhb.object_release_ready`
- `dhb.object_release_contact`
- `dhb.object_release_event`
- `dhb.object_release_phase`
- `dhb.object_release_gripper_state`
- `dhb.object_hold_transport`

These labels are training supervision only. The policy deployment contract
remains object-free: live rollouts use images, language, state history, DHB/
action context, and learned contact/gripper heads, with true gripper gates
disabled for claim-level evaluation.

`QwenDHB` now has configurable target fields for contact, contact phase, and
dense gripper state. The active branch:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml`

preserves the currentready close-readiness head and routes
`contact_phase_head` plus `gripper_state_head` to the new object-release
labels. The refreshed affordance-transport dataset has `500` episodes and
`62,250` frames. A one-step StarVLA smoke passes, so the next evidence point is
the seed-controlled embodied rollout at `steps_150/300`.
