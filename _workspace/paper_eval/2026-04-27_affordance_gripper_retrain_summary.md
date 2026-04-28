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

## Object-Target Release Rollout Result

The first object-target release run completed:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_fromtransport200`

The initial checkpoint sweep accidentally used `--max-policy-steps 0`, which
caps full-task rollout to the nominal demo length (`84` policy steps for the
episode-1 case). That was an invalid full-task comparison because the
controlled baselines often close after the nominal demo horizon under the
`+3 cm` offset. The rollout scripts now emit an explicit warning and record
`demo_length_cap_warning` when this happens. Claim-level full-task sweeps
should set a long horizon such as `--max-policy-steps 240`.

With the corrected fixed-seed, true-no-gate horizon (`rollout_seed=70`,
episode `1`, offset `+3 cm`, `max_policy_steps=240`), the object-release
branch gives a useful partial but not a full pick-place pass:

- `steps_150`: close step `86`, reopen step `201`, minimum bowl XY `30.7 mm`,
  close-vector error `41.8 mm`, reopen target XY `134.1 mm`, final target XY
  `122.6 mm`, and `12` IK failures.
- `steps_300`: close step `82`, reopen step `153`, minimum bowl XY `1.8 mm`,
  close-vector error `51.7 mm`, reopen target XY `121.1 mm`, final target XY
  `116.1 mm`, and `11` IK failures.

The result confirms that the learned no-gate policy can now emit a
close-hold-reopen sequence from offline object-release supervision, but
release still happens far from the plate and late IK failures remain.

Small generic hysteresis diagnostics are informative but not a breakthrough.
At `steps_300`, `close=0.58/open=0.40` reopens later at step `201` and the
same trace contains a very good best close-affordance opportunity (`5.2 mm` at
step `104`), but actual close still misses the affordance and final target XY
is `111.6 mm`. At `close=0.60/open=0.60/debounce=1`, actual close improves to
`37.9 mm` close-vector error and `12.8 mm` close XY, with reopen at step
`139`, but final target XY remains `110.6 mm` and IK failures rise to `64`.
Direct `contact_phase` decoding is worse: close step `75`, reopen step `87`,
`73.7 mm` close-vector error, `129.2 mm` reopen target XY, and `75` IK
failures.

Updated interpretation: object-target release labels are the right conceptual
direction because they create a learned release transition without deployment
object gates. The remaining gap is not whether release labels exist; it is
held-object transport and release calibration. The next model should preserve
the currentready close-readiness path, keep the hybrid decoder, and strengthen
post-close transport/release supervision rather than switching to
`contact_phase` decoding or relying on scalar threshold sweeps.

## Current-Phase Release Decoder Result

The latest implementation adds an object-free hybrid decoder:

`--model-gripper-decode-source contact_release_phase`

This decoder closes from the already-validated currentready contact head and
opens only from the learned release class of `pred_contact_phase_probs`. It is
not an object-distance gate; object geometry is used only in offline labels and
affordance scoring.

The current-phase release continuation was trained from:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_fromtransport200/checkpoints/steps_300_pytorch_model.pt`

with config:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_currentphase.yaml`

It freezes the trajectory, currentready contact, and dense gripper-state paths,
then trains the release phase head against current-aligned
`dhb.object_release_phase`. The corrected fixed-seed, true-no-gate,
long-horizon sweep gives:

- `close=0.55/open=0.55/debounce=2`, step `150`: close step `87`, no reopen,
  minimum bowl XY `32.2 mm`, close-vector error `40.0 mm`, final target XY
  `126.7 mm`, and `0` IK failures.
- `close=0.55/open=0.55/debounce=2`, step `300`: close step `83`, no reopen,
  minimum bowl XY `6.4 mm`, close-vector error `49.1 mm`, final target XY
  `111.8 mm`, and `6` IK failures.
- `close=0.60/open=0.55/debounce=2`, step `300`: close step `99`, no reopen,
  close XY `16.8 mm`, close-vector error `35.8 mm`, final target XY
  `140.3 mm`, and `28` IK failures.
- `close=0.62/open=0.55/debounce=2`, step `300`: no close, minimum bowl XY
  `6.7 mm`, and `88` IK failures.

The trace diagnostic explains why the release decoder does not reopen: the
learned release probability remains near zero (`max ~= 1.4e-4`) while the bowl
is still more than `110 mm` from the plate. That is conceptually the right
behavior for a release-ready head. The remaining blocker is now precise
side-grasp closure plus stable closed-object transport. The release head should
not be forced to open until the transported object actually reaches the target
region in the learned state.

Updated claim boundary: we have a validated learned no-gate close/reopen
mechanism and a validated release-label contract, but the current full-task
result is still a partial. The next breakthrough should optimize close
affordance precision and held-object transport before further release-head
work.

## Close-Contact Recalibration Negative Result

A follow-up branch tested whether the currentready contact head should be
trained directly from explicit `dhb.close_contact` geometry labels instead of
the default dense gripper-state target:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_objectrelease.yaml`

The implementation also fixed the StarVLA wrapper so `contact_pos_weight` is
actually passed into BCE contact loss. The trainer froze everything except
`contact_head` (`3.149M` trainable parameters). Training loss collapsed quickly
(`loss_contact` from `1.74` at step `50` to `0.0039` at step `300`), but the
embodied result regressed badly: both `steps_150` and `steps_300` close at
step `1`, far above and far from the bowl, then never recover (`180 mm`
minimum bowl XY, `76` IK failures).

Interpretation: the explicit close-affordance label is useful, but this
particular sampling/weighting recipe is wrong. High close-phase oversampling
with `gripper_event_target_pos=0` plus a large positive weight produced an
always-close classifier. Do not reuse this recipe as-is. The next close
precision attempt needs balanced hard negatives from "near bowl but too high /
wrong side" and lower positive weighting, or a calibrated margin/ranking loss,
not positive-only close-window fitting.

## Balanced Close-Contact Follow-Ups

Two safer close-contact recalibration branches were then tested from the same
current-phase object-release checkpoint:

- `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_balanced_objectrelease.yaml`
- `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_xattn_objectrelease.yaml`

The balanced head-only branch lowers close-positive oversampling
(`gripper_phase_chunk_prob=0.35`), samples from `dhb.close_contact` directly,
does not force positives into slot `0`, and uses `contact_pos_weight=2.0`.
This avoids the previous always-close collapse, but it under-calibrates the
contact probability. At the default `0.55` threshold, both `steps_150` and
`steps_300` never close despite reaching about `6.8 mm` minimum bowl XY and
then accumulating `88` IK failures. A diagnostic generic threshold of `0.15`
for `steps_150` closes at step `87` with `0` IK failures, but the affordance
score is still only the previous partial level: `40.0 mm` close-vector error,
`42.7 mm` close XY, no reopen, and `126.8 mm` final target XY.

The cross-attention branch unfreezes `contact_cross_attn` in addition to
`contact_head` (`7.347M` trainable parameters). It also avoids immediate close,
but collapses close confidence even further: at the default threshold neither
checkpoint closes, and the maximum rollout close probability is only about
`0.0064` for `steps_150` and `0.0017` for `steps_300`.

Updated interpretation: BCE close-contact relabeling alone is not the
breakthrough. The positive-only recipe saturates to always-close; balanced BCE
and contact-readout adaptation swing to never-close or low-confidence
under-calibration. The remaining close-affordance gap likely requires labels
or losses built from rollout-state hard negatives: "near bowl but too high /
wrong side / not settled" must be contrasted directly against true
side-grasp-ready states. This points toward closed-loop data aggregation,
offline perturbation/replay around the observed rollout states, or a pairwise
margin/ranking close-readiness objective rather than another BCE sampler sweep.

## Closed-Loop Contact Mining and Margin Follow-Up

To make the failure mode explicit, `scripts/libero_rollout_contact_miner.py`
now mines recorded rollout traces against the demo-derived close-affordance
reference. It labels each policy step as `positive`, `missed_positive`,
`hard_negative`, or `background` using only offline trace geometry and the
model's recorded close probabilities.

Mining the current recorded traces gives:

- currentready/current-phase trace: `45` positives, `0` missed positives,
  `170` hard negatives, best close-affordance error `4.1 mm` at step `103`,
  and maximum hard-negative close probability about `0.6` at step `151`.
- currentphase gripper-state trace: `17` positives, `0` missed positives,
  `142` hard negatives, best close-affordance error `25.9 mm` at step `122`,
  and maximum hard-negative close probability about `0.9` at step `107`.
- balanced close-contact trace: `0` positives, `0` hard negatives, best
  close-affordance error `54.2 mm`; this branch is primarily a reach/trajectory
  miss on that offset, not merely a hidden gripper threshold miss.

This confirms the best active partial is not lacking close confidence in
general; it is over-confident at the wrong closed-loop states. A contact-margin
loss was added to `DHBActionHead` and StarVLA's wrapper, then tested with:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_margin_objectrelease.yaml`

The margin run trained only `contact_head` (`3.149M` trainable parameters) from
the current-phase checkpoint and passed the one-step smoke with
`loss_contact_margin` present. However, the fixed-seed, true-no-gate h240 sweep
is also negative:

- `steps_150`: no close at either `0 cm` or `3 cm`; average minimum bowl XY
  `63.1 mm`, average IK failures `75`.
- `steps_300`: no close at either `0 cm` or `3 cm`; average minimum bowl XY
  `63.1 mm`, average IK failures `75`.
- rollout close confidence is strongly suppressed (`max close prob ~= 0.187`
  at `steps_150`, `~= 0.043` at `steps_300`).

Updated conclusion: demo-only contact-head recalibration is exhausted. The
failure is a closed-loop distribution problem: the model must see hard negative
rollout states, not only nominal demo negatives, if we want learned gripper
timing without deployment object gates. The next training intervention should
capture or reconstruct observation/state samples from rollout states and train
a contrastive or DAgger-style close-readiness objective against those states.

## Policy-Trace Hard-Negative Dataloader Follow-Up

The rollout trace export is now connected back into StarVLA training through a
new dataset path:

`vendor/starVLA/starVLA/dataloader/policy_trace_contact_datasets.py`

The loader reads contact-miner policy manifests plus saved per-replan NPZ
chunks and returns normal `QwenDHB` samples: agent-view image, language,
state-history window, dummy action chunk, `dhb.close_contact`, and
`dhb.action_valid_mask`. It can either train only from policy traces or mix
those traces with the existing LeRobot demo dataset. The configured training
branch is:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_policytrace_hardneg.yaml`

Validation passed:

- policy-trace dataset unit tests pass;
- `pytest tests/test_policy_trace_contact_dataset.py tests/test_libero_rollout_contact_miner.py tests/test_action_head.py -q`
  passes (`46 passed`);
- the actual QwenDHB train-step smoke passes and logs both `loss_contact` and
  `loss_contact_margin`;
- the compact 300-step continuation trains under the non-DeepSpeed
  `train_starvla.py` path because this machine lacks `CUDA_HOME`/`nvcc`.

The fixed-seed, true-no-gate h240 sweep is a useful negative result:

- `steps_150`: no close at `0 cm` or `3 cm`; average minimum bowl XY
  `63.1 mm`; average IK failures `75`.
- `steps_300`: no close at `0 cm` or `3 cm`; average minimum bowl XY
  `63.1 mm`; average IK failures `75`.
- rollout close confidence is suppressed by construction: max close probability
  is about `0.15` at step `150` and below `0.03` at step `300`.

Interpretation: the new data path works, but the current manifest content is
one-sided. The two exported policy traces provide `242` hard-negative
supervised steps and `0` closed-loop positive steps. Mixing them with nominal
demo positives is not enough to preserve a real close trigger, because the
closed-loop negative observations dominate the deployment distribution. The
next dataset must include true or near-true closed-loop positive contact states,
either by collecting traces that actually pass the close-affordance gate or by
simulator/world-model perturbation around demo close-affordance states. This is
now the most concrete version of the paper story: use simulation/replay to
populate the missing spatially diverse positive and negative contact states,
then train the learned DHB contact token without deployment object gates.

## Demo-Positive Policy-Trace Follow-Up

A root-cause bug was found in the first demo-positive policy-trace contact
run: the balanced config used `contact_target_alignment: current`, which
broadcast the first label in each chunk. Demo-contact chunks often look like
`[0, 0, 0, 0, 0, 0, 1, 1]`, so the old alignment silently converted them into
all-open targets. Switching the balanced contact config to
`contact_target_alignment: future` makes the late close positives active.

New training data and validation:

- `scripts/libero_demo_contact_manifest_export.py` exports LIBERO demo close
  events as the same NPZ+manifest format used by policy traces.
- The full task-0 export has `500` chunks from `50` demos, with `1,650`
  positive and `2,350` negative supervised contact steps.
- Combined with the two closed-loop hard-negative manifests, the balanced
  contact dataset has `562` rows, `4,242` supervised steps, `1,650` positives,
  and `2,592` negatives.
- Regression tests now cover late future contact positives, and the focused
  probe confirms the positive sample loss is active (`loss_contact ~= 0.62`
  instead of the previous near-zero loss).

Corrected contact-head run:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_demopos_policytrace_balanced.yaml`

The corrected 300-step continuation trains only the contact path from the
currentready/current-phase trajectory checkpoint. A held-out positive demo
chunk now reaches max close probability about `0.54`, while a negative chunk
stays around `0.18-0.32`. In closed-loop rollout, contact confidence is lower:
with neutral action scaling, max close probability is only `0.43-0.46`, so the
head still needs either better calibration or more rollout-positive data.

Execution calibration matters. The strong reach behavior depends on the
previously validated task-agnostic direct-action scale
`x=0.85, y=1.55, z=1.15`; using neutral `1,1,1` regresses average closest bowl
XY to about `63 mm`. With the calibrated scale and the corrected contact head:

- At threshold `0.50`, no close fires, but reach is strong: `22.9 mm` XY on
  `0 cm` and `6.9 mm` XY on `3 cm`.
- At threshold `0.45`, close fires in both cases. The `3 cm` offset case closes
  near the bowl (`~18.9 mm` XY, `21.9 mm` Z), while the `0 cm` case closes too
  far away (`~66.9 mm` XY). Neither solves the full place task.

Current interpretation: the alignment bug is fixed and the learned contact
token can now train from late chunk positives, but the full-task blocker is no
longer just gripper signal plumbing. The active gap is a combined
closed-loop-calibration problem: preserve the calibrated reach profile, train
contact confidence on true rollout-positive side-grasp states, and improve
post-grasp transport/release. Deployment remains object-gate-free; the only
thresholds used above are learned probability thresholds.

## Rollout-Positive Shuffle Follow-Up

The future-aligned contact baseline was used to collect calibrated no-gate
policy traces with the validated direct-action scale
`x=0.85, y=1.55, z=1.15`, fixed `rollout_seed=70`, cases `0:0 cm` and
`1:3 cm`. Both traces reached the bowl without issuing a close: minimum bowl XY
was `22.9 mm` for case `0` and `7.0 mm` for the `3 cm` offset case. Mining
with the original strict `30 mm` positive tolerance found no rollout positives;
relaxing only the offline annotation tolerance to `45 mm` produced `111`
missed-positive supervised steps from the offset trace and no new hard
negatives.

An implementation issue was also found in the StarVLA policy-trace dataloader:
the training loader was deterministic because the upstream `shuffle=True` line
was commented out. For short continuation runs, appended rollout-positive rows
could be skipped entirely. The dataloader now honors
`policy_trace_shuffle: true`, and the active rollout-positive config repeats
the newly mined positives first while still shuffling the dataset.

Corrected run:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_rolloutpos_tol45_balanced.yaml`

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_rolloutposfirst_shuffle_tol45_fromfuturealign300`

The `steps_400` checkpoint proves the learned contact token is now active under
true no-gate deployment. At close threshold `0.50`, case `0` closes at step
`87` with close snapshot EE-minus-bowl about `[-29.1, +0.5, +61.5] mm`; the
`3 cm` offset case closes at step `99` with about `[-8.8, -15.3, +29.2] mm`.
Average minimum bowl XY is `16.4 mm` and average IK failures drop to `21`, but
the task still does not complete. At threshold `0.55`, the offset case improves
to a close snapshot of about `[+3.7, -5.7, +19.3] mm`, while case `0` worsens
to a high close; no full success is observed.

Interpretation: the token path is now trainable and object-gate-free, and the
offset example is close to a plausible side-grasp close in XY/Z. The remaining
gap is label quality and post-grasp behavior. A `45 mm` rollout-positive gate
is useful for bootstrapping, but it is too broad for claim-level grasp
precision; the next dataset should add side-grasp-ready positives and explicit
near-bowl-but-too-high/too-central negatives, still using geometry only for
offline supervision and not for deployment gating.

## Side-Grasp Hard-Negative Calibration Follow-Up

The rollout contact miner now supports component-wise positive tolerances and
near-object hard negatives. This lets the offline labels distinguish
"geometrically close to the demo side-grasp vector" from broad near-bowl states
that are high, central, or otherwise likely to push the bowl instead of grasping
it. These labels are still training/evaluation artifacts only; rollout remains
learned-token/no-gate.

The first strict side-grasp manifest used the demo close vector with a `45 mm`
overall tolerance, component caps of `[30, 35, 25] mm`, and near-object hard
negatives for non-positive states within `100 mm` XY and `0-90 mm` Z. It
produced `240` chunks, `960` steps, `610` supervised steps, only `9` positives,
and `601` hard negatives:

`_workspace/paper_eval/2026-04-27_policy_trace_sidegrasp_v1_contact_manifest.jsonl`

Strict continuation:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1.yaml`

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_fromrolloutpos400`

The strict `steps_300` checkpoint is a useful negative. With calibrated
direct-action scale `0.85, 1.55, 1.15`, true gripper gates disabled, threshold
`0.55`, and fixed `rollout_seed=70`, it never closes in either the `0 cm` or
`3 cm` case. Live max close probability is only about `0.18-0.19`. This proves
the stricter negatives are active but over-dominate the very small positive
set.

A lighter continuation from the same rollout-positive checkpoint reduced the
margin weight and repeated demo positives:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_light.yaml`

It restored no-gate close activation but did not solve the task. At threshold
`0.55`, case `0` closes at step `87` with EE-minus-bowl about
`[-28.8, +0.6, +61.6] mm`, while the `3 cm` case closes at step `103` with
about `[-12.2, -14.6, +21.9] mm`. Average minimum bowl XY is `15.2 mm` and
average IK failures are `23.5`, but there is still no successful
grasp/transport/release.

Current interpretation: the learned contact token path is functioning, but
label balance is the limiting factor. Broad rollout positives make the token
fire; strict side-grasp hard negatives make it stop firing; lighter
calibration recovers activation without sufficient side-grasp precision. The
next useful data intervention is not another scalar threshold sweep. It is to
generate many balanced side-grasp-ready rollout positives, paired with
near-bowl/high-or-center negatives, through simulator replay, DAgger-style
correction, or the planned world-model augmentation path.

## Physical-Close Scoring Correction And Best-Positive Relabeling

The affordance scorer had a metric mismatch: it compared the rollout command
close step against the demo `close_phys` reference. LIBERO gripper closure has
actuator lag, and the rollout traces already record `observed_state_gripper`.
The scorer now reports both command close and physical close, where physical
close is the first observed gripper qpos crossing `0.015` after the close
command.

This correction is important. In the best non-zero-offset trace, the model
commands close at step `99`, but the observed gripper physically closes at step
`101`. The command-time side-grasp vector error is `35.5 mm`; the physical
close vector error is `9.2 mm`, which passes the `30 mm` side-grasp gate:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegrasp_bestposv2_fromrolloutpos400/bestposv2_thr050_ep1_offset3_video_summary.json`

Video:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegrasp_bestposv2_fromrolloutpos400/bestposv2_thr050_ep1_offset3.mp4`

The data intervention that produced this trace added bounded corrective
best-positive relabeling to the miner. The side-grasp v2 manifest keeps the
strict component caps and near-object hard negatives, but also relabels the
closest near-success rollout states as positives if their vector error is below
`58 mm`. The resulting manifest has `610` supervised steps, `43` positives,
and `567` negatives:

`_workspace/paper_eval/2026-04-27_policy_trace_sidegrasp_bestpos_v2_contact_manifest.jsonl`

The 200-step continuation restores useful no-gate contact timing on the
non-zero-offset case, but it is not a full-task solution. Case `0` still closes
high and fails the physical side-grasp gate, no evaluated run reopens, and
final bowl-plate XY remains about `139 mm` in the non-zero-offset video.

Updated interpretation: we now have clean evidence that the learned DHB contact
token can trigger a physically correct side-grasp close under a `+3 cm`
offset without object-distance gates. The remaining gap is narrower and more
defensible: make this side-grasp timing consistent across cases, then solve
post-close hold/transport/release.

## Closed-Transport Residual Follow-Up

A post-close action-only continuation was trained from the bestpos v2 contact
checkpoint:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_bestposv2_closedtransport_actiononly_frombestpos200`

The continuation freezes the contact path and trains only the closed-state
residual/action path. The best evaluated checkpoint is `steps_300`; redundant
`steps_150` and `final_model/pytorch_model.pt` artifacts were removed after
evaluation to preserve disk.

Fixed no-gate eval used threshold `0.50`, calibrated direct-action scale
`0.85, 1.55, 1.15`, `state_history_update_rate=policy`, and
`state_gripper_source=commanded`. Results:

- `steps_300` closes in both fixed cases with zero IK failures.
- Average minimum bowl XY improves to `19.4 mm`.
- The `3 cm` offset case reaches `13.3 mm` minimum bowl XY and has a physical
  close vector error of `10.6 mm`, so the learned side-grasp contact result
  survives the post-close continuation.
- No evaluated case reopens or releases near the plate. Final target XY remains
  about `137 mm` in the `3 cm` case, so this is not a full pick-place success.

Updated interpretation: the contact/reach story is now stronger and smoother,
but the closed-state residual by itself is insufficient for release. The next
useful intervention should add release/placement-positive supervision anchored
to observed physical close, not more scalar contact threshold tuning.

## Release-Aware Continuation Diagnostic

A release-aware continuation was trained from the smoother closed-transport
`steps_300` checkpoint:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_bestposv2_releaseaware_fromclosedtransport300`

This run unfreezes the contact-phase/release heads and closed residual while
keeping the learned contact head fixed. It uses offline
`dhb.object_release_phase` and `dhb.object_release_gripper_state` supervision;
deployment still uses only learned token outputs with object-distance gates
disabled.

The result is diagnostic but not a new baseline. `steps_300` learns to reopen
in both fixed cases, so the release token path is trainable. However, reopen is
not transport-conditioned enough:

- `0 cm`: close step `91`, reopen step `142`, but reopen is about `127 mm` XY
  from the plate and IK failures rise to `45`.
- `3 cm`: close step `95`, physical close step `97`, physical side-grasp vector
  error `13.9 mm`, reopen step `175`, but reopen target XY is `133.8 mm` and
  IK failures rise to `35`.

Updated interpretation: training a release head from nominal object-release
labels alone makes the model willing to reopen, but not at the correct
transport state. The next release intervention should mine rollout states that
are actually post-close and near-target as positives, paired with
closed-but-not-near-target hard negatives. This mirrors the contact-token fix:
the missing piece is balanced off-manifold rollout-state supervision, not a
deployment heuristic.

## Rollout Release Mining Infrastructure

`scripts/libero_rollout_release_miner.py` now exports dense gripper-state
release labels from policy traces. The manifest format intentionally reuses the
policy-trace contact loader: `target_close=1` means keep holding, and
`target_close=0` means release/open. A config can emit this target under
`dhb.object_release_gripper_state` via `policy_trace_close_target_field`.

Strict mining on fresh closed-transport policy traces found no honest
near-target release positives:

`_workspace/paper_eval/2026-04-27_policy_trace_release_strict_manifest.jsonl`

- `0 cm`: `211` closed-far hard negatives, best target XY `103.5 mm`.
- `3 cm`: `59` closed-far hard negatives, best target XY `119.5 mm`.

This is an important guardrail. We should not relabel these far states as
release positives for claim-level training. The immediate use of this manifest
is to pair rollout closed-far hard negatives with demo release positives, so
the learned release head stops reopening far from the plate while retaining the
ability to open when the visual/robot state matches true release.

## Release Hard-Negative Calibration Diagnostic

A conservative gripper-state continuation was trained from the active
closed-transport `steps_300` checkpoint using the strict rollout release
hard-negative manifest mixed with the base LeRobot data:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_releasehardneg_gripperstate_fromclosedtransport300`

The run trained the gripper-state release head with
`policy_trace_close_target_field=dhb.object_release_gripper_state`; deployment
still used learned model gripper decoding and disabled all object-distance
gates.

Fixed no-gate eval used calibrated direct-action scale `0.85, 1.55, 1.15`,
threshold `0.50`, and `model_gripper_decode_source=contact_gripper_state`.
Physical scoring:

`_workspace/paper_eval/2026-04-27_releasehardneg_gripperstate_physical_scores.json`

- `steps_100` reopens too early in the `3 cm` case: close step `91`, reopen
  step `93`, no physical close pass.
- `steps_200` delays reopen in both cases and lowers average IK failures from
  `37` to `21`, showing the hard-negative signal is learnable.
- `steps_200`, episode `1,+3 cm`, keeps the useful physical side-grasp close:
  command close step `99`, physical close step `101`, physical close vector
  error `10.6 mm`.
- Release remains invalid. `steps_200` reopens at target XY `116.4 mm` for
  `0 cm` and `133.6 mm` for `3 cm`; release pass is false in all cases.

Updated interpretation: hard-negative release calibration can delay premature
opening without object gates, but it does not create the missing
transport-to-target behavior. The active baseline remains the closed-transport
`steps_300` checkpoint for smooth reach/contact. The next breakthrough should
generate post-close transport/near-target positives, then use this release
miner to add honest release positives and closed-far hard negatives.

The diagnostic model weights were removed after evaluation to preserve disk;
configs, traces, aggregate summaries, and physical scores were retained.

## Object-Transport Conditioning Diagnostic

A demo-side object-target conditioning continuation was trained from the active
closed-transport `steps_300` checkpoint:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_objecttransport_fromclosedtransport300`

This run used the existing `libero_spatial_dhbxr_affordance_transport` data mix
and trained the object-target head plus closed residual. Deployment still used
learned gripper decoding and disabled object-distance gates.

Physical scoring:

`_workspace/paper_eval/2026-04-27_objecttransport_fromclosedtransport_physical_scores.json`

- `steps_300`, episode `1,+3 cm`, preserves the physical side-grasp pass:
  physical close step `101`, vector error `9.4 mm`.
- It does not solve transport/release. The same `3 cm` case reopens at target
  XY `136.5 mm`, final target XY is `145.0 mm`, and IK failures rise to `34`.
- The `0 cm` case still fails the close affordance, with physical close vector
  error `79.5 mm`.
- The transport miner confirms the object-transport branch does not improve
  post-close progress: `steps_300` improves target XY by only `10.4 mm` at
  `0 cm` and `6.5 mm` at `3 cm`, with no target-reached steps.

Updated interpretation: demo-side object-target conditioning is not sufficient
from the current rollout distribution. It can preserve the learned contact
event but does not pull the held object into the plate affordance. This further
supports the data-centric next step: generate or replay post-close
transport-to-target states, then train on those off-manifold positives.

The diagnostic model weights were removed after evaluation; configs, traces,
aggregate summaries, and physical scores were retained.

## Transport Correction Mining Interface

`scripts/libero_rollout_transport_miner.py` now converts closed-loop rollout
traces into post-close transport diagnostics and correction requests. This is
training/data-generation infrastructure only; deployment still uses learned
tokens and no object-distance gates.

The miner reports physical close, target XY at close, best post-close target
XY, target progress, target-reached counts, and a small set of closed
post-contact correction requests. Each request contains the generic
`object_minus_target_m` vector and the desired object-space delta to bring the
object to the target. This is the concrete interface for the next simulator or
world-model augmentation pass.

Applied to the active closed-transport `steps_300` policy traces:

`_workspace/paper_eval/2026-04-27_closedtransport_transport_mining.json`

`_workspace/paper_eval/2026-04-27_closedtransport_transport_correction_requests.jsonl`

- `0 cm`: physical close step `89`, target XY at close `114.1 mm`, best target
  XY `103.5 mm`, progress only `10.6 mm`, no target-reached steps.
- `3 cm`: physical close step `89`, target XY at close `129.8 mm`, best target
  XY `119.5 mm`, progress only `10.3 mm`, no target-reached steps.

Updated interpretation: the active policy is not merely releasing at the wrong
time; it barely transports the closed object toward the target after contact.
The next high-impact experiment should generate closed-state corrections that
move the object by roughly the exported `object_minus_target_m` vector, then
retrain closed residual and release heads on those corrected post-contact
states.

## Trainable Transport Correction Follow-Up

The transport miner was extended to export policy-trace training rows directly:
normalized corrective action targets, generic object-minus-target vectors, and
dense closed gripper-state targets. The StarVLA policy-trace loader now consumes
these optional fields, so the same manifest can train closed residual,
object-target, and gripper-state heads without adding deployment-time object
gates.

Two warm-start continuations were tested from the active closed-transport
`steps_300` checkpoint:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_transportcorrection_fromclosedtransport300`

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_transportcorrection_v2_fromclosedtransport300`

The first run used aggressive 8-step corrective labels and failed as an
execution candidate: it introduced close/reopen flicker and severe IK
regressions in the `3 cm` case. The second run used conservative 16-step labels
plus dense closed gripper-state supervision. `steps_200` fixed the v1 flicker
and avoided IK collapse, but still did not move the object close enough to the
target and weakened the `0 cm` physical-close result. `steps_400` overfit
further and reopened again.

Updated interpretation: the trainable transport-manifest path is now validated
mechanically, but synthetic residual labels from failed traces are not enough
to create successful post-contact transport. The next run should not be another
scalar sweep of these labels. It should generate successful post-contact
target-reaching positives from mined physically closed states, then retrain
closed residual and release heads on those positives.

The non-best transport-correction weights were removed after evaluation;
configs, summaries, traces, and scores remain for reproducibility.

## Sim-State Reset/Replay Diagnostic

The live rollout and checkpoint sweep now support `--record-sim-state-trace`,
which stores the flattened LIBERO simulator state in every step-trace entry.
This makes the next post-contact augmentation path concrete: restore a mined
closed-loop state, execute a target-reaching correction, then convert the
resulting trajectory into policy-trace positives.

New tools:

`scripts/libero_postcontact_correction_plan.py`

`scripts/libero_postcontact_correction_rollout.py`

The first reset/replay pass used the active closed-transport `steps_300`
checkpoint recaptured with simulator states. Treat this recapture as
infrastructure only, not as a replacement metric baseline: the recaptured
rollout was worse than the earlier best summary and included reopen/IK
failures.

The transport miner was tightened so correction candidates require both
observed physical closure and an executed closed command. It also now reports
object/EE transport coupling. In the sim-state recapture, object/EE path
coupling is only about `0.60`, and neither fixed case reaches the `45 mm`
target window.

Replay results:

- Earliest `+3 cm` physical-close reset: `0.0 mm` target progress.
- Shorter/faster earliest `+3 cm` reset: `0.0 mm` target progress.
- Later selected `+3 cm` reset: only `2.6 mm` target progress, no target reach.

Updated interpretation: physical close near the bowl is not enough. The next
dataset gap is grasp-attached transport: states must be selected or generated
where the object actually follows the end effector before target-reaching
corrections can become useful training positives.

### Coupling-Aware Replay Follow-Up

The transport miner now computes local object/EE coupling for each correction
candidate and supports coupling-aware candidate selection. This avoids both
early physical-close states where the object does not move and static late
states where the coupling ratio is numerically inflated by near-zero EE motion.

Coupling-aware `+3 cm` starts:

`_workspace/paper_eval/2026-04-27_closedtransport_simstate_transport_correction_requests_coupledstarts.jsonl`

Open-loop reset/replay from the best coupled starts gives consistent partial
transport. The five selected `+3 cm` rows improve target XY by about `26-33 mm`
with zero IK failures, ending around `98-105 mm` from target.

Closed-loop recomputed replay from the same coupled starts remains insufficient:
the best rows still stall around `97-100 mm` from target and accumulate IK
failures over a longer horizon.

Updated interpretation: coupling-aware mining is now the right default for
post-contact diagnostics, but these are not positive placement examples. The
next data-generation step needs simulator-assisted grasp-attached target
transport, not further scalar gains on weak naturally mined grasps.

### Attached Transport Positive Export

On 2026-04-28, the reset/replay runner was extended with an offline
`--attach-object-to-ee` mode. This mode keeps the object named by the
object-target field rigidly attached to the end effector during simulator
correction replay. It is data-generation infrastructure only and is not used in
deployment.

Attached replay from the five coupling-selected `+3 cm` reset states reached
the `45 mm` target window in every row:

`_workspace/paper_eval/2026-04-28_closedtransport_simstate_postcontact_replay_attached_coupledstarts_ep1_offset3_rows5_policytrace.json`

- `5/5` target-window reaches.
- Target XY improves by about `87-88 mm`.
- Final target XY is about `42.8-43.7 mm`.
- IK failures are `0`.

The same run exported trainable policy-trace positives:

`_workspace/paper_eval/2026-04-28_attached_transport_policy_manifest_ep1_offset3.jsonl`

`_workspace/paper_eval/2026-04-28_attached_transport_policy_trace_ep1_offset3`

The manifest has `30` NPZ-backed rows and smoke-loads through
`PolicyTraceContactDataset` with valid action, state, image, object-target,
gripper-state, close, and mask tensors.

Updated interpretation: we now have the first successful post-contact
target-reaching data source. It should be used for a narrow closed-transport
continuation from the active closed-transport `steps_300` checkpoint. The
deployment policy must remain object-gate-free; simulator attachment is only
for generating successful grasp-attached transport supervision.
