# DHB-XR Next 10h Execution Plan

## 2026-04-25 Current Correction: Evaluation Contract Was the Root Cause

The latest debugging found that several "good" and "bad" artifacts were not
using the same rollout contract. The older positive reach artifacts used
`init_state_source=demo`, raw LIBERO images, and in some cases demo-length
horizon settings. The stricter reruns with benchmark init, rotated images, and
demo-length caps made the same policies look worse. The corrected current
contract is:

- benchmark init with explicit bowl offset
- raw LIBERO images (`--image-orientation raw`)
- direct-action trajectory execution
- dynamic object-distance diagnostics
- closed-loop horizon longer than the demo (`--max-policy-steps 140` for
  reach-only offset tests)

Under this corrected contract, the frozen actionXYZ checkpoint is again clean:

- config: `configs/libero_dhbxr_qwen_fresh_actionxyz.yaml`
- run: `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000`
- checkpoint: `steps_400_pytorch_model.pt`
- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`
- result: `5/5` reach successes over episode/offset cases
  `0:0`, `0:+3`, `0:-3`, `1:+3`, `1:-3`, with `0` IK failures
- average minimum dynamic bowl XY: `39.2 mm`
- delayed cases are real: episode `1`, `+3 cm` reaches at step `96` while its
  demo length is `84`

This resolves the immediate reach instability. The next work should not tune
reach further unless a new benchmark case fails. The remaining active blocker
is the integrated learned gripper/contact sequence: under the corrected
protocol the residual contact checkpoint closes at step `75`, still about
`100 mm` XY from the dynamic bowl, and never reopens. That is a learned
phase/timing failure caused by demo-time-biased gripper supervision under
delayed offset reach. A diagnostic-only object-distance gate delays close to
step `165`, but still fails transport/release, so the remaining work is not a
scalar threshold problem.

Immediate next actions:

1. Use the corrected reach artifact as the current paper-safe reach/adaptation
   evidence.
2. Update runbooks and project-page captions to state the corrected evaluation
   contract.
3. For full-task work, train/evaluate gripper/contact on offset-augmented or
   rollout-derived observations so close timing is tied to current perception
   and state, not the original demo timestep.
4. Do not add object-distance gripper gates to the policy path; use them only
   as diagnostics or offline supervision for regenerated training data.

## 2026-04-27 Update: Learned Gripper State Works, Full Task Still Fails Affordance/Release

The affordance-labeled branch has now been trained and evaluated:

- action residual run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_fromxy500`
- gripper-only continuation:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_grip_fromaff500`
- concise summary:
  `_workspace/paper_eval/2026-04-27_affordance_gripper_retrain_summary.md`

The result is mixed but useful. The action residual preserves the corrected
5-case reach gate at `5/5`, average dynamic bowl XY `38.2 mm`, and `0` IK
failures. The gripper-only continuation makes learned gripper state actionable
without object-distance gates: `steps_800` closes at step `83` with `0` IK
failures and minimum bowl XY `6.7 mm`. Demo gripper replay is much worse
(`194.8 mm` close-vector error and `112` IK failures), so the learned gripper
path is the right direction.

The remaining blocker is still not solved. The learned close happens too high
and not tightly enough at the demo side-grasp affordance. Best current
close-vector error is `42.3 mm` with generic threshold `0.70`; the pass target
is materially lower. The run also never reopens and remains about `116-128 mm`
from the plate at final placement.

Immediate next actions:

1. Keep the affordance action-residual branch as the active reach-safe branch.
2. Do not use object-distance gates in rollout. Use the affordance scorer as
   the gate and simulator geometry only for offline labels/diagnostics.
3. Retrain event/contact heads with better side-affordance close calibration
   and explicit release/object-target supervision. The current contact-phase
   head is not usable even after fixing thresholded 3-class event decoding.
   The event-BCE continuation from the gripper step-800 checkpoint is also a
   negative result: `0.55` closes at step `3` with `339.3 mm` close-vector
   error, while stricter generic thresholds `0.80-0.90` delay close to
   `74-75` but still leave `76-80 mm` close-vector error and no release.
   A release-heavy dense-gripper continuation shows release supervision is
   learnable (`steps_300` reopens at step `165` with no gripper gates), but it
   either reopens far from the plate or creates an immediate close/open pair;
   by `steps_600` it again fails to reopen. The next branch should couple
   gripper phase persistence with post-contact transport, not only oversample
   release labels.
   A target-aware smoothness loss has now been implemented and tested; it
   removes flicker/IK collapse but suppresses release in the first weighting.
   Use it as a building block, not as the final full-task result.
4. Preserve the reach gate before every full-task claim: `5/5`, average bowl
   XY near `38-40 mm`, and `0` IK failures under benchmark init/raw images.
5. Treat full pick-place as unclaimed until a no-gate run passes both
   close-vector affordance and object-target release metrics.

## 2026-04-27 Overnight Update: Evaluation Seed Is Now Explicit

The live evaluator now accepts `--rollout-seed`, seeds Python, NumPy, Torch,
CUDA, and LIBERO, and writes both `config_seed` and `rollout_seed` into the
summary JSON. This fixes a comparison flaw: several continuation configs used
different training seeds, and the rollout script previously reused those seeds
for the environment. Single-rollout currentready/release comparisons were
therefore not controlled.

Controlled seed-70 reruns show the active blocker more clearly:

- `currentready`, `close=0.55/open=0.40`: close step `91`, no reopen, `0` IK
  failures, actual close-vector error `36.4 mm`, but best possible
  close-affordance error in the same trace is `3.3 mm` at step `110`.
- `currentready`, `close=0.58/open=0.40`: close step `101`, reopen step `121`,
  actual close XY `16.9 mm`, but premature release and `141 mm` final
  bowl-plate XY.
- `stateonly_release`, `close=0.55/open=0.40`: only the dense
  `gripper_state_head` changed from currentready; it closes at step `87`, never
  reopens, reaches a `3.0 mm` best close-affordance opportunity later, and
  finishes `129 mm` from the plate.
- `stateonly_release`, `close=0.60/open=0.50`, debounce `4`: no close and late
  IK failures, so scalar hysteresis is not the missing method.

Immediate next actions:

1. Use fixed rollout-seed matrices for all future benchmark claims. Keep old
   uncontrolled single-run traces only as diagnostics/provenance.
2. Preserve the currentready contact-readiness idea, but do not treat the
   current checkpoint as a solved gripper policy.
3. Train the next branch for explicit phase selection: current contact-ready
   close, persistent held-object transport, and learned release near target.
   The action/reach path should remain frozen or residual-preserved until it
   passes the corrected reach gate again.
4. Add no object-distance logic to deployment. Simulator geometry remains
   acceptable only for offline labels and post-hoc scoring.

Follow-up executed:

- Added
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_transport_release.yaml`.
  It freezes currentready close-readiness and trains only the closed-state
  residual adapter plus dense gripper-state release path.
- The run trains with `4.464M` trainable parameters. Step `200` is the useful
  checkpoint: close step `103`, reopen step `221`, `0` IK failures, close-vector
  error `37.1 mm`, final target XY `114.4 mm`. It restores learned release
  without breaking close, but does not solve object transport.
- Step `400` overtrains into flicker (`85 -> 89` close/open) and `12` IK
  failures. Do not continue this exact weighting longer.

Next action:

Train a held-object transport target more directly. Keep currentready close
frozen, but add a loss that keeps the gripper closed through transport and
weights closed-phase XY action/trajectory toward the object-target segment.
Use `steps_200` from the transport-release branch as the diagnostic reference,
not as a solved policy.

Follow-up executed:

- Added
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_closedtransport_actiononly.yaml`.
  It starts from the transport-release `steps_200` partial, freezes every
  gripper/contact head, and trains only the closed-state residual adapter
  (`0.267M` trainable parameters).
- This isolated action-only test is negative under controlled
  `rollout_seed=70`. Step `150` closes at `91`, never reopens, keeps `0` IK
  failures, and ends `123.9 mm` from the target. Step `300` closes at `103`,
  reopens at `237`, keeps `0` IK failures, and matches the useful
  transport-release close score (`37.1 mm` close-vector error, `19.0 mm` close
  XY), but final target XY remains `112.9 mm`.
- Conclusion: closed residual action MSE alone cannot solve held-object
  transport. The next high-leverage change should add an explicit offline
  object-level transport/release target or post-contact relabeling signal, then
  deploy only learned model outputs. Do not keep sweeping scalar residual
  weights or thresholds.

Follow-up executed:

- Regenerated/exported object-level transport labels in
  `/home/andypark/Projects/data/libero_lerobot_affordance_transport`:
  `dhb.object_to_eef_pos`, `dhb.target_to_eef_pos`,
  `dhb.object_minus_target_pos`, and `dhb.transport_phase`.
- Added configurable object-target supervision in `QwenDHB`, including
  multi-field target concatenation, live rollout diagnostics for
  `pred_object_target`, and an opt-in separate cross-attention readout via
  `object_target_readout: cross_attention`.
- The simple `object_minus_target` branch is negative. It preserves the useful
  close at step `150` but the predicted object-target vector is nearly constant
  in live rollout; step `300` regresses into early close/open and IK failures.
- The richer object-pair target read from frozen action features is also
  negative: step `150` never reopens and step `300` releases far from target.
- The cross-attention object-pair readout produces non-constant live geometry
  predictions, confirming the new readout path works mechanically, but it still
  fails the task: step `150` flickers (`87 -> 89`), while step `300` returns to
  stable close with no release and `117.9 mm` final target XY.

Next action:

Stop treating object geometry as only an action-context auxiliary. The next
branch should train an explicit learned release-ready signal from offline
object-target geometry, while preserving the currentready close head and the
future-aligned dense gripper hold signal. The live decoder should remain
object-free: close from current contact readiness, hold from persistent gripper
state, and open only from the learned release-ready output rather than from a
raw future gripper pulse.

Follow-up executed:

- Added and trained
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml`
  from the useful currentready transport-release partial. It preserves the
  currentready close-readiness head and routes dense gripper/phase supervision
  to offline object-target release labels (`dhb.object_release_*`), while
  keeping rollout object-free.
- The first sweep exposed an evaluation-contract bug: `--max-policy-steps 0`
  caps full-task rollout to the nominal demo length. For episode `1,+3 cm`,
  that stops at `84` policy steps, before known controlled baselines close
  (`91-103`). The rollout and sweep scripts now warn/record
  `demo_length_cap_warning`; full-task offset evaluation must use an explicit
  long horizon such as `--max-policy-steps 240`.
- Under the corrected fixed `rollout_seed=70`, true-no-gate,
  `max_policy_steps=240` protocol, the object-release branch learns a real
  no-gate reopen but still misses placement. Step `300` closes at `82`,
  reopens at `153`, reaches `1.8 mm` minimum dynamic bowl XY, but has
  `51.7 mm` close-vector error, `121.1 mm` reopen target XY,
  `116.1 mm` final target XY, and `11` IK failures.
- Generic hysteresis diagnostics show calibration tradeoffs, not a solution.
  `close=0.60/open=0.60/debounce=1` improves actual close to `37.9 mm`
  close-vector error and `12.8 mm` close XY, with a learned reopen at `139`,
  but final target XY remains `110.6 mm` and IK failures rise to `64`.
  Direct `contact_phase` decoding is worse (`75 -> 87` close/open,
  `73.7 mm` close-vector error, `75` IK failures).

Next action:

Keep the hybrid currentready + dense gripper-state decoder as the active
deployment path. Do not continue scalar threshold sweeps as the main line.
The next high-leverage training change should strengthen held-object
transport/release supervision after close, preferably with a persistent
release-ready or object-target progress signal that cannot collapse into a
short close/open pulse. Continue evaluating with fixed rollout seeds,
`--disable-gripper-gates`, and `--max-policy-steps 240` for full-task offset
runs.

## 2026-04-25 Late Update: Reach Is Now Validated, Full Task Is Phase-Limited

The current highest-confidence executable result is no longer the older
DHB-decoded live trajectory. It is the calibrated direct-action execution path
from the DHB-trained StarVLA model:

- run: `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000`
- checkpoint: `steps_400_pytorch_model.pt`
- execution: `--trajectory-source direct_action`
- calibration: `--direct-action-xyz-scale 1.0 1.35 1.15`
- reach-only result: `5/5` successes across episodes `0,1` and offsets
  `0,+3,-3 cm`, with `0` IK failures
- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_offsets.json`
- representative non-zero-offset video:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3.mp4`
- video JSON:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3_video.json`
  (`success_at=82`, `min_bowl_xy_m=0.0144`, `min_bowl_xyz_m=0.0794`,
  `path_efficiency=0.960`, `ik_fail_total=0`)

This changes the immediate closure target. The remaining blocker is not broad
reach instability, end-effector rotation snapping, or IK collapse. The full
task fails because the policy reaches close-ready bowl geometry near the end of
the rollout. Demo gripper replay has already expired by then, while the model
gripper can close at close-ready geometry but only around step `81`, leaving no
time for lift, transport, and placement.

Immediate work:

1. Treat calibrated reach-only videos as the clean current behavior evidence.
2. Do not add object-specific close-distance heuristics to the policy path.
   Distance gates may be used only as diagnostics/evaluator instrumentation.
3. Train or evaluate the next branch for earlier phase progression and
   full-action/gripper timing together, instead of further replaying demo
   gripper timing or adding rollout-specific gripper triggers.
4. Keep the DHB-decoded trajectory path as an active research gap. The current
   paper-safe live behavior claim should be phrased around the trained
   DHB/direct-action execution path until DHB-decoded rollout catches up.

## 2026-04-25 Late-Night Gate: Joint Gripper Fine-Tune Failed Reach Preservation

The first warm-started full-action/gripper branch is a negative result:

- config: `configs/libero_dhbxr_qwen_fresh_actionfull_grip.yaml`
- run: `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionfull_grip_from400`
- warm start: `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400`
- enabled losses: invariant, direct action including gripper, contact phase,
  and gripper state
- reach-preservation sweep:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionfull_grip_from400/live_reach_checkpoint_sweep_calib_y135_z115_2case.json`

Result: no checkpoint preserved the calibrated reach behavior. Best checkpoint
by the two-case diagnostic was `steps_600`, with `0/2` reach successes and
average minimum bowl XY distance `0.1569 m`; `steps_800` degraded to
`0.2368 m`. IK remained feasible, so the failure is not IK collapse. The
joint objective altered the learned direct-action trajectory before gripper
timing could help.

Updated next experiment:

1. Keep the validated geometry/action execution branch frozen.
2. Train only learned, generic gripper/contact heads on top of the fixed
   actionXYZ representation.
3. Gate this branch first on reach preservation, then on full-task model
   gripper timing.
4. If gripper heads learn timing without harming reach, use that checkpoint for
   a full-task rollout; otherwise keep the current result as evidence that
   phase supervision must be separated more carefully from geometry learning.

Active config:

`configs/libero_dhbxr_qwen_fresh_gripper_heads_only.yaml`

## 2026-04-25 Night Update: Gripper Calibration Helped, Full Task Still Needs Contact-Level Training

The calibrated gripper-head continuation is now a negative result for full
pick-place, but it clarified the root cause:

- config:
  `configs/libero_dhbxr_qwen_fresh_gripper_heads_calibrated.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_gripper_heads_calibrated_from600`
- warm start:
  `dhb_starvla_qwen_fresh_gripper_heads_randompos_from400/steps_600`
- frozen modules: Qwen, trajectory decoder, invariant head, state projection,
  and direct-action head
- trained modules: generic contact/phase/gripper heads only
- final checkpoint: `steps_2000_pytorch_model.pt`

The 64-batch calibration diagnostic improved closed-state recall but over-biased
the close class:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_gripper_heads_calibrated_from600/gripper_calibration_step2000_64b.json`
- target closed fraction: `0.445`
- mean close probability on closed targets: `0.893`
- mean close probability on open targets: `0.664`
- best threshold in the sampled sweep was around `0.8`, but open recall stayed
  weak.

Live full-task diagnostics on `episode_idx=1`, `offset=+3 cm`:

- fast scale `1.5,2.0,1.5`, threshold `0.8`: closes too early
  (`close_step=30`, `z=181.9 mm` above bowl).
- fast scale `1.5,2.0,1.5`, threshold `0.98`, open threshold `0.05`: no
  reopen, but still misses grasp (`min_bowl_xyz=72.8 mm`).
- slower scale `1.0,1.35,1.35`, threshold `0.98`, open threshold `0.05`:
  close stays persistent and the EE reaches `min_bowl_xyz=22.1 mm`, but the
  task still fails.
- gated diagnostic with the same slower scale delays close to local
  bowl-ready geometry (`close_step=83`, `z=42.9 mm`) but still fails
  (`min_bowl_xyz=30.7 mm`), so object-specific gates are not the missing final
  ingredient.

Conclusion: the gripper command polarity is correct (`+1` closes, `-1` opens),
and the learned head can produce confident close probabilities. The remaining
full-task blocker is contact-quality grasp and post-grasp transport, not just
threshold timing. Do not spend more cycles tuning scalar gripper thresholds as
the main path. The next training intervention should explicitly improve
contact-level reach/grasp trajectories while preserving the validated reach
backbone.

Updated next experiment:

1. Keep `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400` as the
   paper-safe reach/adaptation checkpoint.
2. Treat `dhb_starvla_qwen_fresh_gripper_heads_calibrated_from600/steps_2000`
   as a diagnostic showing that learned gripper labels alone are insufficient.
3. Add a contact-reach objective or dataset augmentation that targets
   sub-`15 mm` bowl contact before close, rather than merely increasing gripper
   probability.
4. Only after contact reach is stable, return to learned gripper integration
   with hysteresis/debounce fixed as generic probability decoding, not
   object-specific environment gates.

## 2026-04-25 Overnight Update: Do Not Fine-Tune the Direct-Action Head Yet

The bounded "head-only contact" intervention is also a negative result:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionhead_contact.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionhead_contact_from400`
- warm start:
  `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400`
- frozen modules: Qwen, query embeddings, VLM projection, decoder, invariant
  head, committed-prefix projection, and state projection
- trainable modules: shallow direct-action head plus generic contact/phase and
  gripper-state heads
- reach-preservation artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionhead_contact_from400/live_reach_checkpoint_sweep_calib_y135_z115_2case.json`

Result: this did not preserve the validated reach policy. `steps_400` and
`steps_800` both produced `0/2` reach successes in the two-case diagnostic.
Average minimum bowl XY was `0.2305 m` at step `400` and `0.2328 m` at step
`800`, despite smooth motion and `0` IK failures. The model learned a smooth
but systematically wrong direct-action drift.

Conclusion: the direct-action output head is currently part of the fragile
working reach solution. Do not fine-tune it with contact-heavy data until we
add a stronger reach-preservation constraint or a residual/adaptor structure
that cannot overwrite the base action field.

Updated next path:

1. Keep `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400` frozen for
   paper-safe reach/adaptation evidence.
2. Treat all action-head/full-action fine-tunes as diagnostic negatives unless
   they first pass the `5/5` reach-preservation gate.
3. If training continues, use additive residuals initialized to zero or a
   separate post-reach grasp module; do not overwrite the base action head.
4. In parallel, prioritize matched `QwenDHB` versus `QwenOFT` evaluation and
   paper story evidence, because that path already has a bounded positive
   result.

## 2026-04-25 Overnight Positive: Residual Adapter Preserves Reach

The first reach-preserving contact-training structure is now positive:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_contact.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400`
- warm start:
  `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400`
- trainable modules: zero-initialized additive action residual adapter plus
  generic contact/phase/gripper heads
- frozen modules: Qwen, query embeddings, VLM projection, decoder, invariant
  head, committed-prefix projection, state projection, and the original
  direct-action head
- reach-preservation artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/live_reach_checkpoint_sweep_calib_y135_z115_2case.json`

Result: the residual branch preserves the validated reach behavior in the
two-case gate. At `steps_400`, it gets `2/2` reach successes, average minimum
bowl XY `0.0110 m`, average minimum bowl XYZ `0.0542 m`, average path
efficiency about `0.967`, and `0` IK failures. This is the first useful
training result after the calibrated actionXYZ checkpoint: it confirms that
contact/gripper learning must be additive or staged, not an overwrite of the
base action field.

Full-task diagnostic on `episode_idx=1`, `offset=+3 cm`, with model gripper
and no object-specific gates:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/full_task_step400_calib_y135_z135_thr080_open020_deb2_ep1_offset3_model_gripper_nogates_max180.json`
- result: `success=false`
- proximity: `min_bowl_xy_m=0.00484`, `min_bowl_xyz_m=0.0229`
- gripper: no close fired under threshold `0.8`; learned close probabilities
  stayed under about `0.41`

Interpretation: the residual path can reach contact-ready geometry without
breaking IK, but the learned gripper-state decoder is underconfident in this
checkpoint. The next useful diagnostic is generic probability calibration
only, not object-distance gating. If the low-probability close signal has
usable timing, a lower threshold should close near contact; if not, train the
gripper/contact heads longer while freezing the base reach path and residual
geometry.

## 2026-04-25 Overnight Continuation Diagnostic: More Gripper Weighting Is Negative

The residual-adapter continuation from `steps_400` preserves reach but makes
the learned gripper signal less useful:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_contact_fromres400.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_fromres400`
- warm start:
  `dhb_starvla_qwen_fresh_actionres_contact_from400/steps_400`
- training change: higher `lambda_gripper_state`, higher
  `lambda_contact_phase`, and stronger close/contact class weights while the
  base action path remains frozen.

Reach gate:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_fromres400/live_reach_checkpoint_sweep_calib_y135_z115_2case.json`
- `steps_400`: `2/2`, average minimum bowl XY `11.5 mm`, average minimum bowl
  XYZ `55.7 mm`, `0` IK failures
- `steps_800`: `2/2`, average minimum bowl XY `13.8 mm`, average minimum bowl
  XYZ `57.2 mm`, `0` IK failures

Full-task no-gate diagnostic on `episode_idx=1`, `offset=+3 cm`:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_fromres400/full_task_step400_calib_y135_z135_thr050_open020_deb2_ep1_offset3_model_gripper_nogates_max180.json`
- result: `success=false`
- gripper: closes at step `45`, far before contact
- close geometry: EE is about `116 mm` XY and `147 mm` Z away from the bowl
- later proximity still reaches the bowl (`min_bowl_xy_m=0.000624`,
  `min_bowl_xyz_m=0.020548`), so the early close is a calibration/phase
  failure, not a reach failure.

The original residual checkpoint remains the better contact diagnostic. With a
generic lower gripper-state threshold (`0.395`), it closes near the bowl
(`close_step=83`, `ee_minus_bowl=[-9.4,-8.3,34.6] mm`) and reaches
`0.12 mm` XY / `20.9 mm` 3D minimum distance, but still fails grasp/transport.
The contact-phase decoder is currently inert: using `contact_phase` as the
model-gripper source produced low probabilities around `0.10-0.14` and no
close event.

Decision:

1. Do not continue scalar gripper-state class weighting as the main path.
2. Do not use `dhb_starvla_qwen_fresh_actionres_contact_fromres400` for
   full-task claims.
3. Keep `dhb_starvla_qwen_fresh_actionres_contact_from400/steps_400` as the
   best residual contact diagnostic.
4. Next, use dynamic object-step traces to determine whether the failed close
   produces object motion/lift. If not, the next training target must improve
   contact-quality grasp trajectories while preserving the validated reach
   backbone.

Dynamic trace result:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/full_task_step400_calib_y135_z135_thr0395_open020_deb2_ep1_offset3_model_gripper_nogates_trace.json`
- result: `success=false`, `reward=0.0`, `close_step=63`,
  `reopen_step=null`, `ik_fail_total=44`
- close geometry: EE is still about `66 mm` XY / `93 mm` Z from the dynamic
  bowl pose when the gripper first closes
- closest dynamic contact: EE-bowl XY reaches `2.1 mm` at step `88`, and 3D
  distance reaches `19.2 mm` at step `90`
- object motion: the bowl moves `[38.2, 36.1, 37.5] mm` from its initial pose
  by the end, for `64.5 mm` total displacement and `37.5 mm` lift
- placement miss: final bowl-plate offset is about `127.7 mm` XY and `33.2 mm`
  Z, with the gripper still closed

Updated interpretation: the residual branch is not merely failing to grasp.
With a low generic threshold it can make contact and move/lift the bowl, but it
does not complete transport/placement or release. The next most useful model
target is therefore a reach-preserving post-contact transport/placement phase,
not another close-threshold or polarity experiment.

## 2026-04-25 Transport-Focused Residual Fine-Tune: Negative

A transport-focused continuation tested whether stronger terminal/post-contact
action weighting could improve placement while preserving the validated reach
backbone:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_transport.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_transport_fromres400`
- warm start:
  `dhb_starvla_qwen_fresh_actionres_contact_from400/steps_400`
- training change: higher Z/terminal action weighting and more terminal/event
  sampling while keeping the base direct-action head frozen.

Reach gate:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_transport_fromres400/live_reach_checkpoint_sweep_calib_y135_z115_2case.json`
- `steps_300`: `2/2`, average minimum bowl XY `18.8 mm`, average minimum bowl
  XYZ `65.1 mm`, `0` IK failures
- `steps_600`: `2/2`, average minimum bowl XY `18.6 mm`, average minimum bowl
  XYZ `67.1 mm`, `0` IK failures

Full-task traced diagnostics on `episode_idx=1`, `offset=+3 cm`, threshold
`0.395`, no object-specific gates:

- `steps_300`: no close, max close probability `0.282`, closest EE-bowl
  `8.2 mm` XY / `20.6 mm` XYZ, bowl displacement `53.4 mm`, max lift
  `13.3 mm`
- `steps_600`: no close, max close probability `0.346`, closest EE-bowl
  `8.8 mm` XY / `23.1 mm` XYZ, bowl displacement `53.1 mm`, max lift
  `13.8 mm`

Decision: this branch preserves a usable open-gripper reach but suppresses the
close signal and does not solve placement. Do not spend more time on scalar
transport fine-tuning with shared residual weights. The next model change
should decouple pre-contact close prediction from post-contact transport
adaptation, preferably through a gripper-state/phase-conditioned residual path
initialized to preserve the existing reach and close behavior.

## 2026-04-25 Claim-Closure Execution Order

This supersedes the older token-first execution order. The continuous DHB path
is the current executable policy story; discrete tokens remain a gated
supporting track until predicted-token fidelity is strong enough for embodied
rollout.

Immediate work:

1. Freeze the canonical `predict_trajectory() -> embodik` evaluator contract,
   including history buffering, reference handling, chunk continuation, gripper
   policy, smoothing, and success metrics.
2. Expand the matched `QwenDHB` versus `QwenOFT` comparison under that frozen
   contract. The minimum useful matrix is the same tasks, same checkpoints,
   same offsets, same seeds, and the same downstream IK/sim path.
3. Package paper-grade evidence from the matched comparison: success rate,
   IK-feasibility, final object-relative offsets, path smoothness, videos, and
   runtime/memory footprint.
4. Prepare small-data seed-size curves only after the evaluator contract stops
   changing.
5. Continue discrete-token work only through factorized or trimmed token
   targets. Do not spend more time on same-objective token warm starts unless
   the offline endpoint-drift gate improves.

Claim gates:

- Continuous DHB claim gate: beat matched `QwenOFT` under the frozen evaluator
  without per-model rollout exceptions.
- Scaling claim gate: show better success per real-demo count across seed-size
  curves.
- Discrete-token claim gate: keep GT-token codec near millimeter-scale decode
  quality, beat majority-token prediction, and improve full decoded endpoint
  drift before embodied rollout.

## Harness Review Integration

Evaluator contract review:

- Use StarVLA Stage-1 scripts for paper claims:
  `scripts/starvla_dhb_embodik_stage1.py` for `QwenDHB` and
  `scripts/starvla_oft_embodik_stage1.py` for `QwenOFT`.
- Pack Stage-1 outputs with `scripts/starvla_dhb_embodik_pack.py`.
- Use `_workspace/eval_dhb_head_embodik.py --stage ik` and `--stage sim` only
  as the shared downstream evaluator.
- Treat `scripts/starvla_dhb_live_checkpoint_sweep.py` as diagnostic and
  checkpoint-ranking infrastructure, not the primary paper success contract.
- Fix the current artifact mismatch before quoting numbers: existing DHB
  canonical artifacts use episodes `0,1,2`, while QwenOFT artifacts use
  `0,1,10`.

Immediate evaluator artifacts to regenerate:

1. `QwenDHB` on episodes `0,1,10`, offsets `0,3`, with frozen Stage-1 knobs.
2. `QwenOFT` through the same pack, IK, and sim path with matching artifact
   names.
3. Report default-attempt success separately from any high-gain retry.

Completed result:

- Artifact summary:
  `_workspace/paper_eval/2026-04-25_matched_task0/summary.md`
- `QwenDHB`: `1/3` at `0 cm`, `1/3` at `3 cm`
- `QwenOFT`: `0/3` at `0 cm`, `0/3` at `3 cm`
- Caveat: `QwenDHB` has `140` IK failures across the six cells, while
  `QwenOFT` has `0`. The next closure task is therefore not broader rollout
  yet; it is reducing DHB late-trajectory IK infeasibility under a symmetric
  evaluator.

Root-cause update:

- The DHB encoder/decoder alignment itself is clean, but the current StarVLA
  DHB supervision is not. Full-episode DHB invariant rows were being sliced as
  if they were chunk-local targets. Because DHB-DR frame initialization depends
  on the chunk start, that silently injects `10-73 mm` target reconstruction
  error into the training data.
- The dataloader now supports chunk-local target recomputation via
  `recompute_dhb_invariants_from_actions: true`. Active next-run configs also
  use `state_history_window: 4` so training and evaluation both condition on
  state history ending at the current timestep.
- Next high-impact action: retrain from a clean/frozen-Qwen initialization with
  chunk-local DHB targets, then rerun the same task-0 matched evaluator before
  expanding the matrix.

Matched comparison review:

- Phase A: paper-safe task-0 comparison on the same episodes `0,1,10` and
  offsets `0,3`.
- Phase B: broaden to `libero_spatial` task IDs `0,1,2`, seeds `0,1,2`, and
  offsets `0,3` for a `36`-cell matrix.
- Required metrics: success, final object-relative placement, closest bowl
  approach, IK failure count, FK residual, path efficiency, video, predict
  latency, and peak memory.
- Claim only a prototype executable-trajectory advantage unless the broader
  matrix is completed.

Discrete-token review:

- The next token experiment should not be another flat-token warm start.
- Implement factorized FSQ digit targets, convert predicted factor codes back
  to flat OAT IDs for the existing decoder, and trim supervision/decode to
  `keep_k=7` for the current FSQ4/K8 tokenizer because the tail register is
  low-information.
- New gates should include GT codec endpoint drift `<= 0.025 m`, decoded
  `k=7` endpoint drift `<= 0.18 m`, improvement over
  `fsq4_tokenonly1`, factor accuracy above majority by `0.10`, and repeated
  token sequence fraction `<= 0.50`.

## Current State

The continuous DHB-XR action head remains the executable policy path. It has a
prototype-scale matched control result against `QwenOFT`: `QwenDHB` reaches
`1/3` at both `0 cm` and `3 cm`, while the matched `QwenOFT` control is `0/3`
at both offsets under the same `embodik` downstream evaluator.

The discrete invariant-token path is not yet validated. `short2` and `short3`
fixed the old supervised-token shape/mismatch problem, but corrected sweeps
show both derived GT OAT tokens and predicted tokens collapsing to ID `7`.
Therefore `avg_token_accuracy = 1.0` is a degenerate result and must not be used
as policy evidence.

After adding a standalone tokenizer gate, `oat_tokenizer_gate1` produced the
first positive discrete-token signal: diverse non-collapsed OAT tokens on held
out DHB invariant chunks. This validates the tokenizer direction outside the
VLA loop, but not yet the StarVLA policy.

## Root Cause Hypothesis

The likely blocker is the OAT tokenizer objective. Reconstruction-only OAT
training can let the decoder learn an average trajectory prior while tokens
carry little information. FSQ prevents learned-codebook collapse, but it does
not by itself guarantee occupied bins or useful per-register token diversity.

## Execution Order

1. Freeze or consistently reuse the `oat_tokenizer_gate1` tokenizer source for
   StarVLA labels and discrete decoding.
2. Add action-head loading of pretrained tokenizer weights plus normalization
   stats, or regenerate exported OAT IDs from that exact checkpoint.
3. Re-run StarVLA supervised discrete fine-tuning only after the tokenizer
   source is stable.
4. Run offline discrete StarVLA sweeps and require GT/predicted token histograms
   to beat majority-token baselines.
5. Return to embodied discrete rollout only after both tokenizer and StarVLA
   offline gates pass.

## Strict Gates

- Standalone tokenizer full reconstruction `R2 >= 0`.
- Global token unique count at least `8`.
- Effective token count at least `4`.
- Each register position has at least `2` unique token IDs.
- Collapsed sample fraction no higher than `0.25`.
- StarVLA token accuracy must beat the GT majority-token baseline.
- Full discrete decode must be competitive with continuous decode on endpoint
  drift before `predict_trajectory() -> embodik` rollout.

## Story Implication

The strong current story is: DHB-XR is a geometric action interface that already
has representation stability and prototype executable advantage over a matched
regular action head. The stronger paper story remains pending: discrete
invariant tokens and DHB-adapted data augmentation could enable low-data VLA
scaling, but that requires seed-size and augmentation ablations after the token
pipeline is no longer collapsed.

## Do Not Spend Time On

- More closed-loop heuristics for the current discrete checkpoint family.
- Embodied discrete rollout while GT tokens are collapsed.
- More same-family warm-starts from a collapsed tokenizer.
- Gripper timing experiments before reach/token quality gates are fixed.

## 2026-04-24 Revised State After Token Warmup

The discrete path now has a stronger split:

- Representation/codec: validated on held-out GT chunks. Frozen OAT
  reconstruction gives `0.004207 m` position MAE and `0.016274 m` endpoint
  drift.
- Direct token policy: partially recovered, not yet a headline claim. The
  `dhb_starvla_qwen_discrete_tokenwarm1/steps_1000` checkpoint improves
  full-token endpoint drift from `1.614899 m` to `0.180139 m` versus the
  300-step frozen-OAT run, but predicted token diversity is still much lower
  than GT diversity.

Immediate next steps:

1. Run a small embodied/predict smoke only as a regression test for the
   improved token path, not as a paper result.
2. Add a token overfit probe on a tiny fixed subset. If the model cannot
   overfit token labels, change architecture/codebook before spending more
   full VLA training.
3. If overfit passes, continue token-warmup training longer and evaluate every
   1000-3000 steps with the corrected offline gates.
4. Evaluate smaller or factorized token spaces. Current normalized GT labels
   have `136` unique token IDs across only `192` held-out token positions,
   making the direct 1000-way CE problem unnecessarily hard for low-data
   warmups.
5. Preserve the story distinction: GT-token codec evidence supports invariant
   compression; direct token-policy evidence remains an active gap.

## 2026-04-24 Overfit Gate Result

The token-head overfit probe passed on 4 fixed examples from
`dhb_starvla_qwen_discrete_tokenwarm1/steps_1000`:

- artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1/token_overfit_probe_4ex_step300.json`
- final token accuracy: `1.0`
- final sequence accuracy: `1.0`
- final token loss: `0.000123`
- predicted global token IDs match the target count: `30 / 30`

This rules out the most damaging failure hypothesis: the token branch is not
incapable of representing the frozen OAT labels. The next high-ROI step is now
longer token-warmup training with corrected offline sweeps. If held-out token
diversity still saturates far below GT diversity, the next structural change
should be a smaller/factorized OAT token space or auxiliary per-register losses.

## 2026-04-24 Longer Token Warmup Result

`dhb_starvla_qwen_discrete_tokenwarm2` completed through `steps_3000`. The
missing `steps_1000`, `steps_2000`, and `steps_3000` checkpoints have all been
swept with normalized GT-token labels and teacher-forced GT codec evaluation.

| Checkpoint | Unique token sequences | Pred global IDs | Token accuracy / majority | Full pos MAE | Full endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tokenwarm1/steps_1000` | `5 / 24` | `10` | `0.052083 / 0.03125` | `0.046055 m` | `0.180139 m` |
| `tokenwarm2/steps_1000` | `8 / 24` | `15` | `0.083333 / 0.03125` | `0.123912 m` | `0.479050 m` |
| `tokenwarm2/steps_2000` | `11 / 24` | `20` | `0.072917 / 0.03125` | `0.082015 m` | `0.295894 m` |
| `tokenwarm2/steps_3000` | `9 / 24` | `20` | `0.067708 / 0.03125` | `0.049830 m` | `0.189869 m` |

Decision:

- Best executable discrete checkpoint by endpoint drift remains
  `tokenwarm1/steps_1000`.
- Best token-diversity checkpoint is `tokenwarm2/steps_2000`, but its decoded
  trajectory drift is too high for closed-loop rollout.
- `tokenwarm2/steps_3000` is a useful secondary candidate because it keeps
  better diversity than `tokenwarm1` while recovering most trajectory fidelity,
  but it still does not beat `tokenwarm1`.

Next most impactful work:

1. Add a reduced-cardinality or factorized-token variant of the frozen OAT
   target. The current GT labels use `136` global IDs across only `192` held-out
   token positions, which is too high-cardinality for the current low-data
   StarVLA warmup.
2. Add a token-only full-data warmup option that freezes the continuous
   invariant/action branches after a good continuous checkpoint. This isolates
   token CE learning from trajectory-regression drift.
3. Keep continuous DHB as the executable policy story until a token-native
   checkpoint beats `tokenwarm1/steps_1000` on full decoded endpoint drift and
   passes a small embodied smoke.

Paper-facing claim state:

- Strong enough now: DHB invariant chunks can be discretized and decoded back
  through the trajectory path with millimeter-scale held-out reconstruction
  error.
- Emerging but not final: a VLA can learn non-collapsed DHB invariant token
  predictions, and token learning improves substantially over the collapsed
  frozen-OAT policy.
- Not strong enough yet: token-native closed-loop policies improve task success
  over continuous DHB or regular action heads.

## 2026-04-24 Reduced-Codebook Tokenizer Probe

To test whether exact token prediction is failing partly because the frozen OAT
target is too high-cardinality, two standalone 256-way OAT tokenizer probes
were run before any StarVLA retraining:

| Probe | Codebook | Registers | Gate | R2 | MAE | Unique IDs | Effective IDs | Per-position unique |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `fsq4x4x4x4` | `256` | `8` | fail | `0.244400` | `0.062272` | `218` | `110.861` | `[101,101,96,108,58,56,59,1]` |
| `fsq4x4x4x4_k7` | `256` | `7` | fail | `0.209199` | `0.064328` | `220` | `100.434` | `[118,110,66,82,112,60,1]` |

Interpretation:

- A 256-way codebook can still reconstruct held-out DHB invariant chunks with
  positive R2 and broad global token usage.
- The strict gate correctly fails because the final register collapses to a
  static token. This is not the old global collapse, but it would still waste a
  supervised token slot.
- The next reduced-cardinality policy experiment should either trim/ignore the
  static tail register during token CE or use a slightly larger/factorized
  target that preserves per-position diversity.

## 2026-04-24 Reduced-Codebook StarVLA Token Policy

A bounded 1000-step policy warmup was run with the 256-way, 8-register OAT
checkpoint:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1`

Warm-start source:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_tokenwarm1/checkpoints/steps_1000_pytorch_model.pt`

Loader safeguard: old OAT and token-branch tensors were skipped because the
codebook shape changed, so the run preserved the configured frozen 256-way OAT
tokenizer and initialized the token branch fresh.

Held-out sweep:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1/discrete_offline_sweep_step1000_normfix_gtcodec.json`

| Metric | Value |
| --- | ---: |
| continuous invariant MAE | `0.100575` |
| predicted unique token sequences | `8 / 24` |
| predicted global token IDs | `13` |
| token accuracy / majority baseline | `0.104167 / 0.067708` |
| predicted per-position unique | `[2,3,3,2,2,3,3,2]` |
| GT global token IDs | `80` |
| full continuous-vs-discrete position MAE | `0.097732 m` |
| full endpoint drift | `0.357411 m` |
| GT-token codec position MAE | `0.003903 m` |
| GT-token codec endpoint drift | `0.015620 m` |

Interpretation:

- This is the first reduced-codebook policy checkpoint that clears the loose
  offline token gate: no fixed-sequence collapse, token accuracy beats the
  majority baseline, endpoint drift is under `0.40 m`, and two monotonicity
  flags pass.
- It is not yet the best executable discrete checkpoint. `tokenwarm1/steps_1000`
  still has much lower endpoint drift (`0.180139 m`).
- The strongest next experiment is now a second reduced-token run with either
  longer token-only training or a slightly larger/factorized codebook that keeps
  the token-learning gains while recovering endpoint fidelity.

Regression smoke:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenwarm1/predict_smoke_step1000_discrete.json`

- `predict_trajectory(... use_discrete_tokens=True)` runs end to end.
- first predicted token sequence:
  `[39, 107, 51, 107, 106, 43, 51, 170]`
- chunk action MAE on the smoke batch: `0.177422`

## Reduced-Codebook Token-Only Follow-Up

Run:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1`

This warm-started from `fsq4_tokenwarm1/steps_1000`, froze Qwen plus the
continuous trajectory-producing branch, and optimized token CE only.

Held-out sweep:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1/discrete_offline_sweep_step1000_normfix_gtcodec.json`

| Metric | Value |
| --- | ---: |
| continuous invariant MAE | `0.100575` |
| predicted unique token sequences | `8 / 24` |
| predicted global token IDs | `13` |
| token accuracy / majority baseline | `0.114583 / 0.067708` |
| full continuous-vs-discrete position MAE | `0.098059 m` |
| full endpoint drift | `0.358955 m` |
| GT-token codec position MAE | `0.003903 m` |
| GT-token codec endpoint drift | `0.015620 m` |

Result: token-only training improves token accuracy slightly but does not
improve endpoint fidelity. Do not spend more cycles on the same CE-only
objective. The next high-impact token experiment should use factorized or
trimmed token targets so the policy is not forced to predict static or weakly
informative registers with the same objective weight as trajectory-critical
registers.

Regression smoke:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_fsq4_tokenonly1/predict_smoke_step1000_discrete.json`

- `predict_trajectory(... use_discrete_tokens=True)` runs end to end.
- first predicted token sequence:
  `[39, 106, 35, 107, 106, 43, 35, 170]`
- chunk action MAE on the smoke batch: `0.177422`

## 2026-04-26 Morning Update: Reach-Preserving Gripper Calibration

Repository checkpoints:

- main repo commit: `778a3a9` (`Validate DHB-XR reach protocol and gripper gaps`)
- nested `vendor/starVLA` commit: `02580df`
  (`Add DHB-XR action and token training support`)
- regression tests before the checkpoint commit:
  `59 passed` across the action-head, chunk-rollout, OAT-tokenizer, and DHB
  chunk-target tests

The corrected reach protocol changed the interpretation of the
closed-transport residual branch. Under the paper-safe contract
(`benchmark` init, raw LIBERO images, direct-action source, `1.0,1.35,1.15`
XYZ scale, policy-rate state history, and `140` policy steps), the branch
actually preserves reach:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_closed_transport_hardgate.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_closed_transport_hardgate_fromcontact400`
- artifact:
  `live_reach_corrected_raw_max140_5case.json`
- result: `5/5` reach successes at both checked checkpoints, with average
  minimum dynamic bowl XY about `41.4 mm` and `0` IK failures

The earlier apparent reach regression was therefore another evaluation-contract
artifact, not a true model regression.

The new grip-calibration branch freezes the validated reach/action field and
trains only generic gripper/contact heads plus the physical-gripper-state-gated
closed residual:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_closed_gripcal.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_closed_gripcal_fromhard300`
- warm start:
  `actionres_closed_transport_hardgate_fromcontact400/steps_300`
- trainable parameters during training: about `11.8M`
- final checkpoint: `steps_600_pytorch_model.pt`
- reach gate:
  `live_reach_corrected_raw_max140_5case.json`
- reach result: `5/5`, average minimum dynamic bowl XY `41.5 mm`, average
  minimum dynamic bowl XYZ `77.3 mm`, and `0` IK failures

Full-task no-gate diagnostics on `libero_spatial` task `0`, episode `1`,
offset `+3 cm`:

- threshold `0.50`: no close; max learned close probability is about `0.392`.
- threshold `0.38`: still no close with two-step debounce because the
  probability spikes are too short.
- threshold `0.37`: closes at step `109`, with EE relative to the dynamic bowl
  about `[35.7, -31.8, 11.6] mm`, but still no reopen or successful
  transport/release.

Interpretation:

- The gripper polarity, basic reach, IK feasibility, and object-specific gate
  avoidance are no longer the primary blockers.
- The learned close signal is under-calibrated but peaks at the right late
  phase under delayed offset reach.
- The remaining full-task failure is post-contact transport/release and
  delayed-observation gripper calibration, not a scalar object-distance gate.

Next most impactful actions:

1. Keep the calibrated direct-action reach branch as the current clean
   embodied evidence.
2. Use the grip-calibration branch as the active full-task diagnostic branch
   because it preserves reach while exposing the real post-contact gap.
3. Do not add environment/object-specific close gates to the policy. If object
   geometry is used, keep it restricted to offline label generation or
   diagnostics.
4. The next training run should add delayed-offset or rollout-prefix gripper
   supervision and a post-contact transport/release objective while preserving
   the frozen base reach field.
5. For storytelling, claim robust offset reach and a reach-preserving path
   toward learned contact; do not claim full pick-place until reopen/transport
   succeeds without object-specific gates.

## 2026-04-26 Follow-Up: BCE Gripper-State Head Narrows the Gap

Implementation update:

- fixed the backend so a single-logit BCE gripper-state head can consume the
  existing categorical `dhb.gripper_state` labels by converting them to a
  closed-indicator target
- added a regression test for categorical labels flowing into the BCE
  gripper-state loss
- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_closed_gripbce.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_closed_gripbce_fromgripcal600`
- warm start:
  `dhb_starvla_qwen_fresh_actionres_closed_gripcal_fromhard300/steps_600`
- checkpoint:
  `steps_800_pytorch_model.pt`

Validation:

- smoke: `scripts/starvla_dhb_train_step_smoke.py` passed for the BCE config
- unit test: `tests/test_action_head.py -q` passed (`36 passed`)
- corrected 5-case reach gate: `5/5`, average dynamic bowl XY `41.4 mm`,
  `0` IK failures

Full-task no-gate diagnostics on task `0`, episode `1`, offset `+3 cm`:

- threshold `0.50`: no close; max learned close probability about `0.421`
- threshold `0.35`: closes at step `109`; close relative to bowl is about
  `[35.5, -31.8, 11.7] mm`; still no reopen or successful transport/release

Interpretation:

- BCE improves the learned close-probability scale compared with the CE
  branch (`~0.42` max instead of `~0.39`) and preserves reach.
- It does not change the behavioral failure mode. Once close is triggered at
  a plausible bowl-relative pose, the policy still lacks reliable post-contact
  transport and release.
- Do not spend more time only lowering thresholds or swapping CE/BCE losses.
  The next meaningful work must train a post-contact residual/action module
  that is active after physical gripper closure while preserving the base
  reach behavior.

## 2026-04-26 Overnight Update: Full Task Is Now A Precision-Contact Problem

Two staged branches were added and tested:

- `configs/libero_dhbxr_qwen_fresh_actionres_postcontact.yaml`
- `configs/libero_dhbxr_qwen_fresh_actionres_releasegrip.yaml`

The post-contact residual branch warms from the BCE gripper checkpoint and
trains only the physical-gripper-gated closed residual. It preserves the
corrected reach gate:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_postcontact_fromgripbce800`
- checkpoint: `steps_800_pytorch_model.pt`
- reach gate: `5/5`, average minimum dynamic bowl XY `41.4 mm`, `0` IK
  failures
- full-task no-gate diagnostic at threshold `0.35`: `success=false`,
  `close_step=101`, `ik_fail_total=16`, bowl motion about `80.1 mm`

Compared with the previous BCE checkpoint, this is a modest real improvement:
IK failures drop from `30` to `16`, object motion increases from about
`74-75 mm` to `80 mm`, and final bowl-plate XY improves slightly. It is not a
full-task breakthrough.

The release-gripper branch warms from post-contact and trains only the generic
contact-phase/gripper-state path. It also preserves reach:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_releasegrip_frompost800`
- checkpoint: `steps_600_pytorch_model.pt`
- reach gate: `5/5`, average minimum dynamic bowl XY `41.5 mm`, `0` IK
  failures

However, release training over-calibrates close probability under the default
threshold:

- threshold `0.35`: closes at step `65`, about `124 mm` XY / `115 mm` Z from
  the bowl; no reopen; no success
- threshold `0.90`: closes at step `93`, about `56 mm` XY / `45 mm` Z from the
  bowl; no reopen; no success
- `contact_phase` decoder: no close event, so it is not currently usable as a
  rollout gripper source

The highest-signal new finding is that the old `50 mm` reach success threshold
is too loose for physical grasp. With direct-action Y scale increased from
`1.35` to `1.55`, the same release checkpoint improves the 5-case reach metric
from `41.5 mm` to `34.8 mm` average XY. On the full task with
`--direct-action-xyz-scale 1.0 1.55 1.15` and learned close threshold `0.90`,
close geometry improves to about `[1.5, -25.5, 44.7] mm` relative to the bowl,
and the closest dynamic EE-bowl approach reaches `24.7 mm` XY / `43.4 mm` 3D.
A delayed threshold `0.94` shows the trajectory can reach `17.1 mm` XY /
`30.2 mm` 3D, but it closes too late at step `177` after IK failures.

Decision:

1. Keep both new configs as diagnostic branches because they preserve reach and
   expose the next bottleneck cleanly.
2. Do not claim full pick-place success yet.
3. Replace the old reach-only gate for grasp work with a stricter
   contact-ready gate: target `<=20 mm` dynamic bowl XY and `<=35-40 mm` 3D at
   or before the learned close event.
4. Do not continue gripper-head-only training as the main path. It can preserve
   reach, but it does not learn release from off-demo rollout states and can
   make close timing too early.
5. Next training should target high-precision contact approach while preserving
   the base reach path: residual-only action correction, current-state history,
   strong close/contact sampling, and explicit validation against the stricter
   contact-ready metric before returning to full release.

## 2026-04-26 Contact-Precision Residual Diagnostic: Negative

A narrower pre-contact residual branch was trained to test whether the
learned approach could be tightened without touching the base direct-action
head, the physical-gripper-gated closed residual, or the learned gripper
heads:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_contactprecision.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_fromrelease600`
- warm start:
  `dhb_starvla_qwen_fresh_actionres_releasegrip_frompost800/steps_600`
- trainable path: the small additive action residual adapter only
- final checkpoint: `steps_400_pytorch_model.pt`

This branch failed the corrected reach-preservation gate:

- reach artifact:
  `live_reach_corrected_raw_max140_5case.json`
- result: `2/5` reach successes
- average minimum dynamic bowl XY: `70.8 mm`
- average minimum dynamic bowl XYZ: `73.6 mm`
- average IK failures: `0`

The result is useful because it rules out a tempting but unsafe fix. Even a
small residual with low residual L2 can systematically pull the open-loop
approach field away from the bowl under offset rollout. This means the next
training step should not be "more pre-contact residual tuning" unless it
includes an explicit reach-preservation or base-action-consistency constraint
that is validated before any full-task rollout.

Updated overnight priorities:

1. Keep the `actionxyz_from2000/steps_400` and release/post-contact reach
   gates as the paper-safe behavior evidence.
2. Treat `contactprecision_fromrelease600/steps_400` as a negative diagnostic,
   not as a candidate for full-task videos.
3. If contact precision is revisited, use a constrained objective that
   preserves base endpoint direction and endpoint proximity, not a standalone
   weighted action residual.
4. Spend the next compute on claim closure: matched QwenDHB/QwenOFT tables,
   strict contact-ready diagnostics, and paper-ready reach videos rather than
   unconstrained residual fine-tunes.

## 2026-04-26 Implementation Follow-Up: Constrained Residual Objective

The contact-precision failure showed that per-step residual L2 was not enough:
small residuals can accumulate into a large endpoint drift over the chunk. A
new optional loss was added for residual branches:

- `lambda_action_base_cumsum_l2`
- `action_base_cumsum_dims`

When enabled, the action head compares the cumulative sum of the predicted
residual-corrected action against the frozen base action path over the selected
leading action dimensions. This directly penalizes accumulated XYZ drift while
still allowing bounded local residual corrections.

Validation:

- `tests/test_action_head.py -q`: `37 passed`
- constrained config smoke:
  `_workspace/dhb_contactprecision_cumsum_train_step_smoke.json`

Active constrained config:

`configs/libero_dhbxr_qwen_fresh_actionres_contactprecision_cumsum.yaml`

This should be the next branch to train/evaluate if we revisit contact
precision. It must still pass the corrected `5/5` reach gate before any
full-task rollout.

Evaluation result:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_cumsum_fromrelease600`
- checkpoint: `steps_400_pytorch_model.pt`
- corrected reach gate: `5/5`, average dynamic bowl XY `39.5 mm`, average
  dynamic bowl XYZ `78.3 mm`, and `0` IK failures
- full-task no-gate diagnostic with Y-scale `1.55` and gripper-state threshold
  `0.90`: no success, close at step `101`, close geometry `22.3 mm` XY /
  `33.6 mm` 3D, minimum dynamic bowl distance `18.9 mm` XY / `33.3 mm` 3D,
  bowl motion `98.5 mm`, `29` IK failures, no reopen
- threshold `0.92` delays close to step `129`, worsens close XY to `35.8 mm`,
  and raises IK failures to `39`

Interpretation: the cumulative base-action constraint fixes the reach
preservation failure of the unconstrained contact-precision branch and reaches
the strict contact-ready regime. It still does not solve release or late
placement IK. The next training target should be post-contact transport and
learned reopen, not more close-threshold sweeps.

## 2026-04-26 Closed-Transport Follow-Up: Neutral/Negative

A closed-state transport branch was trained from the contact-ready cumsum
checkpoint while freezing the open-gripper reach/contact path and learned
gripper heads:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_contactready_transport.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactready_transport_fromcumsum400`
- checkpoint: `steps_500_pytorch_model.pt`
- trainable path: physical-gripper-gated closed residual only
- corrected reach gate: `5/5`, average dynamic bowl XY `40.1 mm`, average
  dynamic bowl XYZ `78.2 mm`, and `0` IK failures

Full-task no-gate diagnostic with Y-scale `1.55` and gripper-state threshold
`0.90`:

- no success, no reopen
- close step `93`
- close geometry `23.1 mm` XY / `48.4 mm` 3D
- minimum dynamic bowl distance `23.1 mm` XY / `44.2 mm` 3D
- bowl motion `94.7 mm`
- minimum plate XY `81.4 mm`
- final bowl-plate XY `131.3 mm`
- `28` IK failures

Interpretation: this branch slightly improves plate proximity and IK count
relative to the cumsum full-task diagnostic, but it worsens contact quality and
does not learn release. Treat it as neutral/negative. The next transport
attempt needs better post-contact state distribution or explicit release-phase
supervision, not simply more closed-residual training from demo chunks.

## 2026-04-26 Streaming Gripper Decoder Check

The receding-horizon gripper decoder now preserves hysteresis/debounce state
across replanned chunks. This fixes a generic streaming-policy issue: if a
learned close/open transition begins at the end of one predicted chunk and
continues in the next chunk, the debounce counter should carry across the
chunk boundary instead of resetting every replan.

Targeted tests pass for this cross-chunk transition case. However, replaying
the strongest saved full-task traces through the fixed decoder does not expose
a hidden reopen. The current learned gripper-state signal has a repeated
chunk-slot artifact: the first three executed slots of each chunk predict
closed, while the final executed slot often predicts open. The next chunk then
returns to closed immediately. With proper hysteresis this is not a stable
release decision, so the saved no-reopen diagnosis remains valid.

Updated next action:

1. Keep the streaming decoder fix because it is conceptually correct for VLA
   execution and prevents future cross-chunk debounce bugs.
2. Do not lower debounce just to accept one-slot release spikes; that recreates
   the observed open-close-open jitter.
3. Treat release as a training/data problem: the release head must become
   stable across replans from post-contact observations, not merely produce a
   chunk-tail open prior.

## 2026-04-26 Late Update: Command-State Gripper Alignment

The latest closed-transport failure had an implementation-level cause: the
policy conditioned closed residuals on observed physical gripper qpos, while
the live controller commands close immediately but the observed LIBERO qpos
lags for many policy steps. In deployment terms, the policy should be allowed
to condition on the command stream it just issued, not only the delayed
actuator measurement.

Implemented fixes:

1. `scripts/starvla_dhb_embodik_live_rollout.py` now supports
   `--state-gripper-source observed|commanded`.
2. `scripts/starvla_dhb_live_checkpoint_sweep.py` forwards the same option.
3. `vendor/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py` supports
   `datasets.vla_data.state_gripper_source: commanded` so training and live
   streaming use the same state convention.
4. The trace records both the policy-state gripper value and the observed qpos
   so future diagnostics can detect this mismatch directly.

Validated branches:

- Release-stable from the cumsum checkpoint trains generic gripper/contact
  heads without a fixed chunk slot. It removes the earlier fixed-slot artifact
  but is under-confident and still fails full pick-place.
- Commanded transport from the release-stable checkpoint preserves the
  corrected reach gate at `5/5` with `0` IK failures.
- Commanded transport XY trains only the closed residual and reaches the
  strongest post-close stability so far: full-task diagnostic has `0` IK
  failures, but still no success.

Current best full-task diagnostic:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_transport_xy_fromcommand800`
- checkpoint: `steps_500_pytorch_model.pt`
- state convention: commanded gripper proxy
- threshold/open/debounce: `0.45 / 0.05 / 2`
- close step: `121`
- close geometry: about `29.2 mm` XY and `15.3 mm` vertical offset from the
  bowl
- best approach: about `19.2 mm` dynamic bowl XY near step `97`
- IK failures: `0`
- outcome: no placement success, no stable release

Interpretation: this is meaningful progress because the late IK-collapse
failure is mostly gone. It is not yet the breakthrough because the learned
close happens after the best approach and still outside the likely physical
grasp tolerance. The next compute should not be another transport residual
sweep. It should improve contact-centered approach and close timing while
preserving the command-state reach gate.

Next unattended target:

1. Train a contact-centering residual from the command-state XY checkpoint or
   command-state transport checkpoint.
2. Preserve the base reach field with cumulative base-action loss and a strict
   corrected 5-case reach gate.
3. Oversample close/contact approach chunks, but avoid a fixed gripper event
   slot because that previously created chunk-position artifacts.
4. Use the objective gate `<=15-20 mm` dynamic bowl XY at learned close before
   returning to release/placement training.
5. After contact close is physically plausible, revisit integrated gripper
   release and matched QwenDHB/QwenOFT comparisons for the paper table.

## 2026-04-26 Overnight Update: Contact-Close Gate Passed

The contact-centering branch completed and moved the active blocker forward:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_contactcenter.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_fromxy500`
- checkpoint:
  `steps_500_pytorch_model.pt`
- corrected reach gate:
  `5/5`, average dynamic bowl XY `37.3 mm`, average dynamic bowl XYZ
  `78.7 mm`, and `0` IK failures

With `--direct-action-xyz-scale 0.85 1.55 1.15`, the old gripper timing was
too late, but the geometry became strong: minimum dynamic bowl XY reached
`7.2 mm` and minimum dynamic bowl XYZ reached `21.0 mm`. This showed the
pre-contact action field can reach contact-quality geometry.

The follow-up contact-center gripper continuation then learned a better
generic close timing without object-distance gates:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_contactcenter_grip.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_grip_fromcenter500`
- checkpoint:
  `steps_800_pytorch_model.pt`
- best traced full-task diagnostic:
  close at step `103`, close-time dynamic bowl XY `18.0 mm`, vertical offset
  `22.4 mm`, minimum dynamic bowl XY `8.2 mm`, and `0` IK failures

Interpretation: the contact-close gate is now passed for the active
episode/offset diagnostic. The remaining failure is post-contact physical
interaction: the bowl is contacted and pushed/slid, but not reliably lifted,
transported, and released on the plate. Final bowl-plate XY in the best trace
is still about `116.8 mm`.

Next unattended target:

1. Commit the contact-center configs and notes as a stable checkpoint.
2. Train a post-contact transport/release residual from
   `dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_grip_fromcenter500/steps_800`.
3. Freeze the validated pre-contact contact-centering action path and the
   learned gripper timing heads; train only a commanded-gripper-gated closed
   residual.
4. Gate progress by object outcome, not only EE geometry: bowl lift, bowl
   displacement toward the plate, lower final bowl-plate XY, no IK regression,
   and a learned reopen near the plate.
5. If this staged closed residual does not improve object transport, stop
   residual sweeps and change the data distribution with rollout/contact
   states, because demo-only post-contact chunks are then the limiting factor.

## 2026-04-26 Late Root Cause: Center Contact Is the Wrong Grasp Proxy

The post-contact branch was trained and evaluated from the contact-close
checkpoint:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_postcontact_fromcontactgrip.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_postcontact_fromcontactgrip800`
- trainable scope:
  `0.267M` parameters, only `action_closed_residual_adapter`
- corrected reach gate:
  `5/5`, average bowl XY about `37.4 mm`, `0` IK failures for both
  `steps_250` and `steps_500`

The best full-task trace at `steps_500` improves learned sequencing but not
object success:

- close step:
  `111`
- reopen step:
  `201`
- close EE-bowl offset:
  `[4.5, -5.3, 19.6] mm`
- minimum EE-bowl 3D:
  `20.7 mm`
- best bowl-plate XY:
  about `105 mm`
- IK failures:
  `76`

The important new finding is that the previous center-contact gate was wrong
for this bowl task. The demo event-geometry probe shows that successful demos
close from a side grasp:

- script:
  `scripts/libero_demo_event_geometry_probe.py`
- artifact:
  `_workspace/paper_eval/2026-04-26_demo_event_geometry_task0_bowl_plate.json`
- physical close mean over 50 demos:
  `[-1.5, -49.9, +26.3] mm` EE-minus-bowl, `52.1 mm` XY
- open command/physical open occurs when bowl-plate XY is about `19-22 mm`

Interpretation: the current model did what we asked it to do in the wrong
metric. It centered the EE over the bowl, which improves EE-bowl distance but
causes pushing/sliding rather than side grasp/lift. The remaining breakthrough
path is now better defined: learn an affordance-relative grasp pose and
release pose, not object-center contact.

Next unattended target:

1. Stop using `<=20 mm` bowl-center close as a success gate for bowl grasp.
2. Add an affordance-relative grasp metric to diagnostics: close vector error
   to the demo event mean and bowl-plate XY at reopen.
3. Regenerate or augment training labels with object-relative event geometry
   from simulator states, keeping object geometry as offline supervision or
   diagnostics rather than deployment-time gates.
4. Train the next residual/contact branch to preserve the demo side-grasp
   offset around close, then transport until bowl-plate XY reaches the demo
   release band.
5. Keep the current post-contact branch as a positive partial result:
   reach preserved, learned close and learned reopen present, but physical
   grasp affordance still wrong.

## 2026-04-26 Overnight Metric Closure: Affordance Scorer Added

The affordance-relative diagnostic is now implemented:

- script:
  `scripts/libero_rollout_affordance_score.py`
- tests:
  `tests/test_libero_rollout_affordance_score.py`
- score artifact:
  `_workspace/paper_eval/2026-04-26_affordance_scores_postcontact.json`

The scorer compares rollout close geometry to the demo physical-close
EE-object vector and compares release/final state to demo object-target
placement. This is the correct gate for the next branch because it measures
the actual grasp affordance and object outcome rather than center proximity.

Latest scores:

- Contact-center + learned gripper closes at `18.0 mm` center XY but has
  `37.1 mm` close-vector error to the demo side grasp; it never reopens and
  ends at `116.8 mm` bowl-plate XY.
- Post-contact best closes at `6.9 mm` center XY but has `45.6 mm`
  close-vector error; it reopens at `118.6 mm` bowl-plate XY and ends at
  `105.8 mm`.
- The `-0.3 mm/step` diagnostic Y bias passes the close-affordance gate
  (`18.7 mm` vector error), but still reopens/finalizes at about
  `128/120 mm` bowl-plate XY.
- The side-plus-down bias closes late, increases IK failures, and does not
  improve placement.

Updated execution rule:

1. Do not spend more unattended compute on scalar execution biases.
2. Use the scorer table as the required full-task diagnostic for every new
   branch.
3. The next trainable branch should add offline affordance-relative
   supervision from simulator/demo geometry: demo close EE-object vector,
   post-close object attachment/lift, and release object-target band.
4. Deployment must remain object-free: object and target geometry are allowed
   for dataset generation, diagnostics, and benchmark scoring, not for
   rollout-time hardcoded gates.

## 2026-04-26 Affordance Dataset Regeneration

The offline training path has been updated so the next branch can train on the
correct grasp proxy:

- converter:
  `examples/integration/libero_to_lerobot.py`
- new converter mode:
  `--geometry-close-label-mode demo_affordance`
- regenerated dataset:
  `/home/andypark/Projects/data/libero_lerobot_affordance`
- dataset size:
  `500` episodes, `62,250` frames, about `208 MB`
- label coverage:
  `500/500` episodes have close labels, `470/500` have release labels,
  `2,929` close-phase steps, and `2,853` release-phase steps
- StarVLA mixture:
  `libero_spatial_dhbxr_affordance`
- next configs:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp.yaml`
  and
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_grip.yaml`

The one-step train smoke passed on the affordance-grasp config. The smoke
script only freezes Qwen, so its trainable-parameter count is not the final
trainer freeze count; use the actual trainer log for the real trainable scope.

Next unattended run:

1. Train `affordancegrasp` from the commanded transport XY checkpoint.
2. Re-run the corrected 5-case reach gate.
3. Run the full-task trace and score it with
   `scripts/libero_rollout_affordance_score.py`.
4. Only if reach is preserved and close-vector error improves, train the
   `affordancegrasp_grip` timing branch.

## 2026-04-27 Coupled Post-Contact Result

The next branch has now been executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_coupled_postcontact.yaml`
- checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_coupled_postcontact_fromgrip800/checkpoints/steps_600_pytorch_model.pt`
- gated diagnostic:
  close step `93`, close-vector error `36.0 mm`, `0` IK failures, no reopen
- true no-gate diagnostic:
  close step `81`, reopen step `93`, close-vector error `57.6 mm`, reopen
  bowl-plate XY `129.2 mm`, `0` IK failures

Important harness correction:

- `scripts/starvla_dhb_live_checkpoint_sweep.py` now forwards
  `--disable-gripper-gates`.
- Older sweep artifacts whose filenames say `nogates` are only truly no-gate if
  their JSON contains `"disable_gripper_gates": true`.
- The remaining technical gap is learned phase persistence after close. The
  model can produce release without object gates, but it releases before object
  transport. Next work should train/regularize a hold-transport phase rather
  than only changing scalar gripper thresholds.

## 2026-04-27 Phase-Persistence Result

The phase-persistence branch has now been executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_phasepersistent.yaml`
- checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_phasepersistent_fromcoupled600/checkpoints/steps_300_pytorch_model.pt`
- true no-gate diagnostic:
  close step `80`, reopen step `197`, close-vector error `60.2 mm`, reopen
  bowl-plate XY `118.1 mm`, final bowl-plate XY `116.8 mm`, and `0` IK
  failures
- trace-scored no-gate diagnostic:
  close step `79`, reopen step `233`, close-vector error `63.2 mm`, best
  close-affordance error `26.1 mm` at step `123`, minimum bowl XY `4.5 mm`,
  and `0` IK failures
- threshold diagnostic:
  `close=0.85/open=0.40` closes at step `83` with `51.6 mm` close-vector
  error and no reopen
- contact-phase decode diagnostic:
  closes at step `2` with `343.3 mm` close-vector error

Interpretation: phase persistence improved the learned gripper sequence, but
close is still a phase prior rather than a contact-ready decision. The robot
physically reaches the bowl around steps `136-148`; the model closes around
steps `80-83`. The trace-scored best opportunity confirms that the same
trajectory later becomes much closer to the demo side-grasp affordance, so the
next useful implementation is contact-ready supervision/calibration rather than
another threshold sweep.

## 2026-04-27 Pre-Contact-Open Negative Result

The pre-contact-open branch has also been executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_precontactopen.yaml`
- purpose:
  keep the motion heads frozen and oversample close phases at horizon position
  `7` to create hard open-before-contact negatives
- result:
  close step `87`, reopen step `89`, close-vector error `42.1 mm`, minimum
  bowl XY `9.3 mm`, final bowl-plate XY `122.2 mm`, and `72` IK failures

Interpretation: sampler-only hard negatives are insufficient. They improve the
close-vector score but destabilize phase persistence and IK. The heavy
checkpoint/final-model artifacts were deleted; lightweight summaries and score
JSONs remain.

## 2026-04-27 Sparse Contact-Event Negative Result

The checkpoint sweep now supports explicit learned-gripper decoder selection:

- flag:
  `--model-gripper-decode-source`
- supported values:
  `auto`, `gripper_state`, `contact_phase`, `contact`, `normalized_action`
- regression test:
  `tests/test_live_checkpoint_sweep.py`

The sparse generic contact-event branch has also been executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_contactevent.yaml`
- purpose:
  train the existing generic `contact_head` on 3-way sparse close/open event
  labels, then decode with `--model-gripper-decode-source contact`
- result:
  close step `3`, no reopen, close-vector error `339.3 mm`, minimum bowl XY
  `160.3 mm`, final bowl-plate XY `129.6 mm`, and `0` IK failures

Interpretation: the existing sparse event supervision is also chunk-time-biased
under offset-stretched rollout. Switching among dense state, contact phase, and
generic contact heads does not solve contact readiness by itself.

Updated next unattended target:

1. Do not continue threshold-only sweeps or sampler-only gripper branches.
2. Keep the phase-persistence checkpoint as the current best learned-gripper
   evidence.
3. Add a contact-ready supervision path that is offline-geometry-derived for
   training, but object-free at deployment. The target should be sparse around
   the demo side-grasp affordance and should explicitly penalize early close on
   pre-contact approach frames.
4. Preserve post-close hold/transport persistence separately from close-event
   timing so the model does not collapse back to immediate close/open pulses.
5. Re-evaluate every new branch with the affordance scorer and true
   `"disable_gripper_gates": true` JSON verification.

## 2026-04-27 Current-Alignment Result And Next Target

The current-alignment implementation and two follow-up branches have now been
tested:

- implementation:
  `vendor/starVLA/starVLA/model/framework/VLM4A/QwenDHB.py`
- tests:
  `tests/test_qwendhb_target_alignment.py`
- full current branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentgrip.yaml`
- hybrid current-phase branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentphase.yaml`

Result summary:

- Full current alignment makes close timing more state-dependent but loses
  hold persistence. The no-gate rollout closes near bowl contact at step `127`
  and reaches `6.9 mm` minimum bowl XY, but reopens at step `129`, has `81` IK
  failures, and does not improve the close-vector score enough.
- Hybrid current-phase alignment preserves stability but does not fix early
  close. The no-gate rollout has close step `77`, reopen step `233`, `0` IK
  failures, `4.7 mm` minimum bowl XY, and `69.6 mm` close-vector error. The
  same trace reaches a `25.9 mm` best close-affordance opportunity at step
  `122`.

Next implementation target:

1. Keep dense `dhb.gripper_state` future-aligned for closed-state persistence.
2. Add a current-aligned pre-contact open/contact-ready margin that applies
   only to the prefix actually executed before the next replan, not to the
   whole action horizon.
3. Avoid deployment-time object-distance gates. Offline simulator geometry may
   continue to define labels and scores, but rollout decisions must come from
   the learned heads, images, language, proprioception, history, and DHB/action
   context.
4. Gate the next branch on three metrics simultaneously: actual close-vector
   error below the current `42.1 mm` best learned no-gate score, no immediate
   close/open pulse, and low IK failures.

## 2026-04-27 Currentready Result And Remaining Gap

The prefix-only current target was tested and is not enough:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentprefix.yaml`
- result:
  dense gripper-state decoding closes at step `80`, reaches `0.34 mm`
  minimum bowl XY, keeps `0` IK failures, but still has `60.2 mm`
  close-vector error and `112.1 mm` final target XY
- interpretation:
  current-prefix labels alone do not break the early-close prior

The two-signal currentready branch is the new best learned no-gate close result:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready.yaml`
- decoder:
  `--model-gripper-decode-source contact_gripper_state`
- result:
  close step `103`, `0` IK failures, `10.2 mm` minimum bowl XY, `37.1 mm`
  close-vector error, and `19.0 mm` close XY
- failure:
  no reopen and `118.0 mm` final target XY

Updated unattended target:

1. Preserve the currentready close-readiness head; it is now the strongest
   learned close-timing evidence.
2. Add/validate release diagnostics that expose dense state-close probability
   after the hybrid close transition, because release is now the blocker.
3. Try a release/transport branch that starts from currentready, freezes the
   contact-readiness head, and focuses dense gripper-state/contact-phase
   supervision on terminal/release chunks.
4. Keep rollout decisions object-free. Use simulator geometry only for labels,
   scoring, and diagnostic summaries.

## 2026-04-27 Object-Target Release Branch

The next active branch is now implemented:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml`
- dataset:
  `/home/andypark/Projects/data/libero_lerobot_affordance_transport`
- new offline labels:
  `dhb.object_release_ready`, `dhb.object_release_contact`,
  `dhb.object_release_event`, `dhb.object_release_phase`,
  `dhb.object_release_gripper_state`, and `dhb.object_hold_transport`

The important change is that release/hold supervision is derived from the
generic object-minus-target vector in the dataset, not from rollout-time
object gates. `QwenDHB` now supports configurable `contact_phase_field` and
`gripper_state_field`, so this branch can preserve the currentready
close-readiness head while training the phase/gripper heads against object-
target release labels. The StarVLA sampler can also oversample chunks from
`dhb.object_release_phase`.

Validation so far:

- full dataset refreshed: `500` episodes, `62,250` frames
- regression tests: `60 passed`
- train-step smoke:
  `_workspace/dhb_commanded_affordancegrasp_currentready_objectrelease_train_step_smoke.json`

Next run: train the object-release branch to `steps_150/300`, then evaluate
with fixed `rollout_seed=70`, true `--disable-gripper-gates`, and
`--model-gripper-decode-source contact_gripper_state`. The claim gate is
whether release/hold improves without losing the currentready close score or
reintroducing open-close flicker.

## 2026-04-27 Current-Phase Release Decoder Update

Implemented and tested a stricter learned gripper contract:

- close source: currentready contact probability
- open source: current-aligned `dhb.object_release_phase` class `2`
- deployment inputs: image, language, state history, DHB/action context, and
  learned head outputs only
- no deployment object-distance gates

The decoder path is `contact_release_phase` and is covered by
`tests/test_chunk_rollout.py`. The current-phase continuation config is:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_currentphase.yaml`

Important evidence under fixed `rollout_seed=70`, true no-gate, h240:

- `steps_300`, `close=0.55/open=0.55/debounce=2`: reaches `6.4 mm` minimum
  bowl XY, closes at step `83`, never reopens, final target XY `111.8 mm`,
  and has `6` IK failures.
- `steps_300`, `close=0.60/open=0.55/debounce=2`: delays close to step `99`,
  improves actual close to `16.8 mm` XY / `35.8 mm` vector error, but final
  target XY worsens to `140.3 mm` with `28` IK failures.
- `steps_300`, `close=0.62/open=0.55/debounce=2`: becomes too strict and never
  closes despite `6.7 mm` minimum bowl XY.

Interpretation: the release class is not the primary bug anymore. A traced run
shows release probability remains effectively zero while the object is still
far from the plate, which is the correct learned behavior. The full-task gap is
now close-affordance precision plus closed-object transport. The next branch
should not add object-distance rollout gates or more release-threshold sweeps.

Next most impactful branch:

1. Preserve the currentready contact head and `contact_release_phase` decoder.
2. Train close-affordance precision directly, with current-aligned labels that
   distinguish "near bowl but too high/side-offset" from true grasp-ready
   states.
3. Train closed-object transport on post-close chunks using the existing
   object-hold/transport labels, but treat release as a consequence of reaching
   target, not as a separately forced open event.
4. Gate progress on actual close-vector error below `30 mm`, final
   object-target XY below `40 mm`, low IK failures, and no deployment
   object-distance gates.

## 2026-04-27 Close-Contact Recalibration Negative Result

The first direct close-affordance recalibration branch is negative:

`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_objectrelease.yaml`

It fixed `contact_pos_weight` support in StarVLA BCE contact loss and trained
only `contact_head` from current-aligned `dhb.close_contact`. Mechanically the
run succeeded: `3.149M` trainable parameters, `loss_contact` dropped to
`0.0039`, and checkpoints were saved at steps `150/300`.

Embodied behavior failed: both checkpoints close at step `1`, with the EE about
`287 mm` XY / `271 mm` Z from the bowl at close, then never reach the bowl
(`180 mm` minimum bowl XY) and accumulate `76` IK failures.

Root-cause update: the label path is not enough by itself. The training recipe
over-sampled close-positive current frames (`gripper_phase_chunk_prob=0.95`,
`gripper_close_chunk_prob=0.90`, `gripper_event_target_pos=0`) and used a high
positive BCE weight (`8.0`), so the sparse close head collapsed into
always-close. The next attempt should use balanced hard negatives and a lower
weight, or a pairwise/calibration objective that explicitly keeps pre-contact
approach frames below threshold.

## 2026-04-27 Balanced Close-Contact Follow-Up

The balanced follow-up was executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_balanced_objectrelease.yaml`
- change:
  lower positive oversampling, `gripper_phase_sampling_key: dhb.close_contact`,
  `gripper_event_target_pos: null`, and `contact_pos_weight: 2.0`
- result at `close=0.55/open=0.55`, fixed `rollout_seed=70`, true no-gate:
  both `steps_150` and `steps_300` reach the bowl (`6.8 mm` min XY) but never
  close, then accumulate `88` IK failures
- diagnostic at `close=0.15`, `steps_150`:
  closes at step `87`, `0` IK failures, `40.0 mm` close-vector error,
  `42.7 mm` close XY, no reopen, and `126.8 mm` final target XY

A second follow-up unfreezing `contact_cross_attn` was also executed:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_xattn_objectrelease.yaml`
- trainable scope:
  `7.347M` parameters (`contact_head` plus `contact_cross_attn`)
- result:
  also never closes at the default threshold; close confidence is even lower
  than the head-only balanced branch

Updated plan:

1. Stop BCE-only close-contact sampler sweeps. The failure mode has now been
   bracketed: high-positive BCE becomes always-close, while balanced BCE
   becomes never-close or under-confident.
2. Keep the currentready/current-phase checkpoint as the best active partial,
   because it still gives a real learned close signal and the release head is
   conceptually correct.
3. Next high-impact implementation should create rollout-state hard negatives:
   collect or replay the observed approach states where the arm is near the
   bowl but too high/wrong-side, label them open, and contrast them with true
   side-grasp-ready states.
4. Prefer a pairwise/margin close-readiness objective or DAgger-style
   closed-loop relabeling over more scalar positive weights. The claim-level
   gate remains actual close-vector error below `30 mm`, no immediate flicker,
   low IK failures, and no deployment object-distance gates.

## 2026-04-27 Contact Mining and Margin Result

Implemented:

- `scripts/libero_rollout_contact_miner.py` mines per-step rollout traces into
  `positive`, `missed_positive`, `hard_negative`, and `background` labels using
  the demo close-affordance reference.
- `scripts/starvla_dhb_live_checkpoint_sweep.py --record-step-trace` now
  forwards trace recording into each child rollout so future sweeps can be
  mined directly.
- `scripts/starvla_dhb_embodik_live_rollout.py --save-policy-trace-dir` and
  the sweep-level `--save-policy-trace` flag export per-replan image/state
  tensors plus decoded close/open probabilities. This is the missing substrate
  for constructing a closed-loop hard-negative training set.
- `DHBActionHead` and the StarVLA wrapper now support generic
  `lambda_contact_margin`, `contact_positive_margin`, and
  `contact_negative_margin`.
- `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_margin_objectrelease.yaml`
  tested contact-margin calibration from the current-phase checkpoint.

Validation:

- `pytest tests/test_action_head.py tests/test_libero_rollout_contact_miner.py -q`
  passed (`43 passed`).
- The margin config train-step smoke passed and logged
  `loss_contact_margin`.
- The 300-step margin retrain completed with only `3.149M` trainable
  parameters.

Result:

- The mined currentready/current-phase traces show the active close failure is
  mostly false-positive closed-loop states, not missing confidence at true
  positives.
- The margin run over-corrects: both `steps_150` and `steps_300` never close at
  the default learned threshold on `0 cm` and `3 cm` cases. Average minimum
  bowl XY is `63.1 mm`, average IK failures are `75`, and close confidence is
  suppressed (`max ~= 0.187` at step 150, `~= 0.043` at step 300).

Updated plan:

1. Stop all demo-only contact-head recalibration attempts for now. We have
   bracketed the behavior: positive-heavy BCE always-closes; balanced BCE and
   margin training never-close.
2. Preserve the currentready/current-phase checkpoint as the best active
   partial and use it to collect rollout-state observations around both hard
   false positives and true close-affordance states.
3. Implement a closed-loop correction dataset: replay recorded states, save the
   policy observation/state tensors, and label them with offline
   demo-affordance geometry. This is the minimum intervention aligned with the
   paper story of simulation/world-model spatial augmentation.
4. Only after that dataset exists should we train a contrastive/ranking close
   readiness head or DAgger-style classifier. The next evidence gate is reduced
   hard-negative close probability while keeping true close-affordance
   probability above threshold.

Policy-trace smoke:

- Generated a trainable policy trace for the best currentready/current-phase
  checkpoint, fixed `rollout_seed=70`, case `episode=0, offset=0cm`.
- Artifact directory:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_currentphase_fromobjectrelease300/live_checkpoint_sweep/policy_trace_currentready_currentphase_step300_seed70_case0/step_300/steps_300_pytorch_model_ep0_offset0.0cm_policy_trace`
- It contains `60` compressed replan NPZ files (`8.6M` total), each with
  agentview image, wrist image, state window, predicted chunk positions, and
  close/open probabilities.
- Contact mining on that trace reports `191` hard negatives and best
  close-affordance error `35.5 mm` at step `107`. This is close to but still
  outside the `30 mm` positive gate, which reinforces that the next dataset
  should include corrective/off-manifold contact states rather than simply
  replaying this one rollout.
- The miner can now also write a policy-manifest JSONL that maps each replan
  NPZ to horizon-aligned `target_close`, `train_mask`, and
  `hard_negative_mask` arrays. The smoke manifest has `60` rows at
  `_workspace/paper_eval/2026-04-27_policy_trace_case0_contact_manifest.jsonl`.
- A non-zero-offset trace was also generated for `episode=1, offset=3cm` with
  another `60` NPZ chunks (`8.5M`). Mining reports `171` hard negatives and
  best close-affordance error `54.3 mm`, saved to
  `_workspace/paper_eval/2026-04-27_policy_trace_case1_offset3_contact_manifest.jsonl`.

Policy-trace hard-negative training:

- Added `policy_trace_contact_datasets`, a StarVLA dataloader that converts
  saved policy-trace NPZ chunks plus contact-miner manifests into normal
  QwenDHB samples. It can mix closed-loop policy samples with the LeRobot demo
  dataset, so the policy can learn from rollout observations without adding
  object-distance gates to deployment.
- Added
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_policytrace_hardneg.yaml`.
  It trains the close-readiness contact path from currentready/current-phase
  while preserving the trajectory and release path.
- Validation passed: unit tests for the dataset, the contact miner, and the
  action head pass (`46 passed`), and the actual train-step smoke logs active
  `loss_contact` and `loss_contact_margin`.
- The 300-step continuation completed using `train_starvla.py` with
  `STARVLA_USE_DEEPSPEED=false`; `train_unified.py` is not usable on this
  machine because it unconditionally initializes DeepSpeed and `CUDA_HOME` is
  absent.
- The result is negative but informative. Both `steps_150` and `steps_300`
  never close at the default learned threshold on the fixed-seed `0 cm` and
  `3 cm` h240 true-no-gate sweep. Average minimum bowl XY is still about
  `63.1 mm`, average IK failures are `75`, and max rollout close probability
  drops from about `0.15` at `steps_150` to below `0.03` at `steps_300`.
- Diagnosis: the currently exported policy manifests have `242` hard-negative
  supervised steps and `0` closed-loop positive steps. Demo positives alone do
  not counterbalance those deployment-state negatives. The next high-impact
  task is to generate closed-loop positive contact traces or replay/simulate
  positive off-manifold contact states, then retrain the same dataloader path
  with a balanced positive/negative rollout-state set.

Demo-positive/future-alignment follow-up:

- The first demo-positive policy-trace config had a target-alignment bug:
  `contact_target_alignment: current` broadcast the first chunk label and
  erased late close positives such as `[0, 0, 0, 0, 0, 0, 1, 1]`.
- The balanced config now uses `contact_target_alignment: future`, and a
  regression test covers this exact late-positive case.
- `scripts/libero_demo_contact_manifest_export.py` exports real LIBERO demo
  close windows into policy-trace NPZ+manifest rows. The task-0 export has
  `500` chunks from `50` demos, `1,650` positive supervised steps, and `2,350`
  negative steps. Combined with hard-negative policy traces, the active
  contact dataset has `562` rows and `4,242` supervised steps.
- The corrected contact-only continuation completed for `300` steps:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_demopos_policytrace_balanced_futurealign_fromcurrentphase300`.
- Focused probe: held-out demo-positive close probability rises to about
  `0.54`, while a negative chunk stays around `0.18-0.32`. This validates the
  training signal but also shows under-calibration.
- Evaluation correction: use the already validated task-agnostic direct-action
  scale `0.85, 1.55, 1.15`. Neutral `1,1,1` is a regression and should not be
  used for the main examples.
- With calibrated scale and threshold `0.50`, reach is strong but no close:
  `22.9 mm` minimum bowl XY at `0 cm`, `6.9 mm` at `3 cm`.
- With calibrated scale and threshold `0.45`, close fires in both cases:
  the `3 cm` offset closes near the bowl (`~18.9 mm` XY, `21.9 mm` Z), but
  the `0 cm` case closes too far away and neither case completes placement.

Updated next actions:

1. Treat the future-aligned contact head as the active learned-contact baseline.
2. Stop evaluating examples without the calibrated direct-action scale; it
   confounds trajectory quality with a known execution calibration.
3. Generate or replay rollout-positive contact states around the precise
   side-grasp configuration. The remaining learned-contact gap is calibration
   on rollout observations, not plumbing.
4. Revisit post-grasp transport/release after close timing is consistently
   within about `15-20 mm` XY and `20-30 mm` Z under non-zero offsets.

Rollout-positive shuffled continuation:

- Calibrated no-gate traces from the future-aligned baseline reached the bowl
  without closing: `22.9 mm` minimum XY at `0 cm`, `7.0 mm` at `3 cm`.
- Strict `30 mm` rollout-positive mining produced no positives; offline
  `45 mm` tolerance produced `111` missed-positive steps from the `3 cm`
  trace. This is a bootstrapping label only, not a deployment gate.
- A StarVLA dataloader bug/limitation was fixed: `policy_trace_shuffle: true`
  now actually shuffles policy-trace contact datasets, so short continuations
  can see appended rollout-positive rows.
- The new run
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_rolloutposfirst_shuffle_tol45_fromfuturealign300`
  trains to `steps_400` and makes the learned contact token active under true
  no-gate rollout.
- Threshold `0.50`: close at step `87` for case `0` and step `99` for `3 cm`;
  average minimum bowl XY `16.4 mm`, average IK failures `21`, no success.
- Threshold `0.55`: the `3 cm` case closes at about `6.8 mm` XY and `19.3 mm`
  Z, but case `0` closes too high and full transport/release still fails.

Revised unattended priority:

1. Preserve this checkpoint as the current learned-token baseline because it
   demonstrates object-gate-free close activation on rollout observations.
2. Build a stricter side-grasp-ready contact manifest from policy traces and
   simulator replay: positives should be close to the demo side-grasp vector,
   while negatives should include near-bowl-but-too-high and near-center
   states.
3. Retrain contact calibration with the same shuffled policy-trace loader, then
   evaluate the two fixed cases with thresholds `0.50`, `0.55`, and `0.60`.
4. Only return to post-grasp transport/release after both fixed cases close
   within roughly `15 mm` XY and `25 mm` Z without deployment object gates.

Side-grasp calibration update:

- The contact miner now supports component-wise positive caps and near-object
  hard-negative labels. The first strict side-grasp manifest produced only `9`
  positives versus `601` hard negatives, so it is useful for calibration
  pressure but too negative-heavy for a short continuation by itself.
- The strict continuation
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_fromrolloutpos400`
  suppressed live close probabilities to about `0.18-0.19` and never closed at
  threshold `0.55`.
- A lighter calibration run
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_light.yaml`
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_light_fromrolloutpos400`
  restored close activation. It closes in both fixed cases without deployment
  gates, but case `0` still closes high (`~61.6 mm` Z), case `1,+3 cm` closes
  near but not solved (`~19 mm` XY, `~21.9 mm` Z), and neither transports or
  releases successfully.

Updated priority:

1. Keep `rolloutposfirst_shuffle_tol45/steps_400` as the active learned-token
   baseline and the sidegrasp light run as calibration evidence, not as a new
   baseline.
2. Generate a larger balanced side-grasp-ready positive set. The required data
   is now specific: many rollout-observation chunks at the demo side-grasp
   vector, paired with high/central near-bowl negatives. The current strict
   manifest has the negative side but too few positives.
3. Use simulator replay or correction rollouts to create those positives before
   spending more time on post-grasp transport. Threshold sweeps are exhausted
   until both fixed cases close within the affordance window under true
   no-gate deployment.

Physical-close correction and best-positive follow-up:

- The affordance scorer now separates command close from physical close using
  `observed_state_gripper <= 0.015`. This fixes a real metric mismatch: the
  demo reference is `close_phys`, but previous rollout scoring used the command
  close snapshot.
- Corrective best-positive relabeling was added to the contact miner. The v2
  manifest has `43` positives and `567` negatives, up from `9` positives in
  strict sidegrasp v1, while preserving strict side-grasp component caps and
  near-object hard negatives.
- The bestpos v2 200-step continuation gives a clean non-zero-offset contact
  result under true no-gate deployment. For episode `1,+3 cm`, threshold `0.50`
  commands close at step `99`; physical close occurs at step `101` with
  `9.2 mm` side-grasp vector error. This is a valid learned-token side-grasp
  pass.
- The same run still fails full pick-place: case `0` closes high, no run
  reopens, and final target XY remains around `139 mm` in the non-zero-offset
  video. Debounce `4` made timing worse, so do not pursue debounce as the
  solution.

Updated immediate priority:

1. Treat bestpos v2 as the first clean learned-token side-grasp success under
   non-zero offset, but only for the contact event, not full manipulation.
2. Continue data work toward consistency across cases: more balanced positive
   rollout states around the physical side-grasp event, especially for the
   `0 cm` case that still closes high.
3. After contact consistency, train post-close hold/transport/release using the
   physical-close event as the phase boundary instead of command close.
