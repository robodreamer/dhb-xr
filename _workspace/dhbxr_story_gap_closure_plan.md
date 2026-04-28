# DHB-XR Story Gap Closure Plan

## Purpose

This note turns the current evidence into a concise story plan: what can be
claimed now, what remains weak, and which validations should come next.

## Current Big-Picture Story

DHB-XR is best framed as a geometric action interface for low-data VLA scaling.
The core idea is to separate invariant motion structure from pose, viewpoint,
controller, and embodiment-specific execution so that small real robot datasets
can be reused across offsets and settings.

The current evidence supports a continuous DHB-trained execution prototype and
a now-clean reach-only closed-loop result under calibrated direct-action
execution. It keeps discrete invariant tokens as a scaling/compression
hypothesis, not yet a validated policy path. It does not yet support the
broader low-data or world-model augmentation claim.

2026-04-28 object-pair transport update:

- The strongest integrated full-task result is still not task success. The
  broad attached lift-transport continuation reaches a useful diagnostic trend
  at step `50`: average final bowl-plate XY `96.2 mm`, average minimum EE-bowl
  XY `22.7 mm`, and `0` IK failures across the controlled `0 cm` and `+3 cm`
  sweep.
- The root cause is now narrower than "gripper failed." The policy can close
  near the bowl and move the object, but the learned closed-state action does
  not reliably produce held-object lift/carry/lower/release toward the plate.
- Scalar fixes are now ruled down. Larger closed residual scale creates lift
  but also overshoot and IK failures; generic phase-weighted action-loss
  training improves contact proximity in places but does not beat the broad
  unweighted step-50 transport result.
- The first explicit phase-object diagnostic is better than invariant-only
  continuation and nearly matches the broad baseline (`97.2 mm` average final
  bowl-plate XY at step `50`, `0` IK failures), but it still has no placement
  or release.
- The no-release gap is now an implementation/data gap, not a conceptual
  mystery: the attached positives used for that run were closed-only. The
  code path now supports generic release-phase labels in policy-trace data
  (`target_contact_release_phase` to `dhb.contact_phase`), which trains the
  same learned release head used at deployment. The next paper-relevant path
  is release-tail target-reaching positives plus explicit phase/token
  supervision, not another threshold/gain/loss-weight sweep.

Story implication: the defensible claim is robust offset reaching plus
prototype learned contact/object-motion evidence. Full pick-place remains a
specific post-contact phase/control gap until explicit phase-token transport
and matched regular-head controls are validated.

2026-04-25 late update:

- 2026-04-25 current correction: the canonical reach artifact is now the
  benchmark-init/raw-image/long-horizon rerun, not the older demo-init artifact.
  Use
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`
  for paper-safe reach claims. It gets `5/5` reach successes with `0` IK
  failures over episode/offset cases `0:0`, `0:+3`, `0:-3`, `1:+3`,
  `1:-3`. The main lesson is that non-zero-offset closed-loop evaluation must
  allow delayed reaching beyond the original demo length; episode `1,+3 cm`
  succeeds at step `96` while the demo length is `84`.
- Best current embodied result:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`
- Policy/checkpoint: `dhb_starvla_qwen_fresh_actionxyz_from2000`,
  `steps_400_pytorch_model.pt`
- Execution path: `--trajectory-source direct_action`
- Calibration: `--direct-action-xyz-scale 1.0 1.35 1.15`
- Reach-only outcome: `5/5` successes across benchmark-init episode/offset
  cases, average minimum dynamic bowl XY `39.2 mm`, with `0` IK failures
- Full-task gap: learned gripper/transport phase timing. Under the corrected
  benchmark-init/raw-image protocol, the residual contact checkpoint closes at
  step `75`, still about `100 mm` XY from the dynamic bowl, and never reopens.
  The close signal is therefore still demo-time biased under delayed offset
  reach. A diagnostic-only object-distance gate delays close to step `165`,
  but still fails transport/release, so the next claim gap is contact-quality
  transport and release, not simply gripper thresholding.
- Representative video:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3.mp4`.
  This non-zero-offset run succeeds at step `82`, has `0` IK failures, and
  ends with EE relative to bowl `[-9.2, -11.0, 78.1] mm`.

Interpretation: the current clean behavior evidence is robust offset reaching.
The next claim gap is not "can DHB-XR move the arm"; it is "can the learned
policy complete the full phase sequence early enough without object-specific
gripper heuristics."

2026-04-25 late-night refinement:

- Jointly fine-tuning the calibrated checkpoint with full action/gripper and
  contact-phase losses did not preserve reach. The diagnostic sweep for
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionfull_grip_from400`
  produced `0/2` reach successes for all tested checkpoints; the best average
  minimum bowl XY distance was `0.1569 m` at `steps_600`, far worse than the
  validated `5/5` actionXYZ result.
- This is useful root-cause evidence: the next blocker is not missing gripper
  labels in the dataset. The labels exist, but naive joint phase/gripper
  supervision can interfere with the geometry/action representation that
  currently makes reaching work.
- The current active branch therefore freezes the validated geometry/action
  path and trains only generic learned contact/gripper heads:
  `configs/libero_dhbxr_qwen_fresh_gripper_heads_only.yaml`.

2026-04-25 night gripper/contact update:

- A calibrated heads-only continuation
  (`configs/libero_dhbxr_qwen_fresh_gripper_heads_calibrated.yaml`,
  `steps_2000`) improves close probability on closed labels but over-biases the
  close class. It is a useful diagnostic, not a full-task solution.
- The gripper command sign is verified in LIBERO: `+1` closes the gripper qpos
  to about `0.0005`, while `-1` opens it to about `0.0395`.
- With the slower direct-action scale `1.0,1.35,1.35`, the policy can bring the
  EE within `22.1 mm` 3D distance of the bowl while keeping the gripper closed,
  but still fails to complete the grasp/transport task.
- A local object-relative gate delays close to a more plausible geometry but
  also fails. This means the remaining full-task gap is not just a missing
  learned close trigger; it is contact-quality grasp and post-grasp trajectory
  reliability.

Story implication: keep the main claim focused on robust offset reaching and
prototype DHB advantage over the regular head. Full pick-place should be framed
as the next contact-level training gap until sub-centimeter grasp contact and
post-grasp transport are validated without environment-specific gates.

2026-04-25 overnight action-head update:

- A deliberately bounded action-head/contact fine-tune
  (`configs/libero_dhbxr_qwen_fresh_actionhead_contact.yaml`) also failed the
  reach-preservation gate. Even with the representation stack frozen and only
  about `11.6M` parameters trainable, checkpoints `400` and `800` both got
  `0/2` reach successes in the two-case diagnostic.
- The failure mode is informative: the arm moves smoothly with no IK failures,
  but the learned direct-action head drifts away from the bowl, yielding
  average minimum bowl XY of about `23 cm`. This means the direct-action head is
  not a safely separable component under the current objective.
- Do not claim that generic contact/action fine-tuning solved full
  manipulation. The safe story remains: DHB-XR currently gives a robust
  reach/adaptation policy and a prototype matched-head advantage; the full
  grasp/transport story needs a reach-preserving residual or staged contact
  module.

2026-04-25 overnight residual-adapter update:

- A zero-initialized additive residual adapter is the first contact-training
  path that preserves the reach result. The residual branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_contact.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400`)
  gets `2/2` reach successes in the active two-case gate, with average minimum
  bowl XY `11.0 mm`, average minimum bowl XYZ `54.2 mm`, and `0` IK failures.
- The implementation preserves the base action field by freezing the original
  direct-action head and training only an additive residual initialized to
  zero, plus generic contact/phase/gripper heads. Unit tests now cover that
  zero initialization preserves base actions and that residual L2 regularization
  is reported when the adapter moves.
- A full-task no-gate diagnostic reaches very strong contact proximity
  (`4.8 mm` XY, `22.9 mm` XYZ) but does not close because the learned gripper
  probability remains under about `0.41` while the tested threshold was `0.8`.

Story implication: the clean technical message is now sharper. DHB-XR has a
validated reach/adaptation backbone, naive full-head/contact fine-tuning breaks
that backbone, and a reach-preserving residual module is the correct direction
for closing the contact/gripper gap. Full pick-place is still not claimable
until the generic learned gripper/contact signal is calibrated and transport
succeeds without environment-specific gates.

2026-04-25 continuation update:

- A higher-weight gripper/contact continuation from the residual checkpoint
  (`configs/libero_dhbxr_qwen_fresh_actionres_contact_fromres400.yaml`) keeps
  the reach backbone intact but worsens gripper timing. It still passes the
  two-case reach gate at `steps_400` and `steps_800`, but the full-task model
  gripper closes at step `45`, around `116 mm` XY and `147 mm` Z from the bowl.
- The original residual checkpoint is more useful diagnostically: lowering the
  generic gripper-state threshold to `0.395` creates contact and object motion,
  but the dynamic trace shows the first close is still early (`66 mm` XY /
  `93 mm` Z from the moving bowl). Closest dynamic proximity occurs later
  (`2.1 mm` XY / `19.2 mm` XYZ), and the rollout still fails transport/release.
- The learned `contact_phase` decoder is not yet a reliable gripper source; its
  probabilities stay low and do not trigger a close in the same no-gate
  diagnostic.

Story implication: the remaining gap is not "can the policy reach the bowl" or
"is the gripper polarity wrong." The defensible blocker is learned contact
quality and post-contact transport. The next evidence should show dynamic
object motion after close, then train a reach-preserving contact-quality module
rather than more threshold or class-weight tuning.

Dynamic-trace refinement:

- The traced original residual run with a low generic gripper-state threshold
  moves the bowl by `64.5 mm` and lifts it by `37.5 mm`, despite failing the
  task reward. Closest dynamic EE-bowl proximity occurs after close
  (`2.1 mm` XY and `19.2 mm` 3D), which means the branch can create contact and
  object motion.
- Failure is now more specifically post-contact: final bowl-plate offset is
  about `127.7 mm` XY and `33.2 mm` Z, the gripper remains closed, and the
  rollout never reaches release-ready placement.

Story implication: for the paper, call the full pick-place gap a
post-contact transport/release gap. This is stronger and more precise than
saying the policy simply fails to grasp.

Contact-ready constrained residual update:

- The unconstrained contact-precision residual branch is negative because it
  fails the corrected 5-case reach gate (`2/5`, average dynamic bowl XY
  `70.8 mm`). It should not be used for claims.
- Adding cumulative base-action preservation fixes that regression. The
  constrained branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_contactprecision_cumsum.yaml`)
  passes the corrected reach gate at `5/5`, keeps `0` IK failures, and reaches
  average dynamic bowl XY `39.5 mm`.
- In the best no-gate full-task diagnostic, the constrained branch reaches
  `18.9 mm` minimum dynamic bowl XY and `33.3 mm` 3D, closes at step `101` at
  about `22.3 mm` XY / `33.6 mm` 3D from the bowl, and moves the bowl
  `98.5 mm`. It still never reopens and ends about `140 mm` XY from the plate.

Story implication: the current best full-task diagnostic is contact-ready but
not placement-ready. This strengthens the claim boundary: DHB-XR can preserve
robust reach while getting into a plausible contact regime, but the remaining
unvalidated piece is learned post-contact transport and release.

2026-04-26 affordance-metric update:

- The demo event-geometry probe shows that the bowl task's successful grasp is
  a side affordance, not a bowl-center grasp. The physical close mean is about
  `[-1.5, -49.9, +26.3] mm` EE-minus-bowl, while demo release happens with
  bowl-plate XY around `19 mm`.
- A new rollout affordance scorer compares full-task traces against those demo
  event geometries. This replaces the misleading center-contact gate with two
  paper-safe metrics: close-vector error to the demo grasp affordance and
  object-target XY at reopen/final.
- The latest table confirms the active gap. Center-contact checkpoints can
  reach `6-18 mm` EE-bowl XY but are `37-46 mm` away from the demo side-grasp
  vector. A diagnostic `-0.3 mm/step` Y bias passes the close-affordance gate,
  but all runs remain over `100 mm` bowl-plate XY at release/final. Therefore,
  constant execution bias is not the solution; training must learn
  affordance-relative grasp and object-level transport/release.

Story implication: the narrative is cleaner now. The robust reach result is
already validated, and the full-task failure has a specific missing variable:
object-affordance-relative contact and release supervision. This directly
supports the broader DHB-XR story because the action representation can be
paired with compact geometric event targets, but those targets must be learned
from simulator/demo geometry rather than replaced by deployment heuristics.

Implementation update: the LIBERO-to-LeRobot converter now has a
`demo_affordance` close-label mode, and the regenerated
`libero_lerobot_affordance` dataset exposes that supervision through the same
generic DHB gripper/contact columns. This is an important storytelling detail:
the method is not using object-distance gates at rollout. It uses simulator
geometry only to produce better offline training labels, then deploys from
images, language, state history, DHB/action tokens, and learned heads.

2026-04-27 affordance-gripper retrain update:

- The side-affordance action residual branch preserves the corrected 5-case
  reach result: `5/5`, average dynamic bowl XY `38.2 mm`, and `0` IK failures.
  This means the offline affordance-label data path did not break the
  paper-safe reach/adaptation evidence.
- The gripper-only continuation is the first no-gate learned gripper result
  that closes reliably near the bowl under the delayed +3 cm rollout. At
  `steps_800` with generic dense-state hysteresis threshold `0.55`, it closes
  at step `83`, reaches minimum bowl XY `6.7 mm`, and has `0` IK failures.
- The same run still fails the side-grasp affordance metric: close-vector
  error is `51.8 mm`; a generic `0.70` close threshold improves it to
  `42.3 mm` but still does not pass. The best possible close-vector error in
  the trace is about `25 mm`, later than the actual close.
- Demo gripper replay remains clearly worse for non-zero offsets: it closes at
  step `42`, with `194.8 mm` close-vector error and `112` IK failures.
- A real event-decoder bug was fixed: 3-class contact-phase decoding now
  requires thresholded close/open confidence instead of toggling by argmax.
  After the fix, the current contact-phase head still closes too early, so the
  remaining issue is event calibration/supervision, not only decode logic.
- The event-BCE continuation confirms the same boundary. It trains cleanly, but
  no-gate contact-phase rollout still fails: threshold `0.55` closes at step
  `3` with `339.3 mm` close-vector error, while thresholds `0.80-0.90` delay
  closure to steps `74-75` but still miss the side-grasp affordance by
  `76-80 mm` and never release. The trace shows close probabilities ramping
  within chunks, so the head is still too chunk-time-biased.
- A release-focused dense-gripper continuation gives one useful partial result:
  `steps_300` can close and reopen with no object-distance gates
  (`close_step=137`, `reopen_step=165`). It still fails the side-grasp and
  release metrics, and a lower close threshold produces the unwanted
  close/open timing collapse (`87 -> 89`). By `steps_600`, the branch no
  longer reopens. This means release labels are available and learnable, but
  release must be coupled to a stable held-object/transport phase rather than
  trained as isolated per-step gripper state.

Story implication: the claim boundary improves but remains honest. We can say
that learned gripper state from DHB/observation context is better than demo
timing replay and works without object-distance gates. We still cannot claim
full pick-place. The remaining gap is compact and specific: learn the
side-affordance close event and object-target release/transport tightly enough
to pass the affordance scorer.

Transport residual update:

- A transport-focused residual continuation
  (`configs/libero_dhbxr_qwen_fresh_actionres_transport.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_transport_fromres400`)
  preserves the reach gate at `steps_300` and `steps_600`, but suppresses the
  learned close signal. In traced full-task runs at threshold `0.395`, it never
  closes; max close probability is only `0.282` at `steps_300` and `0.346` at
  `steps_600`.
- The branch still moves the bowl about `53 mm`, but lift is only `13-14 mm`
  and placement remains far from the plate. This means stronger terminal
  action weighting in a shared residual is not enough.

Story implication: the next implementation should not be another scalar
loss-weight sweep. The defensible next direction is a decoupled, reach-safe
model path: preserve the original reach and weak close behavior, then apply
post-contact transport adaptation only when the learned/current gripper state
indicates the object is being grasped.

Closed-state transport follow-up:

- A physical-gripper-gated closed residual branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_contactready_transport.yaml`)
  was trained from the constrained contact-ready checkpoint and preserved the
  corrected reach gate: `5/5`, average dynamic bowl XY `40.1 mm`, average
  dynamic bowl XYZ `78.2 mm`, and `0` IK failures.
- In the no-gate full-task diagnostic with Y-scale `1.55` and gripper-state
  threshold `0.90`, it closes at step `93`, reaches only `23.1 mm` minimum
  dynamic bowl XY / `44.2 mm` 3D, moves the bowl `94.7 mm`, never reopens, and
  still has `28` IK failures.
- Relative to the constrained contact-ready diagnostic, this slightly improves
  final/minimum plate proximity but worsens contact quality and does not solve
  release.

Story implication: closed residual training from demo chunks alone is not the
breakthrough path. The next model-data intervention should change the
post-contact state distribution and release supervision, not continue scalar
transport residual sweeps.

## 2026-04-25 Claim Closure Strategy

The near-term paper story should be anchored on the validated continuous-DHB
execution path, not on discrete-token closed-loop execution. The current
continuous result already has the right scientific shape: same Qwen backbone,
same short training budget, same downstream `embodik` evaluator, and a matched
regular-head control where `QwenDHB` succeeds in cases where `QwenOFT` does not.

The discrete-token story should be kept as a supporting representation result
until it clears its own policy gates. The strongest safe token claim is that
DHB invariant chunks can be compressed into discrete tokens and decoded with
millimeter-scale teacher-forced trajectory error. The token-native StarVLA
policy is promising but still not ready for closed-loop task claims because
held-out predicted-token fidelity and endpoint drift remain weak.

Priority order:

1. Freeze one canonical `predict_trajectory() -> embodik` evaluator contract.
2. Expand the matched `QwenDHB` versus `QwenOFT` comparison under that contract.
3. Package paper-grade videos and object-relative metrics for zero and non-zero
   offsets.
4. Run small-data seed-size curves only after the evaluator is stable.
5. Continue the discrete-token path only with factorized or trimmed targets,
   not more same-objective warm starts.

Safe claim boundary:

- Safe now: DHB-XR provides a structured trajectory interface that is more
  useful than a matched regular QwenOFT head in the current short-budget
  prototype evaluator.
- Safe now: DHB invariant chunks can be discretized and reconstructed with
  low held-out trajectory error under teacher-forced GT tokens.
- Not safe yet: DHB-XR generally beats regular action heads at benchmark scale.
- Not safe yet: token-native discrete DHB policies improve closed-loop success.
- Not safe yet: DHB-adapted simulation or world-model augmentation scales real
  teleop data unless seed-size and augmentation ablations are completed.

Harness-review refinement:

- The first paper-safe matched comparison must fix the episode mismatch between
  current DHB and QwenOFT artifacts by regenerating both on episodes `0,1,10`.
- Live closed-loop rollout remains diagnostic until it is robust enough; paper
  success should use the shared Stage-1 trajectory, pack, IK, and sim evaluator.
- The broader comparison should expand across `libero_spatial` task IDs
  `0,1,2` only after the task-0 matched contract is reproducible.
- The next discrete-token attempt should use factorized FSQ digit supervision
  with the low-information tail register trimmed, rather than another flat-token
  warm start.

2026-04-25 execution update:

- The corrected task-0 matched comparison is now regenerated:
  `_workspace/paper_eval/2026-04-25_matched_task0/summary.md`.
- `QwenDHB` reaches `1/3` final success at both `0 cm` and `3 cm`; matched
  `QwenOFT` remains `0/3` at both offsets.
- This preserves the prototype outcome advantage, but it also exposes the next
  gap: DHB has many late IK infeasible points (`140` total), while QwenOFT has
  none. The paper story must not overclaim execution-quality superiority until
  that gap is reduced.

2026-04-25 root-cause refinement:

- The main failure is not the DHB encoder/decoder math. A direct
  encode/decode smoke with default padding reconstructs current-aligned
  waypoints exactly.
- The implementation bug is in the StarVLA supervision contract: the converter
  exports one DHB invariant vector per episode timestep, then the dataloader
  slices `rows[t:t+8]` as if those rows were chunk-local action targets. For
  DHB-DR this is invalid because the double-reflection frame initialization is
  chunk-start dependent.
- On actual converted task-0 data, static per-timestep DHB rows decoded from a
  sampled current state produce target reconstruction errors around `10-73 mm`.
  Recomputing chunk-local DHB targets from the same sampled raw delta-action
  window gives `0.0 mm` reconstruction error on the same bases.
- The dataloader now has an opt-in fix:
  `recompute_dhb_invariants_from_actions: true`. It integrates the sampled raw
  EE delta-action chunk from the current state and encodes that local waypoint
  sequence before training. Use this for the next fresh DHB runs; old
  checkpoints trained on static per-timestep invariant rows should be treated as
  contaminated/diagnostic.
- A second train/eval mismatch was identified: the legacy StarVLA LIBERO state
  window ends at `t-1` (`state_indices = range(-16, 0)`) while evaluation feeds
  state ending at `t`. Active retraining configs should use
  `state_history_window: 4`, which routes through the corrected current-ending
  history builder.

## Strong Evidence Already In Hand

### Representation

- DHB invariants provide a cleaner geometric reuse story than direct action
  regression.
- Magnitude-only invariant tokenization supports high compression and SE(3)
  stability in the supplementary analysis.
- The standalone OAT tokenizer and teacher-forced GT-token codec now validate
  discrete invariant compression on held-out chunks:
  `0.004207 m` GT-token position MAE and `0.016274 m` endpoint drift.
- The direct StarVLA token policy is no longer a total fixed-sequence failure,
  but token-native execution is still not mature enough for a closed-loop
  claim.

### State-Conditioned Action Decoding

- The old SmolVLA path that could drop robot state has been fixed.
- Regression tests lock down state forwarding.
- The prior persistent end-effector rotation runaway is gone.
- `embodik` IK is stable and not the dominant bottleneck.

### Executable Closed Loop

SmolVLA validation established a real executable baseline:

| Checkpoint | Best loss | `0 cm` | `3 cm` |
| --- | ---: | ---: | ---: |
| smoke `10ep` | `0.94095` | `2/3` | `1/3` |
| full `50ep` | `0.69913` | `1/3` | `1/3` |
| longer `200ep` | `0.06337` | `1/3` | `1/3` |

The key result is that lower supervised loss did not improve closed-loop
success. The remaining error is structured placement failure, mainly
under-reaching in `Y` on `demo_1` and `demo_10`, not gross IK or rotation
collapse.

### Matched Regular-Head Control

The first matched `QwenOFT` control is now real:

- same Qwen backbone
- same dataset family
- same short `200`-step budget
- same `embodik` downstream evaluator

Current matched result:

| Model | Path | `0 cm` | `3 cm` |
| --- | --- | ---: | ---: |
| `QwenDHB` | `predict_trajectory() -> embodik` | `1/3` | `1/3` |
| `QwenOFT` | delta-action integration -> `embodik` | `0/3` | `0/3` |

This supports a prototype-scale DHB advantage over the regular head, but not
yet a broad benchmark claim.

This is the strongest current execution story: continuous DHB has prototype
closed-loop evidence over a matched QwenOFT baseline under the same short-budget
evaluation harness.

### Discrete Token Progress And Remaining Gap

The old `short2`/`short3` collapsed-token path is obsolete. The current valid
path uses `oat_tokenizer_gate1` as a frozen tokenizer source and derives GT
labels through the same normalized path used during training.

Current evidence:

- standalone tokenizer gate: `R2 = 0.294061`, `550` unique token IDs,
  effective token count `306.131`, collapsed sample fraction `0.0`
- teacher-forced GT-token codec: `0.004207 m` position MAE and `0.016274 m`
  endpoint drift
- token-branch overfit probe: `1.0` final token accuracy and `1.0` final
  sequence accuracy on four fixed examples
- best direct token-policy executable checkpoint so far:
  `tokenwarm1/steps_1000`, `0.046055 m` full position MAE and `0.180139 m`
  full endpoint drift

The remaining gap is that held-out token prediction and executable decode
quality are not aligned enough. Longer token training improves diversity
(`tokenwarm2/steps_2000` reaches `11 / 24` unique predicted sequences), but the
best endpoint drift still comes from `tokenwarm1/steps_1000`. Do not claim
token-native closed-loop success until a token checkpoint passes the offline
gate and a small embodied smoke.

## Main Remaining Claim Gaps

| Gap | Current status | Needed evidence |
| --- | --- | --- |
| Benchmark-scale superiority | only short-budget matched control exists | broader `QwenDHB` vs `QwenOFT` evaluator |
| Token-native policy | codec and overfit gates pass; held-out token policy is partial | reduced/factorized tokens or token-only warmup, then embodied smoke |
| Robust offset reach | calibrated direct-action path reaches `5/5` across `0,+3,-3 cm` offsets | video artifact plus matched baseline/control framing |
| Full pick-place success | reach happens too late for grasp/transport | earlier phase progression and learned full-action/gripper timing |
| Learned gripper behavior | demo replay expires before close-ready; model close can occur but too late | learned phase-aligned gripper/action objective without object-specific rollout gates |
| Data-scaling claim | not yet tested | seed-size learning curves |
| Augmentation claim | not yet tested end to end | seed-size plus augmentation ablations |

## Run Hygiene

Exclude from claims:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short1`

This run is contaminated because malformed token target supervision was used.
It should not be counted as a discrete-policy result.

These older corrected runs are useful historical diagnostics, not policy
evidence:

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short2`

`_workspace/starvla_runs/dhb_starvla_qwen_discrete_supervised_short3`

## Next Validation Milestones

### Milestone 1: Freeze the Canonical Evaluator

Define one shared StarVLA evaluation contract for:

- history buffering
- reference handling
- chunk continuation
- gripper debouncing
- `predict_trajectory() -> embodik` execution

No major claim should depend on per-run rollout exceptions.

### Milestone 2: Turn Discrete Tokens Into A Policy Result

The anti-collapse tokenizer gate now passes, so the next milestone is no longer
tokenizer debugging. The next milestone is a token-native StarVLA checkpoint
whose held-out token predictions decode into trajectories competitive with the
continuous DHB checkpoint.

Proceed only if:

- predicted token diversity beats the majority-token baseline and avoids
  fixed-sequence collapse
- full decoded endpoint drift improves over `tokenwarm1/steps_1000`
- GT-token codec quality remains near the current `4.2 mm` position-MAE result
- the selected checkpoint passes a small embodied discrete smoke

The most credible next interventions are reduced-cardinality/factorized OAT
targets or token-only warmup with the continuous trajectory branch frozen.

First reduced-cardinality check: a 256-way OAT codebook preserves positive
held-out reconstruction (`R2 = 0.244400`, `218` unique IDs) but collapses the
final register to a static token. This supports the direction but argues for a
trimmed/factorized token target rather than a naive smaller global codebook.

Policy follow-up: the first 256-way policy checkpoint improves token-learning
metrics but not endpoint fidelity (`0.357411 m` full endpoint drift). A
token-only continuation with the continuous branch frozen improves token
accuracy slightly (`0.114583` vs `0.104167`) but leaves endpoint drift unchanged
(`0.358955 m`). This is useful negative evidence: the paper story can claim the
codec and partial token-policy recovery, but a closed-loop discrete-token
execution claim still needs a target-structure change, not just longer CE
training.

### Milestone 3: Repeat the Matched Head Comparison

Carry both `QwenDHB` and `QwenOFT` through the same frozen evaluator and report:

- success at `0 cm` and `3 cm`
- IK feasibility
- final object-relative placement offsets
- videos
- predict-time and memory footprint

### Milestone 4: Convert Prototype Advantage Into Scaling Evidence

After the evaluator is stable, run small-seed studies:

- `N=1`
- `N=3`
- `N=5`
- `N=10`

The target claim is not just higher final success. The target claim is better
success per unit of real data, with confidence across seeds.

### Milestone 5: Test the Full Scaling Story

Pair DHB-adapted trajectories with simulator or world-model observation
diversity. Compare against non-DHB augmentation under the same seed sizes,
augmentation budgets, seeds, and same evaluator.

This is the experiment that would support the strongest final thesis:

> DHB-XR makes small real robot datasets reusable at larger scale.

## 2026-04-26 Story Status

Story completeness is about `70%` for a strong CoRL narrative.

The clean story is now:

- DHB-XR provides a structured geometric action interface for low-data VLA
  execution.
- The corrected closed-loop evaluator shows robust non-zero-offset reaching
  under benchmark initialization, raw LIBERO observations, long enough horizons,
  and dynamic object-relative metrics.
- The strongest current embodied evidence is the calibrated direct-action
  DHB-trained StarVLA path: `5/5` reach successes across the current
  episode/offset gate with `0` IK failures.
- Naive full-action/contact fine-tuning breaks the reach backbone, while the
  zero-initialized residual and gripper-calibration branches preserve reach.
  This strengthens the engineering story because it shows why the staged
  reach-preserving design is necessary.
- Discrete invariant tokens are validated as a codec and partial policy path,
  but not yet as the primary closed-loop success path.

The open claim gaps are:

- Full pick-place is not yet solved without object-specific rollout gates.
- The learned close probability is under-calibrated under delayed offset reach;
  it peaks around the correct late phase but remains below a natural `0.5`
  decision threshold.
- The policy still lacks reliable post-contact transport/release after close.
- The big scaling claim still needs seed-size and augmentation ablations.

2026-04-26 contact-ready update:

- The constrained cumsum contact branch is now the best contact diagnostic:
  reach is preserved at `5/5`, contact reaches `18.9 mm` dynamic bowl XY, and
  the bowl moves `98.5 mm`, but learned release/placement still fails.
- The closed-state transport continuation is neutral/negative: reach remains
  `5/5`, but contact worsens to `23.1 mm` XY / `44.2 mm` 3D and there is still
  no reopen.
- Do not spend more unattended time on more gripper thresholds or the same
  closed-residual objective. The next useful evidence is a changed
  post-contact training distribution, a learned release gate, or a stronger
  matched-baseline/scaling experiment.

Streaming gripper decoder update:

- The chunk runner now carries learned gripper debounce state across replans,
  which is the correct closed-loop semantics for VLA execution.
- Replaying the saved full-task traces with the fixed decoder does not reveal a
  hidden release. The model repeatedly predicts closed in the first executed
  slots of each chunk and open in the final executed slot, then returns to
  closed on the next replan.
- Therefore the release gap is not an evaluator gate hiding a valid open
  command. It is a learned chunk-slot artifact and post-contact/release data
  coverage problem.

Latest full-task diagnostic:

- config:
  `configs/libero_dhbxr_qwen_fresh_actionres_closed_gripcal.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_closed_gripcal_fromhard300`
- checkpoint: `steps_600_pytorch_model.pt`
- reach gate:
  `5/5`, average minimum dynamic bowl XY `41.5 mm`, `0` IK failures
- full-task no-gate threshold `0.37`:
  close at step `109`, close relative to bowl about
  `[35.7, -31.8, 11.6] mm`, no reopen, no successful placement

Current paper-safe boundary:

- Safe to claim: robust offset reach under the corrected evaluator, clean
  implementation fixes, and evidence that reach-preserving residual structures
  are the right path for learned contact.
- Safe to claim: DHB invariant tokens can be encoded/decoded with strong
  teacher-forced codec fidelity and can be integrated into streaming
  prediction code paths.
- Not safe yet: full pick-place success, broad benchmark superiority, or
  closed-loop discrete-token superiority.

Next most impactful work:

1. Train delayed-offset gripper calibration and post-contact transport/release
   without modifying the validated base reach field.
2. Convert the current reach result into paper-grade videos and compact tables
   with matched QwenOFT controls.
3. Run a small multi-seed or small-data ablation only after the full-task or
   matched-head evaluator is frozen.
4. Keep discrete-token work focused on factorized/trimmed targets until
   endpoint drift becomes competitive with the continuous branch.

2026-04-26 BCE gripper-head update:

- The BCE gripper-state path is now implemented and tested. This matters
  because it lets the model learn a single closed-probability logit from the
  same categorical `dhb.gripper_state` labels used by the current data export.
- The BCE branch preserves the corrected reach result: `5/5` reach successes,
  average minimum dynamic bowl XY `41.4 mm`, and `0` IK failures.
- It improves probability scale only modestly. The full-task no-gate trace has
  max close probability about `0.421`, still below a natural `0.5` threshold.
- With a calibrated generic threshold of `0.35`, the model closes at step
  `109` near the bowl, but still fails transport and release.

Story implication:

- This is useful negative evidence. It shows the remaining gap is not caused
  by a missing BCE loss path or gripper-command polarity. The bottleneck is
  learned post-contact behavior after the close event.
- The stronger story should now emphasize the validated reach/adaptation
  backbone plus a concrete, isolated post-contact training gap. Full pick-place
  should not be claimed until a physically closed-state residual can move the
  object toward the plate and reopen without object-specific gates.

2026-04-26 overnight precision-contact update:

- A post-contact residual branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_postcontact.yaml`,
  `steps_800`) preserves the corrected `5/5` reach gate and modestly improves
  full-task transport diagnostics. IK failures drop from `30` to `16`, bowl
  displacement increases to about `80 mm`, but the rollout still never
  reopens or succeeds.
- A release/gripper-state branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_releasegrip.yaml`, `steps_600`)
  also preserves the `5/5` reach gate, but it over-calibrates close
  probability. At low threshold it closes much too early; at higher threshold
  it closes near contact but still does not reopen.
- The most important result is that grasp requires a stricter metric than the
  current reach-only success threshold. The default calibrated scale
  `1.0,1.35,1.15` reaches reliably, but often only to `~40-50 mm` bowl XY,
  while physical grasp likely needs `<=15-20 mm`.
- A global, object-agnostic direct-action Y-scale calibration from `1.35` to
  `1.55` improves the 5-case average dynamic bowl XY from `41.5 mm` to
  `34.8 mm`. In the full task, the same calibration brings the close pose to
  about `[1.5, -25.5, 44.7] mm` from the bowl, and the closest dynamic
  approach reaches `24.7 mm` XY / `43.4 mm` 3D.
- A delayed close threshold (`0.94`) shows the trajectory can reach
  `17.1 mm` XY / `30.2 mm` 3D, but it closes too late (`step 177`) and after
  IK failures. This cleanly separates the remaining problem into contact-ready
  precision plus learned phase timing.

Story implication:

- The robust offset-reach story is intact and now better bounded: it is a
  reach/adaptation result, not yet a grasp success result.
- The next breakthrough claim should be framed around turning robust reach into
  contact-ready grasp without object-specific gates. The target evidence is
  `<=20 mm` dynamic bowl XY at learned close time, object lift/transport toward
  the plate, and a learned reopen event.
- The DHB-XR story is still coherent for CoRL: DHB gives a structured,
  reach-preserving action interface; naive joint contact training breaks or
  miscalibrates behavior; staged residual/contact modules are the principled
  route to full manipulation.

2026-04-26 contact-precision diagnostic:

- A narrow pre-contact residual branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_contactprecision.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_fromrelease600`,
  `steps_400`) trained only the small additive action residual adapter from
  the release/gripper checkpoint.
- It failed the corrected 5-case reach-preservation gate: `2/5` reach
  successes, average dynamic bowl XY `70.8 mm`, average dynamic bowl XYZ
  `73.6 mm`, and `0` IK failures.
- This is a clean negative result. The failure is behavioral drift, not IK
  infeasibility. It shows that residual capacity alone is not enough; the
  objective must explicitly preserve the validated base reach field while
  improving contact readiness.

Story implication:

- The strongest story remains robust offset reaching plus a principled
  isolation of the remaining contact/release gap. Do not present the latest
  residual branch as progress toward full pick-place.
- The next breakthrough evidence should be a constrained contact-ready policy
  that reaches `<=20 mm` dynamic bowl XY before learned close while preserving
  `5/5` reach under the canonical benchmark-init/raw-image/long-horizon
  evaluator.
- Until that result exists, the paper narrative should emphasize that DHB-XR
  exposes and preserves geometric reach structure, while full grasping needs a
  separate contact-quality module rather than naive action-head fine-tuning.

2026-04-26 video packaging update:

- Added a reach video manifest:
  `_workspace/paper_eval/2026-04-26_reach_video_manifest.md`.
- The manifest points to three non-zero-offset successful reach videos under
  the canonical contract: episode `0,-3 cm`, episode `0,+3 cm`, and episode
  `1,+3 cm`.
- These videos should be used as visual evidence for robust offset reaching
  only. They are not full pick-place evidence because the gripper is held open
  and rollout stops on reach success.

2026-04-26 contact-gap packaging update:

- Added a consolidated contact-gap table:
  `_workspace/paper_eval/2026-04-26_contact_gap_summary.md`.
- The table separates the best current contact signal from the remaining
  failure. Y-scale `1.55` improves close-time lateral error to `25.5 mm`, and
  a delayed threshold can reach `17.1 mm` dynamic bowl XY, but both still fail
  learned release and incur IK failures.
- New claim gate: a full-task branch must preserve `5/5` reach, close before
  step `120` at `<=20 mm` dynamic bowl XY, move the object at least as much as
  the current best diagnostic, and learn a plate-side reopen without
  object-specific gates.

2026-04-26 constrained residual update:

- Added and validated a cumulative base-action preservation loss for residual
  adapters. The intent is to prevent the exact failure seen in the previous
  contact-precision branch, where small per-step residuals accumulated into
  endpoint drift.
- The constrained branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_contactprecision_cumsum.yaml`,
  `steps_400`) restores reach preservation: `5/5` corrected reach successes,
  average dynamic bowl XY `39.5 mm`, and `0` IK failures.
- Its best full-task no-gate diagnostic reaches the strict contact-ready
  regime: minimum dynamic bowl XY `18.9 mm`, minimum dynamic bowl 3D
  `33.3 mm`, close at step `101` with `22.3 mm` XY / `33.6 mm` 3D close
  geometry, and bowl motion `98.5 mm`.
- It still does not solve full pick-place: no reopen, `29` IK failures, and
  final bowl-plate XY remains about `140 mm`.

Story implication:

- This is the strongest partial progress since the reach result. The narrative
  can now say the remaining gap was narrowed from robust reach to
  contact-ready grasp geometry, while the unresolved blocker is post-contact
  transport and release.
- Do not continue scalar close-threshold sweeps. Threshold `0.92` delayed
  close, worsened close XY to `35.8 mm`, and increased IK failures to `39`.

2026-04-26 command-state transport update:

- The latest implementation gap was the gripper state used for closed-loop
  conditioning. Live rollout commanded the gripper closed, but the observed
  physical qpos stayed open for many policy steps. The closed residual was
  therefore disabled during the intended grasp/transport phase.
- Rollout and dataset loading now support a commanded gripper-state proxy.
  This is a general streaming VLA convention, not an environment-specific
  object-distance heuristic: the policy conditions on its own command history
  when actuator observations lag.
- A command-state transport branch preserves the corrected reach gate:
  `5/5`, average dynamic bowl XY `40.3 mm`, and `0` IK failures.
- A command-state XY closed residual branch gives the cleanest integrated
  behavior so far: no late IK failures in the full-task diagnostic, but still
  no pick-place success.
- The remaining miss is now sharply localized. The arm reaches about
  `19.2 mm` dynamic bowl XY before close, but the learned close fires later at
  about `29.2 mm` XY. The likely physical grasp tolerance is closer to
  `<=15 mm`, so the unresolved gap is contact-centered approach and learned
  close timing, not broad arm smoothness or gripper polarity.

Story implication:

- The story is now around 80-85% complete for a strong paper narrative. The
  validated pieces are robust offset reach, correct streaming DHB/token
  mechanics, reach-preserving residual design, and a clear explanation of why
  naive gripper/release training fails under delayed offset rollout.
- The missing 15-20% is one clean integrated success result or a stronger
  matched-control table that makes the partial result scientifically decisive.
  Full pick-place should still not be claimed.
- The next breakthrough path is contact-centering under the command-state
  streaming convention, followed by release/placement after close geometry is
  physically graspable.

2026-04-26 contact-centered gripper update:

- The contact-centered residual branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_contactcenter.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_fromxy500`,
  `steps_500`) preserves the corrected reach gate at `5/5`, average dynamic
  bowl XY `37.3 mm`, average dynamic bowl XYZ `78.7 mm`, and `0` IK failures.
- Under the stronger diagnostic scale `0.85,1.55,1.15`, the old gripper head
  exposes very strong contact geometry but closes too late: minimum dynamic
  bowl XY reaches `7.2 mm`, while the close event waits until step `149`.
- The contact-center gripper continuation
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_contactcenter_grip.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_grip_fromcenter500`,
  `steps_800`) moves close timing into the plausible grasp window without
  object-specific gates: close step `103`, close-time dynamic bowl XY
  `18.0 mm`, close-time vertical offset `22.4 mm`, minimum dynamic bowl XY
  `8.2 mm`, and `0` IK failures.
- It still does not solve full pick-place. The bowl is contacted but
  pushed/slid instead of reliably lifted and transported; final bowl-plate XY
  remains about `116.8 mm`.

Story implication:

- The story is now around 85-90% complete for the current prototype. We can
  support robust offset reach, correct streaming state/action conventions,
  reach-preserving residual design, and learned contact-close timing under a
  generic commanded-state VLA loop.
- The remaining unclosed claim is full manipulation after contact. For a
  strong CoRL submission, the next result should show object-level transport
  and release, or a matched-control table that makes the staged DHB advantage
  decisive even if full pick-place remains marked as ongoing.
- The next technical move should be a post-contact transport/release module
  from the contact-close checkpoint, with progress judged on bowl lift,
  movement toward the plate, no IK regression, and learned reopen stability.

2026-04-26 post-contact affordance update:

- The post-contact branch from the contact-close checkpoint
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_postcontact_fromcontactgrip.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_postcontact_fromcontactgrip800`)
  preserves the corrected reach gate at `5/5`, average dynamic bowl XY about
  `37.4 mm`, and `0` IK failures.
- It adds a real learned reopen event. The best trace closes at step `111` and
  reopens at step `201`, which is a useful progression beyond the prior
  no-reopen branch.
- It still fails object-level manipulation. The bowl-plate XY distance only
  improves to about `105 mm`, and the best trace has `76` late IK failures.
- The root cause is now clearer: our previous center-contact gate was the
  wrong physical grasp proxy. A reusable demo event-geometry probe shows that
  task `0` demos physically close at about `[-1.5, -49.9, +26.3] mm`
  EE-minus-bowl, with `52.1 mm` lateral offset. The current learned close is
  near bowl center, about `[4.5, -5.3, 19.6] mm`.

Story implication:

- This is a stronger and more honest narrative. DHB-XR did not simply fail
  because of generic instability; it exposed the difference between object
  center reaching and task affordance grasping.
- The next claim gap is affordance-relative action generalization: learn the
  side-grasp invariant/relative event geometry from minimal demos, then adapt
  it across offsets without deployment-time object gates.
- For the paper story, this supports the broader thesis better than another
  threshold sweep. DHB-XR should encode the reusable geometric relation needed
  for grasping, not just minimize distance to the object's center.

2026-04-27 smooth gripper-state update:

- The release-focused dense-gripper branch showed that no-gate reopen is
  learnable, but early checkpoints flicker and late checkpoints suppress
  release. To isolate that failure, a target-aware gripper-state smoothness loss
  was added to the DHB action head and exposed through the StarVLA wrapper.
- The first smooth branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip_smooth.yaml`,
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip_smooth_fromgrip800`)
  trains cleanly and removes the immediate close/open collapse. Step `200` and
  step `400` both keep `0` IK failures and reach extremely close center contact
  (`0.2-0.6 mm` minimum bowl XY).
- It is not a full-task solution. The policy still does not reopen under the
  tested weighting, and close-vector error remains `46.8-49.0 mm` because the
  contact remains center-biased rather than matching the demo side-grasp
  affordance.

Story implication:

- The gripper story is now more precise: learned gripper state can outperform
  demo replay and can be stabilized without deployment-time object gates, but
  the missing result is a coupled phase policy, not a smoother scalar gripper
  probability.
- The next high-impact branch should combine side-affordance close supervision,
  held-object/transport persistence, and object-target release. The smooth loss
  should remain available as a regularizer, likely at lower weight or applied
  only within same-phase hold segments, but more threshold sweeps are unlikely
  to close the remaining claim gap.

2026-04-27 coupled post-contact update:

- A coupled post-contact branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_coupled_postcontact.yaml`)
  was trained from the affordance-gripper `steps_800` checkpoint. It keeps the
  open/reach backbone frozen and jointly trains the closed-state residual with
  generic contact/gripper heads.
- The gated diagnostic improves close geometry but not full task success:
  step `600`, `0` IK failures, close step `93`, and `36.0 mm` close-vector
  error. It still never reopens under the previous `open=0.005` dense-state
  hysteresis.
- The evaluation harness had a real no-gate bug. `live_checkpoint_sweep.py`
  did not expose or forward `--disable-gripper-gates`, so some artifacts named
  `nogates` were actually using evaluator object gates. The harness now exposes
  that flag and has a regression test.
- With true gripper gates disabled and generic `close=0.55/open=0.40`
  hysteresis, the same checkpoint emits both learned transitions but too early:
  close step `81`, reopen step `93`, `57.6 mm` close-vector error, and
  `129.2 mm` bowl-plate XY at reopen.

Story implication:

- The claim boundary is sharper. We can say learned release exists without
  rollout-time object gates, but the model has not learned a persistent
  close-hold-transport-release phase. The next gap is phase persistence tied to
  transport progress, not gripper polarity, missing labels, or an evaluator
  gate.
- For paper evidence, any future "no-gate" table must explicitly verify
  `"disable_gripper_gates": true` in the JSON artifacts.

2026-04-27 phase-persistence and pre-contact-open update:

- The phase-persistence branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_phasepersistent.yaml`)
  adds dense gripper-state margin supervision on top of target-aware
  smoothness. With true gripper gates disabled and generic
  `close=0.55/open=0.40` hysteresis, step `300` is the current best
  learned-gripper sequence: close step `80`, reopen step `197`, `0` IK
  failures, and `2.3 mm` minimum bowl XY.
- It still fails the affordance/full-task gates. Close-vector error is
  `60.2 mm`, close XY is `64.7 mm`, reopen target XY is `118.1 mm`, and final
  target XY is `116.8 mm`.
- The trace shows why: the arm reaches the bowl physically around steps
  `136-148`, but the dense closed-state probability is already high from step
  `80`. A stricter generic close threshold (`0.85`) only moves close to step
  `83` and still has `51.6 mm` close-vector error. Contact-phase decoding is
  worse, closing at step `2`.
- A trace-scored true no-gate rerun gives the best current calibration
  evidence: actual close step `79` has `63.2 mm` close-vector error, while the
  same rollout later reaches a `26.1 mm` best close-affordance opportunity at
  step `123` with `0` IK failures. This shows the motion path is close enough
  to support a stronger grasp result if the gripper/contact-ready signal fires
  from current state rather than learned chunk phase.
- The pre-contact-open sampling branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_precontactopen.yaml`)
  is a negative result. It improves close-vector error to `42.1 mm`, but
  immediately reopens (`87 -> 89`) and introduces `72` IK failures. Its heavy
  checkpoints were deleted; summaries and scores are retained.
- The sparse generic contact-event branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_contactevent.yaml`)
  is also negative. It trains the existing `contact_head` with 3-way sparse
  event labels and evaluates with `--model-gripper-decode-source contact`, but
  closes at step `3` with `339.3 mm` close-vector error and no reopen. This
  rules out a simple "use a different existing gripper head" fix.

Story implication:

- Learned gripper control is now paper-usable as a partial result: it clearly
  beats demo replay and can produce a no-gate close-hold-reopen sequence.
- The remaining full-task blocker is contact-readiness calibration. The model
  currently predicts "the task is in the grasp phase" before the robot is at
  the side-grasp affordance.
- The next compelling result should use learned/perception-conditioned
  contact-ready supervision, not deployment-time object gates or threshold
  sweeps. The story should frame this as the remaining bridge from DHB-XR
  geometric adaptation to full object-level manipulation.

2026-04-27 current-alignment update:

- Explicit target-alignment controls were added for the StarVLA DHB contact and
  gripper heads. The default remains future chunk supervision; `current`
  broadcasts the first timestep label through the chunk for current-observation
  phase classification. Unit tests cover future/default behavior, current
  broadcast, multi-channel targets, and invalid modes.
- The fully current-aligned gripper branch is a useful negative/diagnostic
  result. It delays close to near actual bowl contact (`close_step=127`,
  `6.9 mm` minimum bowl XY, close snapshot about `[14.9, -3.1, 17.0] mm`
  EE-minus-bowl), but it immediately reopens at step `129` and creates `81`
  IK failures. This shows that current labels suppress the premature phase
  prior, but destroy persistent hold/transport state when applied to the dense
  gripper target.
- The hybrid current-phase branch keeps dense gripper-state labels
  future-aligned and current-aligns only contact/contact-phase targets. It is
  stable but not solved: true no-gate rollout has `0` IK failures, close step
  `77`, reopen step `233`, `4.7 mm` minimum bowl XY, and `0.75` path
  efficiency. It still closes too early for the side-grasp affordance
  (`69.6 mm` actual close-vector error), while the same trajectory reaches a
  `25.9 mm` best close-affordance opportunity at step `122`.

Story implication:

- The root cause is now narrower than "gripper timing." The model needs two
  separable signals: a current-state contact-ready/open-before-contact signal
  for the next executed prefix, and a future/phase-persistent gripper-state
  signal for hold, transport, and release.
- This strengthens the method story because DHB-XR can expose a compact
  geometric phase structure, but the manuscript should not claim full
  manipulation until this current-ready versus persistent-hold split is
  validated in no-gate rollouts.

2026-04-27 current-prefix/currentready update:

- The `current_prefix` alignment branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentprefix.yaml`)
  is a negative result. Prefix-scoped current labels preserve IK stability and
  reach the bowl (`0.34 mm` minimum bowl XY), but dense gripper-state decoding
  still closes at step `80` with `60.2 mm` close-vector error and no target
  progress (`112.1 mm` final target XY). Contact-phase decoding closes at step
  `74` and never reopens.
- The `currentready` branch
  (`configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready.yaml`)
  is the first clean learned no-gate close-readiness result. It trains the
  generic contact head as a current-observation close-readiness classifier and
  keeps the dense gripper-state head future-aligned for persistence. The hybrid
  decoder is object-free: contact readiness closes, dense gripper state reopens.
- Currentready closes at step `103`, has `0` IK failures, reaches `10.2 mm`
  minimum bowl XY, and scores `37.1 mm` actual close-vector error with
  `19.0 mm` close XY. This beats the prior learned no-gate actual-close score
  and validates the two-signal design for close timing.
- It still never reopens and leaves the bowl `118.0 mm` from the plate. The
  remaining high-impact gap is now post-close transport/release, not
  open-before-contact calibration.

Story implication:

- The current story is credible for robust DHB-XR reaching/adaptation and for
  learned, object-free contact-readiness improving over demo replay and pure
  future gripper-state labels.
- The full manipulation claim should remain conditional until a no-gate
  release/transport branch moves the bowl toward the plate. The next best
  result would preserve the currentready close head and train release/transport
  without changing the deployment contract.

2026-04-27 seed-controlled evaluation update:

- The live rollout script now has an explicit `--rollout-seed` that seeds
  Python, NumPy, Torch, CUDA, and LIBERO, and summaries now record both
  `config_seed` and `rollout_seed`. This matters because several continuation
  configs had different training seeds, which previously also changed the
  evaluation environment seed.
- Under controlled `rollout_seed=70`, the prior currentready result should be
  treated as diagnostic rather than final benchmark evidence. Currentready with
  `close=0.55/open=0.40` closes at step `91`, never reopens, and keeps `0` IK
  failures, but actual close-vector error is `36.4 mm`. The same trajectory
  reaches a much better `3.3 mm` best close-affordance opportunity at step
  `110`, so the trajectory is not the limiting factor.
- A generic `close=0.58` calibration delays currentready close to step `101`
  and reaches `16.9 mm` close XY, but reopens at step `121` and still fails
  placement. The state-only release continuation changes only the dense
  `gripper_state_head`, but under the same seed it closes at step `87`, never
  reopens, and finishes `129 mm` from the plate.

Story implication:

- The safe claim boundary tightens: do not promote single uncontrolled
  gripper-rollout videos as final evidence.
- The positive story remains robust reach/adaptation plus a learned
  contact-ready design that creates near-perfect grasp opportunities; the
  missing result is seed-controlled phase selection for close-hold-transport
  and learned release.
- Future paper tables should report fixed rollout-seed matrices, not one-off
  traces, especially for gripper/release claims.

2026-04-27 transport-release continuation update:

- A currentready transport-release branch now freezes currentready
  close-readiness and trains only the closed-state action residual plus dense
  gripper-state release path. This is the right architecture split for the
  story because it tests whether post-close behavior can improve without
  moving the learned contact-ready trigger.
- Step `200` restores a learned late release while preserving close: close
  step `103`, reopen step `221`, `0` IK failures, `37.1 mm` close-vector error,
  and `114.4 mm` final target XY under controlled `rollout_seed=70`. This is a
  partial validation of the close-hold-release decomposition, but not full
  manipulation.
- Step `400` overtrains into close/open flicker (`85 -> 89`) and adds IK
  failures. The missing piece is not another longer run; it is a stronger
  held-object transport objective and phase-persistence constraint.
- A follow-up closed-transport action-only branch tested that hypothesis by
  starting from transport-release `steps_200`, freezing all gripper/contact
  heads, and training only the closed-state residual adapter. It is negative:
  step `150` has no release and `123.9 mm` final target XY; step `300` closes
  at `103`, reopens at `237`, keeps `0` IK failures, and preserves the
  `37.1 mm` close-vector score, but final target XY is still `112.9 mm`.
  This rules out pure post-close action MSE as the missing mechanism.

Story implication:

- The paper story can now say the representation/architecture naturally
  decomposes reach, current contact readiness, and post-contact residuals, but
  the full pick-place result remains open because learned object transport is
  still weak.
- Do not overclaim release. The best release result is still about `114 mm`
  from the plate, so it is evidence of learned phase transition, not placement.
- The next compelling result needs an offline object-transport or relabeled
  post-contact supervision signal. This can use simulator/demo geometry during
  data generation and scoring, but the deployment path should remain learned
  and object-free.

2026-04-27 object-geometry transport update:

- The converter and StarVLA wrapper now support generic offline object-geometry
  transport labels: `dhb.object_to_eef_pos`, `dhb.target_to_eef_pos`,
  `dhb.object_minus_target_pos`, and `dhb.transport_phase`. Live rollout also
  records `pred_object_target` diagnostics when the head exists.
- A simple object-minus-target auxiliary head conditioned into the closed
  residual is negative. It preserves the useful close at step `150`, but the
  predicted vector is nearly constant in live rollout; by step `300` it
  regresses into early close/open and IK failures.
- A richer object-pair target
  `[object_to_eef_pos, target_to_eef_pos]` read from frozen action features is
  also negative. It improves neither release nor placement.
- A separate cross-attention object-pair readout is mechanically better: live
  predictions vary across chunks, and the context projection is zero-initialized
  to preserve the warm start. The embodied result is still negative. Step
  `150` flickers (`87 -> 89`), while step `300` closes at `103`, has `0` IK
  failures, but never releases and ends `117.9 mm` from target.

Story implication:

- Object geometry supervision is a necessary infrastructure piece for the
  manuscript story, but auxiliary object-vector conditioning alone is not the
  breakthrough. The remaining gap is learned phase selection: hold through
  transport and open only when a learned release-ready signal says the object is
  at the target.
- The next result should explicitly train release readiness and hold
  persistence from the offline object-target geometry, while keeping the live
  policy object-free and learned.

2026-04-27 object-target release objective update:

- The converter now exports generic object-target release labels:
  `dhb.object_release_ready`, `dhb.object_release_contact`,
  `dhb.object_release_event`, `dhb.object_release_phase`,
  `dhb.object_release_gripper_state`, and `dhb.object_hold_transport`.
- These labels come from offline simulator/demo object-minus-target geometry
  with a sustained-ready requirement. They are not rollout gates.
- `QwenDHB` now supports configurable `contact_target_field`,
  `contact_phase_field`, and `gripper_state_field`, and StarVLA sampling can
  use `gripper_phase_sampling_key` for the object-release phase.
- The active branch is
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml`.
  It preserves the currentready close-readiness head and trains the phase plus
  dense gripper heads against object-target release/hold labels.

This is now the cleanest implementation path for the manuscript story: minimum
demo data plus simulation-derived spatial supervision, learned into the VLA
heads without object-specific deployment gates.

2026-04-27 object-release rollout update:

- The first object-release evaluation exposed a protocol trap: full-task
  offset sweeps using `--max-policy-steps 0` stop at demo length. For the
  episode `1`, `+3 cm` case that means `84` policy steps, before controlled
  baselines usually close. The rollout scripts now warn and record
  `demo_length_cap_warning`; full-task claims should use an explicit long
  horizon such as `--max-policy-steps 240`.
- Under the corrected fixed-seed protocol (`rollout_seed=70`, true gripper
  gates disabled, `max_policy_steps=240`), object-release `steps_300` closes at
  step `82`, reopens at step `153`, and reaches `1.8 mm` minimum dynamic bowl
  XY. This validates a learned object-free close/reopen sequence from the new
  offline release labels.
- It is still not a full pick-place result: actual close-vector error is
  `51.7 mm`, reopen target XY is `121.1 mm`, final target XY is `116.1 mm`,
  and late IK failures remain (`11`). Step `150` is similar but weaker:
  close `86`, reopen `201`, final target XY `122.6 mm`, `12` IK failures.
- Generic hysteresis calibration can improve one metric at the cost of
  another. `close=0.60/open=0.60/debounce=1` improves actual close to
  `37.9 mm` close-vector error and `12.8 mm` close XY, with learned reopen at
  step `139`, but final target XY remains `110.6 mm` and IK failures rise to
  `64`. Direct `contact_phase` decoding is worse (`75 -> 87` close/open,
  `73.7 mm` close-vector error, `75` IK failures).

Story implication:

- We can now claim the object-release supervision path is mechanically
  validated and produces no-gate learned release behavior, but not successful
  placement.
- The current paper boundary is clean: robust offset reach and learned
  close/reopen sequencing are supported; held-object transport and release to
  the plate are not yet supported.
- The next compelling result must reduce final bowl-plate XY from roughly
  `110-120 mm` toward the scorer gate while preserving learned close
  affordance and avoiding IK failures. More threshold sweeps or direct
  `contact_phase` decoding are not the main path.

2026-04-27 current-phase release decoder update:

- A new deployment decoder, `contact_release_phase`, now separates learned
  close and learned open more cleanly: currentready contact probability closes
  the gripper, while the current-aligned object-release phase class opens it.
  This keeps the live policy object-free and preserves the manuscript claim
  boundary.
- The current-phase release continuation validates the interface but not the
  full task. At `steps_300`, `close=0.55/open=0.55/debounce=2` reaches
  `6.4 mm` minimum bowl XY, but actual close is still too early/high
  (`49.1 mm` vector error), no release occurs, and final target XY is
  `111.8 mm`. A stricter learned close threshold (`0.60`) improves close to
  `35.8 mm` vector error and `16.8 mm` XY, but increases IK failures and
  worsens final target XY. At `0.62`, the model never closes.
- The release head is behaving conceptually correctly in the trace: release
  probability stays near zero while the object is far from the plate. That
  means the next missing result is not "make open fire"; it is precise
  side-grasp close and stable held-object transport so the learned release head
  has a valid target-arrival state to detect.

Story implication:

- The strongest story remains DHB-XR as a compact geometric action/state token
  that enables spatial adaptation, plus offline simulation-derived labels for
  phase semantics. The current evidence supports this as an architecture and
  evaluation story, but the full success video still requires solving the
  closed-object transport gap.
- For paper risk control, separate claims into "validated now" and "pending
  final result": reach/offset adaptation, no-gate learned close, and
  object-release label contract are validated; full pick-place success and
  bowl-to-plate release are pending.

2026-04-27 close-contact recalibration update:

- A direct close-contact recalibration branch trained only `contact_head` from
  current-aligned `dhb.close_contact`, after fixing `contact_pos_weight` in the
  StarVLA BCE contact loss. The training objective converged, but the rollout
  collapsed to immediate close at step `1` for both checkpoints.
- This is a useful negative result for the story: offline geometry labels are
  powerful only if the sampling objective includes hard negatives. Positive-
  only close-window fitting does not demonstrate invariance or perception
  grounding; it just creates an always-close shortcut.

Story implication:

- The next convincing result should frame close as a calibrated affordance
  decision, not merely a sparse event class. The model must learn that "near
  the bowl but too high / wrong side / not settled" is still open. That is the
  missing supervision shape for a clean DHB-XR manipulation story.

2026-04-27 balanced close-contact update:

- A balanced close-contact branch and a contact-cross-attention variant were
  tested after the immediate-close collapse. These are also negative, but they
  clarify the root cause. With balanced sampling and low positive weight, the
  model reaches the bowl but does not close at the default threshold. Lowering
  the generic threshold to `0.15` recovers a close at step `87`, but only
  matches the previous partial close score (`40.0 mm` close-vector error) and
  still has no release or target placement. Unfreezing `contact_cross_attn`
  suppresses close confidence further.
- This means the current limitation is not a missing `dhb.close_contact` label,
  a gripper polarity issue, or a final-layer capacity issue. The labels and
  heads exist, but iid demo BCE does not cover the closed-loop rollout states
  where the policy is near the bowl but still not at the side-grasp affordance.

Story implication:

- The paper story is stronger if we present this honestly as the remaining
  closed-loop distribution gap. DHB-XR already gives a stable spatial reaching
  backbone and a compact learned phase interface. The missing final result is
  the data/learning recipe that turns simulator-derived affordance labels into
  a calibrated closed-loop contact decision under offset-induced timing drift.
- The next credible breakthrough should use rollout-state relabeling or
  contrastive hard negatives, not another BCE sampler sweep. This aligns with
  the larger storyline: simulation/world-model augmentation is valuable because
  it can create exactly these spatially diverse corrective contact states that
  the minimum-demo dataset lacks.

2026-04-27 contact-mining and margin update:

- A rollout contact miner now converts recorded traces into explicit
  close-affordance positives and hard negatives. On the best current partial,
  the trace contains both true close-affordance states and many hard negatives
  where close probability remains high. This is the cleanest empirical support
  so far for the closed-loop distribution-gap diagnosis.
- A generic contact-margin loss was added and smoke-tested, then trained from
  the same checkpoint. The result is negative: the head suppresses close
  confidence and never closes at the default learned threshold. This brackets
  demo-only contact-head training from both sides: positive-heavy BCE
  always-closes, balanced/margin demo training never-closes.

Story implication:

- This actually strengthens the big-picture story rather than weakening it:
  minimum-demo imitation is missing the exact spatially diverse off-manifold
  contact states needed for robust closed-loop manipulation. The next result
  should be positioned as simulation/world-model data augmentation producing
  those hard negative and corrective states, with DHB-XR serving as the compact
  geometry-aware action/phase representation that can absorb the augmentation.
- For CoRL framing, avoid claiming solved full pick-place from the current
  checkpoint. Claim the validated pieces separately, then make the next
  breakthrough experiment a direct test of the central thesis: closed-loop
  rollout-state augmentation should improve calibrated contact decisions
  without adding object-distance gates at deployment.

2026-04-27 policy-trace hard-negative dataset update:

- The policy-trace bridge has now been implemented end-to-end. Saved rollout
  images/state windows and contact-miner manifests can be loaded as QwenDHB
  training samples and mixed with the normal LeRobot demo dataset. This closes
  an important infrastructure gap: closed-loop observations are no longer just
  diagnostics; they can directly supervise the learned DHB contact token.
- The first training run with this bridge is negative but clarifying. The two
  available policy-trace manifests contain `242` hard-negative supervised steps
  and no closed-loop positive steps. Training on these hard negatives plus demo
  positives suppresses close confidence: both `steps_150` and `steps_300`
  never close on fixed-seed true-no-gate `0 cm`/`3 cm` h240 sweeps.
- This result is exactly the expected failure if the augmentation set is
  one-sided. It proves the data path and the hard-negative concept, but it also
  shows that the missing data is now positive off-manifold contact states, not
  another gripper decoder or threshold.

Story implication:

- The strongest next experiment is now concrete and story-aligned: generate a
  balanced rollout-state contact dataset with both near-miss hard negatives
  and successful/near-success side-grasp positives. Sources can be successful
  policy traces, simulator replay around demo close-affordance poses, or a
  world-model/contact-state generator. This directly realizes the manuscript's
  claim that simulation/world-model augmentation plus DHB-XR invariants can
  turn minimum teleop data into spatially robust manipulation behavior.

2026-04-27 future-aligned contact-token update:

- A target-alignment bug explained the failed demo-positive contact retrain:
  `contact_target_alignment: current` converted late future close labels into
  all-open chunks. The active balanced contact config now uses
  `contact_target_alignment: future`, with a regression test for late close
  positives.
- Demo close events are now exported into the same policy-trace format as
  rollout chunks. This gives a unified representation for nominal positives,
  rollout hard negatives, and future simulator/world-model contact positives.
- The corrected contact-only run proves the training signal is real but
  under-calibrated. Demo-positive chunks peak around `0.54`, rollout chunks
  peak around `0.46-0.48`, and threshold `0.45` is needed to close without
  object-distance gates.
- A separate evaluation lesson is now locked in: the examples should use the
  validated direct-action calibration `x=0.85, y=1.55, z=1.15`. With that
  scale, reach is compelling (`6.9 mm` XY on the `3 cm` offset case), whereas
  neutral scaling hides the trajectory progress.

Story implication:

- We can now claim a cleaner mechanistic story: DHB-XR contact tokens are
  trainable and object-gate-free, but minimum-demo positives do not calibrate
  them on closed-loop observations. The missing ingredient is precisely the
  spatially diverse simulator/world-model augmentation proposed in the
  manuscript.
- The strongest near-term result is not "full task solved"; it is a rigorous
  staged result showing calibrated reach under offsets, future-aligned learned
  contact supervision, and a clearly diagnosed remaining gap: rollout-positive
  contact-state generation plus post-grasp transport.

2026-04-27 rollout-positive contact-token update:

- A calibrated no-gate future-aligned baseline produced two useful policy
  traces that reach the bowl without closing. The `3 cm` offset trace reaches
  `7.0 mm` minimum bowl XY, which is strong evidence that trajectory execution
  is no longer the only blocker.
- Offline contact mining at `45 mm` tolerance found `111` rollout-positive
  missed-close steps. This made it possible to train on actual deployment
  observations rather than only nominal demo positives and hard negatives.
- The policy-trace dataloader now supports `policy_trace_shuffle`; this fixed
  a practical training issue where short continuations could miss appended
  rollout-positive samples.
- The shuffled rollout-positive continuation activates the learned contact
  token under true no-gate deployment. At threshold `0.55`, the non-zero-offset
  case closes near the object (`~6.8 mm` XY, `~19.3 mm` Z). This is the
  strongest current evidence that the learned DHB contact token can become a
  deployment signal instead of an object-distance heuristic.
- It still does not solve full pick-place. Case `0` calibration is inconsistent
  and post-grasp transport/release remains weak. The `45 mm` offline positive
  tolerance is too broad for final grasp claims because it can include high
  near-bowl states.

Story implication:

- The paper-safe claim is now staged, not binary. We have a compelling
  progression from spatially calibrated DHB-XR reaching, to future-aligned
  learned contact-token training, to no-gate close activation on rollout
  observations. The remaining gap directly motivates the manuscript's core
  augmentation thesis: generate side-grasp-ready positives and hard negatives
  from simulator/world-model rollouts, then train the compact DHB-XR token
  interface on that spatially diverse distribution.
- Do not overclaim success yet. The current result supports "learned contact
  token is trainable and can fire near the object without deployment geometry
  gates"; it does not yet support "full pick-place solved under offsets."

2026-04-27 side-grasp calibration update:

- The next stricter contact-label pass added component-wise side-grasp
  positives and near-object hard negatives. This is the right label structure
  for the story because it distinguishes the demo side-grasp affordance from
  broad center/high near-bowl states without adding deployment geometry gates.
- The first strict manifest is too imbalanced for a short continuation: `9`
  positives versus `601` hard negatives. The resulting checkpoint suppresses
  close confidence and never closes in the fixed no-gate eval.
- A lighter side-grasp continuation restores close activation and keeps the
  learned-token/no-gate contract, but it does not solve grasp or placement.
  The best fixed non-zero-offset close remains near the bowl, while the
  zero-offset case closes too high.

Story implication:

- This strengthens, rather than weakens, the central manuscript thesis. The
  current bottleneck is no longer code plumbing, gripper polarity, or a missing
  token path. It is the scarcity of balanced off-manifold side-grasp-positive
  states. That is exactly where simulator replay, world-model augmentation, or
  DAgger-style correction should provide the missing data.
- The clean claim boundary is now: DHB-XR has validated offset reach and a
  learned object-gate-free contact token that can fire near the object, but
  full pick-place needs balanced side-grasp-ready positive augmentation plus
  post-grasp transport/release validation. Do not claim the full task solved
  until both fixed cases close within the side-grasp affordance window and
  release near the plate under true no-gate deployment.

2026-04-27 physical-close storytelling correction:

- The side-grasp scorer now uses observed physical gripper closure when
  available. This matters because the demo reference is physical close, while
  model rollout records both command close and observed gripper qpos.
- With this corrected metric, the bestpos v2 non-zero-offset run gives the
  first clean learned-token side-grasp result: episode `1,+3 cm` commands
  close at step `99`, physically closes at step `101`, and reaches `9.2 mm`
  close-vector error to the demo side-grasp affordance without object-distance
  gates.
- This is not yet a solved full-task result. The same branch fails the
  zero-offset side-grasp gate and does not reopen/place.

Story implication:

- The staged claim is stronger now. We can say the DHB-XR learned contact token
  can produce a physically aligned side-grasp event under spatial offset, not
  only a rough near-object close. The remaining paper gap is consistency and
  post-contact manipulation, not whether the learned token can ever represent a
  correct contact event.

2026-04-27 closed-transport residual update:

- A closed-state action-only continuation from bestpos v2 improves rollout
  smoothness and feasibility: the best `steps_300` checkpoint closes in both
  fixed cases with zero IK failures and average minimum bowl XY of `19.4 mm`.
- The key non-zero-offset contact claim remains intact. Episode `1,+3 cm`
  physically closes with `10.6 mm` side-grasp vector error and no deployment
  object-distance gate.
- The continuation still does not learn release or placement. There is no
  reopen event, and final target XY remains about `137 mm` from the plate in the
  `3 cm` case.

Story implication:

- The current evidence supports a staged CoRL story: DHB-XR gives a compact
  invariant token that can generalize reach and contact under offset, and the
  remaining failure is specifically data coverage for post-contact release and
  placement. This matches the larger augmentation thesis: spatially diverse
  simulator/world-model rollouts should fill the missing off-manifold
  post-contact supervision more directly than another execution heuristic.

2026-04-27 release-aware diagnostic:

- Training release heads from nominal offline object-release labels is
  sufficient to produce learned reopen events under true no-gate deployment.
  `steps_300` reopens in both fixed cases.
- The reopen is not successful: it happens about `127-134 mm` from the plate,
  and IK failures increase. The non-zero-offset trace still preserves a valid
  physical side-grasp close (`13.9 mm` vector error), so contact is not the
  reason release fails.

Story implication:

- This is useful negative evidence. It shows the architecture can express
  release through the DHB-XR token path, but nominal demo release labels do not
  close the rollout distribution gap after contact. The next strong story step
  is rollout/world-model augmentation for post-contact release: positives must
  come from states that are actually closed and near target, while hard
  negatives must be closed-but-far states.

2026-04-27 rollout release mining update:

- The new release miner provides the missing training interface for
  rollout-state release labels. It deliberately separates release positives
  from closed-far hard negatives and can feed the existing policy-trace loader
  as `dhb.object_release_gripper_state`.
- Fresh policy traces from the active closed-transport checkpoint contain no
  honest near-target release positives. The best target XY remains over
  `100 mm`, so the current policy has a transport-to-target gap before a
  release-data gap.

Story implication:

- This gives a clean failure boundary for the paper narrative. The architecture
  supports learned release, and the tooling can mine release supervision, but
  the dataset must first create off-manifold post-contact states that actually
  reach the target. This directly supports the proposed simulator/world-model
  augmentation storyline.

2026-04-27 release hard-negative calibration:

- A conservative continuation mixed strict rollout closed-far hard negatives
  into the gripper-state release head while keeping deployment learned-token
  and object-gate-free.
- The intervention has the expected direction but not the final behavior:
  `steps_200` delays reopening and preserves the `3 cm` physical side-grasp
  pass (`10.6 mm` vector error), but it still reopens over `100 mm` from the
  plate and fails release in every fixed case.

Story implication:

- This result tightens the claim boundary. We can now say the learned DHB-XR
  gripper/release path is controllable by rollout-state supervision, but the
  decisive missing evidence is transport-to-target coverage after physical
  close. The strongest paper story is therefore not "more tuning solved
  release"; it is "DHB-XR exposes the right compact action-token interface, and
  simulator/world-model augmentation must fill the off-manifold post-contact
  states needed for robust full manipulation."

2026-04-27 object-transport conditioning diagnostic:

- Training the existing object-target-conditioned closed residual from the
  active closed-transport checkpoint did not close the full-task gap.
- The run preserves the learned `3 cm` physical side-grasp pass (`9.4 mm`
  vector error), but release still happens over `130 mm` from the plate and IK
  failures increase.

Story implication:

- This is direct evidence that nominal demo-side transport labels are not enough
  once closed-loop execution has drifted into a different post-contact state.
  The story should emphasize augmentation of post-contact rollout states, not
  merely adding an object-target auxiliary loss. The breakthrough experiment is
  to create spatially diverse closed-state corrections that actually reach the
  target, then train DHB-XR on those compact invariant/action tokens.

2026-04-27 transport correction miner:

- The new transport miner turns rollout traces into generic post-close
  correction requests: object-minus-target vector, desired object-space delta,
  and target-window status.
- On the active closed-transport baseline, target XY improves by only about
  `10 mm` after physical close and never enters the `45 mm` target window.

Story implication:

- This is the cleanest quantitative statement of the remaining gap. DHB-XR has
  the compact token path for contact and post-contact action, but the current
  dataset does not contain enough off-manifold closed-state transport examples.
  The manuscript's strongest next story is now concrete: use simulator or
  world-model augmentation to create spatially diverse post-contact correction
  trajectories from these mined requests, then show DHB-XR learns them better
  than ordinary action heads because the invariant/action token compresses the
  spatial transformation.

2026-04-27 trainable transport correction result:

- The transport miner now exports trainable policy-trace rows with normalized
  corrective actions, object-minus-target context, and closed gripper-state
  targets. The StarVLA policy-trace loader can consume all three fields.
- Aggressive corrective labels were not usable: v1 learned some action loss but
  introduced gripper flicker/reopen and IK regressions.
- Conservative labels plus dense gripper-state supervision were mechanically
  better: v2 `steps_200` removed the v1 flicker and avoided IK collapse, but it
  still did not transport the object into the target window and weakened the
  `0 cm` close. v2 `steps_400` overfit further and reopened again.
- Both transport-correction branches remain negative diagnostics. Their
  checkpoint weights were removed to keep disk available.

Story implication:

- This result sharpens the story rather than invalidating it. The DHB-XR token
  pipeline now has the right training interface for post-contact corrections,
  but failed rollout traces do not contain enough successful transport signal
  to learn from. The next compelling result must add successful post-contact
  target-reaching states through simulator reset/replay, trajectory generation,
  or world-model augmentation. That directly supports the paper's big-picture
  claim: DHB-XR is useful because it provides a compact spatially adaptable
  token for augmented trajectories, not because execution-side heuristics can
  rescue missing post-contact data.

2026-04-27 sim-state reset/replay diagnostic:

- Live rollouts can now optionally record flattened LIBERO simulator states per
  step. This creates a concrete reset/replay bridge from closed-loop failures
  to augmented post-contact data.
- The new correction planner and correction rollout runner can restore mined
  post-contact states and execute closed-gripper target-direction corrections.
- The first replay attempts did not produce successful target-reaching
  positives. Early `+3 cm` physical-close reset states produced `0.0 mm` target
  progress; a later selected reset state produced only `2.6 mm` progress.
- Tightening the miner to require the executed command to remain closed avoids
  stale post-open candidates. New coupling diagnostics show the active policy
  has weak object transport despite visible object motion: object/EE path
  coupling is only about `0.60` in the sim-state recapture, and target XY never
  enters the `45 mm` release window.

Story implication:

- The failure boundary is now more precise: the unsolved gap is not just
  release timing and not just target-direction action labels. The policy must
  create or be trained from grasp-attached post-contact states where the object
  follows the end effector. This strengthens the augmentation narrative: a
  simulator/world-model module should synthesize verified grasp-attached
  transport trajectories, and DHB-XR should be evaluated on whether its compact
  invariant/action token learns those spatially adapted trajectories more
  efficiently than ordinary action heads.

2026-04-27 coupling-aware replay update:

- Coupling-aware mining can now rank post-contact reset states by local
  object/EE coupling while filtering out static numerical artifacts.
- This found better `+3 cm` reset states: fixed correction plans produce about
  `26-33 mm` target progress with zero IK failures, compared with `0 mm` for
  earliest physical-close resets.
- The result is still not a solved data source. Closed-loop recomputed
  correction replay stalls around `97-100 mm` target XY and accumulates IK
  failures over longer horizons.

Story implication:

- This is a useful partial bridge from failure diagnosis to the big story.
  DHB-XR now has infrastructure to mine and replay spatial correction starts,
  and coupling-aware selection measurably improves target transport. However,
  the decisive missing ingredient is still successful grasp-attached
  target-reaching augmentation. The next manuscript-strengthening experiment
  should show simulator/world-model generation of those successful coupled
  trajectories, then compare DHB-XR token learning against regular action-head
  learning on the same augmented data.

2026-04-28 attached-positive generation update:

- Disk cleanup freed enough room for the next training runs: available space is
  now about `712G` after deleting superseded model weights while preserving run
  metadata and selected active checkpoints.
- The post-contact reset/replay runner now has an offline
  `--attach-object-to-ee` augmentation mode and can export policy-trace NPZ
  chunks plus JSONL manifest rows.
- The first attached `+3 cm` replay result is successful as a data-generation
  source: five coupling-selected reset states all reach the `45 mm` target
  window with `0` IK failures and about `87-88 mm` target XY progress.
- The generated manifest
  `_workspace/paper_eval/2026-04-28_attached_transport_policy_manifest_ep1_offset3.jsonl`
  has `30` trainable rows and loads through the existing StarVLA
  policy-trace dataset.

Story implication:

- This is the first concrete realization of the larger augmentation story:
  start from minimal real/teleop-like closed-loop data, use simulator-assisted
  spatial correction to generate successful grasp-attached post-contact
  trajectories, and feed those trajectories back through the DHB-XR action-token
  training path. The next evidence gap is whether a narrow no-gate policy
  continuation trained on these positives improves closed-loop target transport
  over the regular action-head/control baseline.

2026-04-28 gripper-contract and strategy audit:

- Community-code check: OpenVLA and StarVLA both execute LIBERO gripper commands
  with `-1 = open`, `+1 = close`, while treating gripper as a special binary
  absolute dimension rather than an ordinary normalized Cartesian delta. VLA
  Foundry follows the same broader contract: explicit action/proprio fields,
  relative-pose conversion through a reference pose, and gripper hysteresis in
  the execution adapter. This validates our object-free deployment direction
  and makes a sign bug unlikely.
- The reproduced step-100 attached-transport trace confirms the gripper actuator
  works. At close step `99`, commanded gripper is `+1.0`; observed qpos falls
  from about `0.034` to `0.0069` by step `101`.
- The command-close event is near bowl center, but the physical close event is
  not. The first close command occurs at EE-minus-bowl about
  `[-8.8, -15.3, +29.2] mm`; by physical close the vector is about
  `[-9.0, -44.5, +26.6] mm`, only `4.6 mm` from the first-three-demo side-grasp
  mean of `[-13.6, -45.1, +26.7] mm`.
- This means the current `13 mm` bowl-center proximity metric is insufficient
  but not because the policy cannot reach a side-grasp affordance. The stronger
  failure is after physical close: no reopen, best bowl-plate XY about
  `125 mm`, final bowl-plate XY about `130 mm`.

Story implication:

- Stop framing the remaining gap as gripper polarity, center reach, or
  side-affordance discovery. The next scientific and engineering gate is
  target-directed held-object transport plus release after a physically
  plausible side-grasp close. Attached simulator positives still support the big
  augmentation story, but the current 3-D target continuation is too weak; the
  next evidence should use stronger object/target conditioning and report
  physical close-vector pass, post-close coupling, target progress, and release
  proximity together.

2026-04-28 lift-transport supervision update:

- A direct audit found that the first attached-positive rows taught nearly
  zero-Z lateral transport. That explains the observed dragging behavior and is
  a concrete implementation gap rather than a theoretical failure of the
  DHB-XR story.
- The offline post-contact replay exporter now supports a generic
  lift/carry/lower transport profile. It is simulator-assisted augmentation
  only; deployment still uses learned observations, state history, action
  tokens, and gripper heads without object-distance gates.
- New lift-transport manifests for `0 cm` and `+3 cm` offsets produce non-zero
  Z supervision, but a small warm-start continuation still fails the no-gate
  placement benchmark. The best stable checkpoint keeps `0` IK failures but
  ends about `123 mm` from the plate; the checkpoint with better final distance
  incurs many IK failures.

Story implication:

- This strengthens, rather than weakens, the paper story boundary. We can show
  that the infrastructure is now capable of generating spatially adapted
  grasp-attached trajectories with lift/lower structure, but the current
  evidence does not yet validate the full low-data transport claim. The next
  compelling experiment must scale this augmentation beyond two narrow
  manifests and compare DHB-XR against a matched regular action head on the
  same successful augmented trajectories.

2026-04-28 external infrastructure check:

- The newly released VLA Foundry report/code is most relevant as validation of
  infrastructure choices, not as a drop-in fix. It uses explicit robotics
  preprocessing, action/proprioception fields, WebDataset-style manifests,
  gripper fields separate from pose fields, and past/future masks for chunked
  diffusion-policy generation.
- TRI-LBM's useful lesson is action-type-aware normalization. That maps well to
  DHB-XR's claim that compact geometric/phase tokens can make augmented
  trajectories easier to learn; it does not remove the need for successful
  grasp-attached transport data.

Story implication:

- This supports a cleaner claim: DHB-XR should be presented as a compact
  geometric action-token interface that can absorb simulator/world-model
  augmentation, while the benchmark infrastructure follows modern VLA practice
  by keeping action fields, gripper fields, proprioception, and chunk masks
  explicit. The evidence gap remains empirical: demonstrate that DHB-XR learns
  the broadened successful lift/transport/release positives better than the
  matched regular action head.

2026-04-28 broad lift-transport continuation update:

- The next broader attached-positive run used `65` successful offline
  lift/carry/lower policy-trace rows across step-50 and step-100 source
  checkpoints, two episodes, and `0 cm`/`+3 cm` offsets.
- This improved the best stable transport metric but did not solve full
  placement. The best checkpoint, step `50`, reaches average final bowl-plate
  XY `96.2 mm` with `0` average IK failures across the controlled `0 cm` and
  `+3 cm` cases. The narrow lift run's best stable checkpoint was about
  `122.6 mm`, so the broader positives matter. However, later checkpoints
  overfit or create IK, and none of the checkpoints produce a clean full-task
  success.

Story implication:

- The story is now more concrete but still bounded. We can honestly say that
  simulator-assisted attached lift/transport augmentation improves the
  direction of learned object transport, while also showing that the current
  objective/data diversity is insufficient for a paper-safe pick-place claim.
  The next decisive story gap is a matched comparison on a larger, more diverse
  successful augmentation set: DHB-XR action tokens versus a regular action
  head, both trained on the same lift/transport/release positives.

2026-04-28 lift/carry/lower diagnosis:

- Targeted traces show the current best broad checkpoint has learned some
  post-close object coupling, but not a robust lift/carry/lower sequence. It
  moves the bowl after close, yet the learned closed Z action is near zero or
  slightly downward in live chunks.
- Capacity-only fixes are negative. A high closed-residual scale can induce
  lift, but it also induces large uncontrolled arm excursions and IK failures;
  a midpoint scale loses object motion. This rules out scalar gain tuning as
  the next credible route.

Story implication:

- This is a useful refinement for the DHB-XR narrative. The paper should not
  claim that a generic continuous residual head plus a few replay positives is
  enough. The stronger and more DHB-specific claim to validate next is that
  compact phase/action tokens can represent lift, carry, lower, and release as
  distinct geometric events with phase-specific normalization. That aligns
  better with the invariant-token story than treating post-contact manipulation
  as one undifferentiated Cartesian residual.
