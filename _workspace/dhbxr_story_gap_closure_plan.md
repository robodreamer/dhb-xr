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
