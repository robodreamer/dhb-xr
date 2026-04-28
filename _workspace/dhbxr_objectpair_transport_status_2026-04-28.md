# DHB-XR Object-Pair Transport Status, 2026-04-28

This note records the current stop point for the object-pair attached transport
attempt. The gripper close issue is not the active blocker: the policy can still
command a physically valid close near the bowl. The remaining blocker is learned
post-close transport to the plate.

## What Changed

- `PolicyTraceContactDataset` now supports multiple object-target fields. This
  lets policy-trace manifests emit `target_object_to_eef` and
  `target_target_to_eef` as separate 3-D labels, while QwenDHB concatenates them
  into the configured 6-D object target.
- `scripts/libero_postcontact_correction_rollout.py` now exports
  `target_object_to_eef`, `target_target_to_eef`, and the unused convenience
  `target_object_pair_to_eef` during offline attached-object replay.
- Added
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_policytrace.yaml`
  for a bounded object-pair continuation from the guarded attached-transport
  step-100 checkpoint.
- Live rollout chunk diagnostics now split 6-D object-target predictions into
  object-to-EEF, target-to-EEF, and derived object-minus-target summaries.
- The transport miner now tolerates summaries with `step_trace: null` instead
  of crashing.

## Verified Runs

The original attached replay manifest was regenerated with object-pair fields:

`_workspace/paper_eval/2026-04-28_attached_transport_objectpair_policy_manifest_ep1_offset3.jsonl`

It contains 20 NPZ-backed chunks. The pair labels satisfy the LeRobot convention
`object_minus_target = object_to_eef - target_to_eef` to numerical precision.

The object-pair config passed a train-step smoke:

`_workspace/paper_eval/2026-04-28_attached_transport_objectpair_train_step_smoke.json`

The 200-step continuation completed:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_policytrace_fromguarded100`

The best non-zero-offset sweep was:

`_workspace/paper_eval/2026-04-28_attached_transport_objectpair_ep1_offset3_seed70_sweep.json`

| Step | Close | Reopen | Min EE-bowl XY mm | Final bowl-plate XY mm | IK fails |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 99 | none | 13.3 | 121.0 | 0 |
| 100 | 87 | none | 43.9 | 98.3 | 0 |
| 150 | 99 | none | 13.3 | 218.5 | 0 |
| 200 | 83 | none | 55.1 | 129.6 | 15 |

Compared with the guarded pre-object-pair reference, step 50 preserves the good
side-grasp approach and moves the object slightly toward the target, but it does
not cross the transport success gate. Step 100 improves final target distance
more, but the close affordance is weaker, so it is not a clean task solution.

## DAgger-Style Repair Attempt

A step-50 rollout was recaptured with simulator-state traces:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_policytrace_fromguarded100/live_checkpoint_sweep/attached_transport_objectpair_step50_ep1_offset3_seed70_simtrace/step_50/steps_50_pytorch_model_ep1_offset3.0cm_summary.json`

Transport mining found only limited natural post-close target progress:

`_workspace/paper_eval/2026-04-28_attached_transport_objectpair_step50_ep1_offset3_seed70_simtrace_transport_mining.json`

The attached replay from these current failure states did not produce clean
target-reaching positives:

| Replay | Rows | Target reached | Best observation |
| --- | ---: | ---: | --- |
| default closed-loop, gain 2 | 3 | 0 | row 0 reached 49.6 mm XY, just outside the 45 mm gate |
| stronger, 96 steps, gain 3 | 3 | 0 | more IK failures and worse final geometry |
| gentle row 0 | 1 | 0 | unstable final transport |

Conclusion: do not keep fine-tuning on these DAgger rows as positives yet. They
are useful diagnostics, but they would pollute the attached-transport objective
unless filtered or improved.

## Current Interpretation

The model can fit the 6-D object-pair auxiliary target on the small manifest,
but fitting that auxiliary loss is not enough to make the closed residual
execute reliable target transport. The predicted object-pair target appears
partly constant and not sufficiently state-calibrated under the current
20-chunk policy-only continuation.

The next high-impact step is not more warm-start steps on this same small
manifest. The next step should be broader and cleaner data generation:

1. Generate target-reaching attached transport positives from more diverse
   resettable post-close states, and export only rows whose replay actually
   reaches the target gate.
2. Prefer several episodes/offsets/checkpoints over one +3 cm failure trace, so
   the object-pair readout cannot solve the auxiliary loss with a near-constant
   target.
3. Revisit fresh/full dataset training with inline DHB invariants plus
   object-pair targets, rather than another narrow warm-start from the current
   prior.
4. Train release only after transport has a clean target-reaching gate; current
   no-reopen behavior is expected because this continuation deliberately kept
   release/contact heads frozen.

## Lift-Transport Label Audit

The next failure mode was a supervision bug in the attached-positive export, not
a gripper sign problem. The original attached transport rows had effectively
zero raw Z action in the closed phase, so the model was trained to drag the
attached object laterally rather than lift, carry, and lower it.

`scripts/libero_postcontact_correction_rollout.py` now has a generic offline
`lift_transport_place` profile for data generation. It raises an attached object
to a clearance height, carries laterally, and lowers near the target. This stays
outside deployment: no object-distance gates or environment-specific gripper
logic are added to the policy path.

Two new replay manifests were generated:

- `_workspace/paper_eval/2026-04-28_steps100_ep0_attached_replay_lifttransport_gated_manifest.jsonl`
- `_workspace/paper_eval/2026-04-28_steps100_ep1_offset3_attached_replay_lifttransport_gated_manifest.jsonl`

The manifest audit confirms non-zero lift/lower supervision. Raw Z action spans
about `-3.7 mm` to `+8.0 mm` per action, while the original no-lift rows were
near zero in raw Z.

The narrow continuation run is:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_lifttransport_fromdagger100`

Controlled no-gate sweep, `rollout_seed=70`, cases `0:0` and `1:+3 cm`:

| Step | Avg min EE-bowl XY mm | Avg min plate XY mm | Avg final bowl-plate XY mm | Avg IK fails | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 27.8 | 67.7 | 122.6 | 0.0 | stable but no placement gain |
| 100 | 19.3 | 96.7 | 113.9 | 28.5 | slight final gain, unacceptable IK |
| 150 | 29.8 | 104.0 | 131.5 | 44.0 | regressed |

Conclusion: the lift/lower label fix is necessary and should be kept, but this
small warm-start continuation is not a breakthrough. The branch improves the
training signal quality and confirms the data-generation path, yet it does not
produce a clean embodied placement result. Do not spend more unattended time on
closed-gain or more epochs for this exact run.

Updated next step:

1. Move from narrow policy-trace warm-starts to a broader regenerated training
   set that includes successful grasp-attached lift/transport/lower positives
   across multiple episodes and offsets.
2. Keep the current lift-transport generator as offline augmentation
   infrastructure, and require each exported row to pass target-window and IK
   gates before it becomes training data.
3. Compare the same augmented positives against the matched regular action head
   before claiming DHB-XR transport superiority.

## Broader Lift-Transport Continuation

The narrow two-manifest result above was expanded with additional successful
offline lift-transport replays from both the step-50 and step-100 source
checkpoints. The combined manifest is:

`_workspace/paper_eval/2026-04-28_attached_replay_lifttransport_gated_step50_100_manifest.jsonl`

It contains `65` rows:

- step-50, episode `0`, offset `0 cm`: `15` target-reaching rows.
- step-50, episode `1`, offset `+3 cm`: `18` target-reaching rows.
- step-100, episode `0`, offset `0 cm`: `24` target-reaching rows.
- step-100, episode `1`, offset `+3 cm`: `8` target-reaching rows.

The broader continuation run is:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_lifttransport_broad_fromdagger100`

Controlled no-gate sweep, `rollout_seed=70`, cases `0:0` and `1:+3 cm`:

| Step | Avg min EE-bowl XY mm | Avg min plate XY mm | Avg final bowl-plate XY mm | Avg IK fails | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 22.7 | 107.1 | 96.2 | 0.0 | best current transport trend, still no placement |
| 100 | 13.1 | 80.0 | 114.4 | 2.0 | good grasp proximity, worse final transport |
| 150 | 16.5 | 75.5 | 117.8 | 13.5 | increasing IK |
| 200 | 35.0 | 43.3 | 120.9 | 19.5 | reaches plate area in one case, unstable |

Conclusion: broadening the attached lift positives helped relative to the
narrow lift run. The best checkpoint improves average final bowl-plate XY from
about `122.6 mm` to `96.2 mm` while keeping `0` IK failures. It is still not a
paper-safe full-task result because it does not produce reliable placement or
release, and later checkpoints overfit into IK-heavy behavior. This is a data
diversity and objective-design gap, not an indication to keep training this
same run for more epochs.

Updated next step:

1. Preserve the broad step-50 checkpoint as the current best diagnostic
   transport trend, but do not claim success from it.
2. Run targeted traces on step 50 to inspect whether improvement comes from
   genuine lift/carry or residual sliding.
3. Generate the next augmentation set with more diverse source episodes,
   target placements, and phase/action-type normalization, then compare against
   a matched regular action head.

## Step-50 Trace and Capacity Diagnostics

Two no-video simulator-state traces were captured for the broad step-50
checkpoint:

- `_workspace/paper_eval/2026-04-28_broad_lifttransport_step50_ep0_trace.json`
- `_workspace/paper_eval/2026-04-28_broad_lifttransport_step50_ep1_offset3_trace.json`

The traces show real post-close object motion but inconsistent lift:

| Case | Close | IK fails | Final bowl-plate XY mm | Bowl Z lift after close mm | Post-close bowl XY motion mm | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ep0, `0 cm` | 91 | 0 | 100.4 | 17.5 | 112.8 | partial lift and coupled transport |
| ep1, `+3 cm` | 99 | 0 | 113.3 | 1.4 | 139.8 | mostly lateral motion; plate can be disturbed |

Chunk diagnostics expose the active implementation/objective gap. After the
gripper closes, the learned normalized Z action is about `+0.013`, which
denormalizes to roughly `-0.35 mm` per action step. The offline lift labels
contain initial normalized Z targets around `+0.56`, so the labels exist but
are not being learned strongly enough in the streaming policy.

The residual-capacity hypothesis was tested:

- Directly evaluating the same step-50 weights with
  `action_closed_residual_scale=0.9` creates strong lift, but overshoots badly:
  the `+3 cm` trace lifts the bowl to about `1.10 m`, ends `206 mm` from the
  plate, and accumulates `40` IK failures.
- A bounded retrain from the same base with `action_closed_residual_scale=0.9`
  is also negative. Step `50` averages `22` IK failures and step `100` averages
  `15` IK failures across the `0 cm`/`+3 cm` sweep.
- A midpoint inference diagnostic at `action_closed_residual_scale=0.6` keeps
  `0` IK failures on the `+3 cm` case but closes earlier/high, fails to move
  the bowl, and worsens min bowl distance.

Conclusion: the root cause is not a simple execution gain. The current closed
adapter/objective sits between two bad regimes: low scale learns stable lateral
sliding with too little lift, while high scale learns lift but produces large
uncontrolled excursions. The next fix should make lift, carry, lower, and
release explicit phase/action types with their own normalization or loss
weights, rather than amplifying the whole closed residual vector.

## Phase-Weighted Action-Loss Diagnostic

The next controlled intervention added a generic per-step action-loss weight to
policy-trace training. The trace manifest can now carry `action_loss_weight`,
and the StarVLA/DHB action head applies it only to supervised action residual
loss. This keeps deployment object-free: no object-distance gates or
environment-specific runtime logic are added.

The phase-weighted manifest is:

`_workspace/paper_eval/2026-04-28_attached_replay_lifttransport_gated_step50_100_phaseweighted_manifest.jsonl`

It contains the same `65` broad lift-transport rows as the previous manifest,
but labels per-step target actions as `lift`, `carry`, or `lower` from raw Z
motion with a `1.0 mm` threshold. The counts are `175` lift, `237` carry, and
`60` lower steps. The first diagnostic used weights `lift=4.0`, `carry=1.0`,
and `lower=2.0`.

Implementation commits:

- parent repo: `e907750 Add phase-weighted action trace loss`
- vendor StarVLA: `7b7aa1e Support per-step DHB action loss weights`

Tests:

`PYTHONPATH=vendor/starVLA:src:${PYTHONPATH:-} ./.pixi-home/bin/pixi run pytest tests/test_policy_trace_contact_dataset.py tests/test_libero_postcontact_correction_plan.py tests/test_action_head.py -q`

Result: `53 passed`.

Controlled no-gate sweep, `rollout_seed=70`, cases `0:0` and `1:+3 cm`:

| Run | Step | Avg min EE-bowl XY mm | Avg min plate XY mm | Avg final bowl-plate XY mm | Avg IK fails | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| broad baseline | 50 | 22.7 | 107.1 | 96.2 | 0.0 | best stable transport trend |
| phase-weighted, scale `0.75` | 50 | 13.2 | 121.3 | 153.6 | 10.0 | worse transport and IK |
| phase-weighted, scale `0.75` | 100 | 8.9 | 108.9 | 177.9 | 0.5 | better contact, much worse target |
| phase-weighted, scale `0.45` | 50 | 7.7 | 93.5 | 106.5 | 0.0 | stable contact gain, no transport gain |
| phase-weighted, scale `0.45` | 100 | 13.4 | 77.8 | 113.7 | 2.0 | similar to baseline step 100 |

Conclusion: per-step loss weighting is useful infrastructure but not a
breakthrough by itself. It improves contact proximity in some checkpoints, but
it does not produce learned held-object transport or release, and the strongest
stable final-target result remains the unweighted broad step-50 checkpoint.
This closes the scalar tuning loop: more residual scale, more epochs, or more
loss-weight sweeps on this same warm-start objective are unlikely to be the
most effective path.

Updated next step:

1. Preserve per-step action-loss weighting because it is generic and tested,
   but treat the phase-weighted runs as negative diagnostics.
2. Generate a larger, filtered set of successful attached-object positives that
   includes explicit hold, lift, carry, lower, and release phases across more
   episodes and offsets.
3. Add an explicit learned phase/token objective for post-contact transport so
   the model predicts the action type, not only a weighted continuous residual.
4. Recompute or export chunk-local DHB invariant/action-token targets for the
   attached traces; the current attached-positive loop trains action residuals
   but still warns that DHB invariant targets are absent.
5. Run the matched regular-head control on the same augmented positives before
   claiming a DHB-XR transport advantage.

## Chunk-Local DHB Target Retrofit

The attached-positive supervision gap was then fixed directly. New policy-trace
exports now include chunk-local `target_linear_invariants` and
`target_angular_invariants`, computed from the raw action chunk and the saved
chunk-start state. The policy-trace dataset now emits those fields as
`dhb.linear_invariants` and `dhb.angular_invariants`, so QwenDHB can supervise
the invariant branch and derive OAT/discrete token labels from the same data.

Implementation commits:

- parent repo: `d00ec3c Export DHB targets for policy trace positives`
- vendor StarVLA: `c8b5f38 Expose policy trace DHB invariant targets`
- parent repo: `863836d Add policy trace DHB target augmentation`

Regression:

`PYTHONPATH=vendor/starVLA:src:${PYTHONPATH:-} ./.pixi-home/bin/pixi run pytest tests/test_policy_trace_contact_dataset.py tests/test_libero_postcontact_correction_plan.py tests/test_action_head.py -q`

Result: `53 passed`.

The existing broad `65`-row manifest was retrofitted without rerunning LIBERO:

`_workspace/paper_eval/2026-04-28_attached_replay_lifttransport_gated_step50_100_dhbtargets_manifest.jsonl`

Summary:

- rows: `65`
- invariant-supervised steps: `472`
- finite targets: yes
- nonzero linear-invariant rows: `65/65`

A bounded 100-step continuation was trained from the same broad objective with
`lambda_invariant=0.15`, training the closed adapter plus invariant head:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_attached_transport_objectpair_lifttransport_broad_dhbtargets_fromdagger100`

Training was mechanically healthy: `loss_inv_raw` dropped from about `0.048`
at step `50` to `0.021` at step `100`, and the previous missing-invariant
warning is gone.

Controlled no-gate sweep, `rollout_seed=70`, cases `0:0` and `1:+3 cm`:

| Run | Step | Avg min EE-bowl XY mm | Avg min plate XY mm | Avg final bowl-plate XY mm | Avg IK fails | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| broad baseline | 50 | 22.7 | 107.1 | 96.2 | 0.0 | best stable transport trend |
| DHB-target continuation | 50 | 22.0 | 109.1 | 105.3 | 0.0 | improves `+3 cm`, hurts episode 0 |
| DHB-target continuation | 100 | 24.9 | 101.7 | 110.7 | 0.0 | removes IK but no target breakthrough |

Conclusion: chunk-local DHB target export is a necessary correctness fix and a
useful token-supervision bridge, but invariant loss alone does not solve
post-contact transport. It improves the non-zero-offset final target distance
in this two-case diagnostic while regressing the zero-offset case, so the
missing piece is still an explicit held-object phase/action-type objective and
more diverse target-reaching positives.
