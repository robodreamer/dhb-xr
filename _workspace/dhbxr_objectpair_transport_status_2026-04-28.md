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
