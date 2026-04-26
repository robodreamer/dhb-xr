# DHB-XR Matched Head Comparison v1

## Purpose

This note summarizes the current evidence comparing DHB-XR action heads against
regular StarVLA action heads under matched conditions. It is an internal claim
discipline artifact, not a final paper result.

## Comparison Boundary

The fairest regular-head control currently available is:

- model: `QwenOFT`
- base VLM: `Qwen2.5-VL-3B-Instruct`
- dataset: `libero_spatial_dhbxr`
- training budget: `200` steps
- frozen module policy: same local frozen-Qwen setup
- checkpoint: `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/checkpoints/steps_200_pytorch_model.pt`

The matched DHB reference is:

- model: `QwenDHB`
- base VLM: `Qwen2.5-VL-3B-Instruct`
- dataset: `libero_spatial_dhbxr`
- checkpoint: `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- primary execution path: `predict_trajectory() -> embodik IK -> JOINT_POSITION sim`

## Current Evidence

### Calibrated Direct-Action Reach Diagnostic

A newer diagnostic branch establishes that the DHB-trained model can produce a
stable reach controller when executed through direct delta-action integration
with a small global XYZ calibration:

| Model/checkpoint | Path | Calibration | Reach offsets | Reach success | IK failures |
| --- | --- | --- | --- | ---: | ---: |
| `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400` | `direct_action` | `1.0,1.35,1.15` XYZ scale | `0,+3,-3 cm` | `5/5` | `0` |

Artifact:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_offsets.json`

Interpretation: this is not yet a matched regular-head paper comparison, but
it is important engineering evidence. It separates the remaining full-task
failure from basic reach, IK, and rotation bugs. The full-task blocker is now
phase timing: reaching close-ready geometry happens too late for the learned or
replayed gripper sequence to complete grasp, lift, transport, and release.

### Sample-Level Prediction

On the same real-sample predict smoke:

| Model | Chunk MAE | Predict time | Peak memory |
| --- | ---: | ---: | ---: |
| `QwenDHB` | `0.1545` | `683.41 ms` | `4193.97 MB` |
| `QwenOFT` | `0.2626` | `616.81 ms` | `4161.64 MB` |

Interpretation: under this short training budget, DHB has a cleaner supervised
chunk prediction signal than the regular head. The runtime footprint is close
enough that this is not a memory or latency story.

### Direct LIBERO Rollout

| Model | Direct path success | Steps |
| --- | ---: | ---: |
| `QwenDHB` direct `predict_action()` | `0/1` | `220` |
| `QwenOFT` direct `predict_action()` | `0/1` | `220` |

Interpretation: direct controller rollout is not the right differentiator yet.
Both paths fail under the direct action contract, and prior diagnostics show the
DHB direct path under-moves in position even after normalization and
absolute-vs-delta bugs were fixed.

### Matched EmbodiK Execution

The stronger comparison routes both policies through the same downstream
`embodik` evaluator:

| Model | Path | `0 cm` | `3 cm` |
| --- | --- | ---: | ---: |
| `QwenDHB` | `predict_trajectory() -> embodik` | `1/3` | `1/3` |
| `QwenOFT` | delta-action integration -> `embodik` | `0/3` | `0/3` |

Important qualifier: the `QwenOFT` Stage-2 IK was clean. All `6/6` cases were
IK-feasible with mean FK error around `0.48 mm`, so the regular-head failure is
not an IK-contract artifact.

### 2026-04-25 Paper-Safe Task-0 Rerun

The first regenerated paper-safe task-0 comparison now uses the same episode
set for both models: episodes `0,1,10`, offsets `0 cm` and `3 cm`, the same
Stage-1 knobs, the same pack step, and the same downstream IK/sim evaluator.

Artifact summary:

`_workspace/paper_eval/2026-04-25_matched_task0/summary.md`

| Model | `0 cm` final success | `3 cm` final success | IK failures |
| --- | ---: | ---: | ---: |
| `QwenDHB` | `1/3` | `1/3` | `140` |
| `QwenOFT` | `0/3` | `0/3` | `0` |

Interpretation: the matched task-success pattern survives the corrected
episode set, but the result is not execution-quality clean. `QwenDHB` wins on
final task success, while `QwenOFT` has cleaner IK feasibility. The safe claim
is therefore a bounded prototype task-outcome advantage under this evaluator,
with an explicit caveat that DHB late-trajectory feasibility remains a gap.

Follow-up diagnosis: the old `QwenDHB` checkpoint was trained with static
per-timestep DHB invariant rows sliced as action chunks. That is a real target
construction bug for DHB-DR, because frame initialization depends on the chunk
start. On task-0 converted data, those sliced rows reconstruct sampled chunks
with `10-73 mm` error, while chunk-local recomputation from raw delta actions
reconstructs with `0.0 mm` error. Therefore this comparison remains useful as a
historical prototype control, but the next clean matched comparison should use
a freshly trained DHB checkpoint with
`recompute_dhb_invariants_from_actions: true` and current-ending
`state_history_window: 4`.

## Strongest Safe Claim

The current matched evidence supports this claim:

> Under the current short StarVLA budget and the same `embodik` evaluator,
> DHB-XR produces a more useful structured trajectory signal than the matched
> regular `QwenOFT` action head.

After the 2026-04-25 rerun, state the claim with the IK caveat:

> On the paper-safe task-0 rerun, continuous DHB preserves a final-success
> advantage over matched QwenOFT, but DHB also produces late trajectory segments
> with more IK infeasibility. This is a prototype outcome result, not yet a
> clean execution-quality superiority result.

It does not yet support this stronger claim:

> DHB-XR generally beats regular action heads at benchmark scale.

## Why This Still Matters

The comparison closes a major fairness gap: the regular-head control now uses
the same base VLM, same dataset family, same short training budget, and the same
downstream IK/sim stack. The evidence favors DHB-XR, but the margin is still at
prototype scale.

## Remaining Comparison Gaps

- The comparison is still short-budget and narrow-task, not benchmark-scale.
- The DHB baseline itself is only around `1/3` at each offset in the current
  shared offline setting, so superiority over `QwenOFT` is not yet a strong
  task-success story.
- Direct rollout is not a useful final metric until the action contracts are
  equally calibrated.
- The next comparison should reuse one centralized adapter contract for history,
  reference handling, gripper decoding, and rollout chunk policy.

## Next Validation Milestones

1. Freeze the canonical StarVLA `predict_trajectory() -> embodik` contract.
2. Run `QwenDHB` and `QwenOFT` through the same broader evaluator without
   per-model rollout exceptions.
3. Report success, IK feasibility, placement geometry, and videos together.
4. Only after the shared evaluator is stable, broaden to seed-size and
   augmentation studies.

## Next Matrix

Immediate paper-safe matrix:

- Models: `QwenDHB` short-budget checkpoint and matched `QwenOFT` short-budget
  checkpoint.
- Task: `libero_spatial` task ID `0`.
- Episodes: `0,1,10` for both models. This fixes the current artifact mismatch:
  existing DHB canonical files used `0,1,2`, while QwenOFT files used
  `0,1,10`.
- Offsets: `0 cm` and `3 cm`.
- Execution path: one frozen `predict_trajectory() -> embodik -> sim` adapter
  with identical smoothing, gripper, replan, and IK settings for both models.
- Metrics: success, final object-relative placement error, closest bowl
  approach distance, IK failure count, FK residual, path efficiency, video, and
  predict-time/memory.

Broader matrix after Phase A is reproducible:

- Tasks: `libero_spatial` task IDs `0,1,2`.
- Seeds/episodes: `0,1,2` under the same seed semantics for every task.
- Offsets: `0 cm` and `3 cm`.
- Size: `2` models x `3` tasks x `3` seeds x `2` offsets = `36` default-attempt
  rollouts.

Pass/fail discipline:

- Claim a prototype DHB advantage only if `QwenDHB` beats `QwenOFT` on success
  or object-relative placement without worse IK feasibility.
- Require at least two distinct task/seed pairs with DHB-only wins before
  describing the broader matrix as more than a single-case prototype.
- Do not claim benchmark-scale superiority until the same pattern holds across
  a broader task set or seed-size curve.
- Treat direct LIBERO action rollout as diagnostic until the direct action
  contracts are equally calibrated.
