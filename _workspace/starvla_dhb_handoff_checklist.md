# StarVLA DHB Handoff Checklist

## SmolVLA Baseline To Beat

- Checkpoint: `_workspace/dhb_head_e2e_fix_smoke.pt`
- Evaluation path: `_workspace/eval_dhb_head_embodik.py`
- Validation protocol: offsets `0cm` and `3cm`, `3` seeds, recorded videos
- Current executable baseline:
  - `0cm`: `2/3`
  - `3cm`: `1/3`

## What Is Already Validated

- DHB state-forwarding bug is fixed.
- Persistent EE rotation / orientation runaway is gone.
- `embodik` IK is stable and not the bottleneck.
- The current bottleneck is closed-loop placement robustness.

## Failure Signature To Target

Primary failing demos:
- `seed 1` -> `demo_1`
- `seed 2` -> `demo_10`

Dominant failure mode:
- under-reaching in `Y` at final placement
- secondary effect: slightly high final `Z`
- not primarily a grasp/release timing issue

Representative final offsets relative to the plate from the SmolVLA smoke diagnostic:
- `demo_0` success at `0cm`: `ee_minus_plate ~= [0.018, -0.020, 0.072]`
- `demo_1` fail at `0cm`: `ee_minus_plate ~= [0.001, -0.047, 0.075]`
- `demo_10` fail at `0cm`: `ee_minus_plate ~= [-0.006, -0.055, 0.073]`

## Required Evaluation Settings

- Use `embodik` as the primary evaluation path.
- Record videos for every validation run.
- Keep per-demo diagnostics enabled in sim:
  - first close step
  - first reopen step
  - final EE pose
  - object-relative offsets to bowl and plate

## Go / No-Go Rule For StarVLA

StarVLA is worth continuing if it improves either:
- overall success rate over the SmolVLA smoke baseline, or
- final `Y` placement offset for `demo_1` and `demo_10`

Do not treat lower train loss alone as progress. SmolVLA already showed that train loss can improve substantially without improving closed-loop execution.

## First StarVLA Experiments

1. Use the lightweight `Qwen2.5-VL-3B` + DHB path as the development baseline on the laptop GPU.
2. Reproduce the SmolVLA smoke validation protocol as closely as possible, starting with single-episode LIBERO rollout smoke checks.
3. Compare `demo_0`, `demo_1`, and `demo_10` side by side once rollout videos exist.
4. Check whether final `ee_minus_plate[1]` improves on the failing demos.
5. Only then broaden to larger sweeps or longer training.

## Current StarVLA Readiness

- Vendor patch present:
  - `vendor/starVLA/starVLA/model/framework/WM4A/WM4A_DHB.py`
  - `vendor/starVLA/starVLA/model/modules/action_model/DHBActionHeader.py`
- Lightweight laptop path present:
  - `vendor/starVLA/starVLA/model/framework/VLM4A/QwenDHB.py`
  - `configs/libero_dhbxr_qwen.yaml`
- Local LeRobot converter is patched for the current LIBERO HDF5 layout.
- StarVLA dataloader smoke passes on both:
  - single-demo export
  - multi-task smoke export
- Active dataset rebuild target:
  - `/home/andypark/Projects/data/libero_lerobot`
- Qwen+DHB laptop smokes now pass:
  - forward smoke
  - predict smoke
  - one-step training smoke

## Current Blocking Issues

1. The heavy Cosmos/WM4A path is still too expensive for iterative laptop debugging.
2. Harness-side StarVLA integration is not yet wired to the Qwen+DHB local path.
3. The first direct rollout smoke now works, but it still uses the LIBERO controller directly and shows control / nullspace issues that should be compared against an `embodik` execution path.

## Immediate Next Steps

1. Review the first direct local LIBERO rollout smoke video for `Qwen+DHB`.
2. Compare that failure mode against the SmolVLA smoke baseline on the same task family.
3. Then decide whether to:
   - wire `Qwen+DHB` into `vla-evaluation-harness`, or
   - add a slightly broader StarVLA rollout sweep first.

## First Direct Rollout Baseline

- Script: `scripts/starvla_dhb_libero_rollout_smoke.py`
- Config: `configs/libero_dhbxr_qwen.yaml`
- Task: `libero_spatial/task_id=0/episode_idx=0`
- Outcome: failure after `220` steps
- Runtime: about `66.1 s`
- Peak GPU memory: about `4286.9 MB`
- Artifact:
  - `_workspace/starvla_runs/dhb_starvla_qwen/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`

Interpretation:
- The direct StarVLA+DHB rollout stack is now executable end to end on the laptop GPU.
- The next question is no longer “can it run?” but “does its failure mode match the known SmolVLA placement-margin signature?”

## EmbodiK Bridge Status

- New Stage-1 export script:
  - `scripts/starvla_dhb_embodik_stage1.py`
- New capability:
  - `QwenDHB.predict_trajectory(...)` now returns DHB-decoded positions and quaternions, not only normalized actions
- Latest Stage-1 artifact from the `steps_200` checkpoint:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_stage1_step200.npz`
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_stage1_step200_summary.json`

Interpretation:
- The Qwen+DHB StarVLA path can now emit SE(3) chunks directly from a live LIBERO observation.
- That is the correct handoff point for `embodik`, and should reduce the nullspace sensitivity seen in the direct LIBERO controller rollout.

## First Trained Checkpoint Comparison

- Short training run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2`
- Intermediate checkpoint:
  - `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2/checkpoints/steps_50_pytorch_model.pt`
- Same-task rollout summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2/libero_rollout_smoke_summary_step50.json`

Current reading:
- Short training is enough to prove the checkpoint-save / reload loop works on the laptop path.
- It is not enough to fix task success or clean up EE orientation behavior.
- The remaining issue is no longer the original normalized-action rollout bug; it is now either:
  - insufficient training, or
  - remaining mismatch in dataset / action conventions that should be cleaned up before broader sweeps.

## Longer Checkpoint Status

- Longer run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_longtrain1`
- Intermediate checkpoints:
  - `steps_100_pytorch_model.pt`
  - `steps_200_pytorch_model.pt`
- Latest same-task rollout summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_longtrain1/libero_rollout_smoke_summary_step200.json`
- Latest video:
  - `_workspace/starvla_runs/dhb_starvla_qwen/rollout_steps_200_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`

Current reading:
- `200` steps are still not enough for success on the reference LIBERO rollout.
- The checkpoint-specific rollout path now works, but reloading quantized checkpoints needed `llm_int8_enable_fp32_cpu_offload=true`.
- That reload path emitted meta-parameter warnings, so benchmark-scale evaluation should wait until checkpoint reload semantics are cleaned up or moved to a non-fragile loading path.
