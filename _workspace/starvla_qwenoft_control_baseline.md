# StarVLA QwenOFT Control Baseline

Date: 2026-04-19

## Goal

Create the first matched regular-head control for the corrected StarVLA DHB work:

- same base VLM: `Qwen2.5-VL-3B-Instruct`
- same dataset: `libero_spatial_dhbxr`
- same short training budget: `200` steps
- same frozen-Qwen setting
- same direct LIBERO rollout smoke task:
  - `libero_spatial`
  - `task_id=0`
  - `episode_idx=0`

This note now covers two levels of control:

1. direct-action rollout smoke
2. matched `QwenOFT -> delta-action integration -> embodik` evaluation

So it now closes the most important comparison gap that was still open on
2026-04-19: `QwenOFT` has been run under the same `embodik` Stage 2/3 path as
the DHB baseline.

## Control config

- config: `configs/libero_qwenoft_qwen.yaml`
- framework: `QwenOFT`
- quantization:
  - `load_in_8bit: true`
  - `device_map: auto`
- trainer overrides used:
  - `trainer.freeze_modules=qwen_vl_interface`
  - `trainer.max_train_steps=200`
  - `trainer.save_interval=50`
  - `trainer.eval_interval=50`
  - `trainer.logging_frequency=10`

## Artifacts

### QwenOFT short-train run

- run dir:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1`
- checkpoint:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- rollout summary:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/libero_rollout_smoke_summary_step200.json`
- predict summary:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/predict_smoke_summary_step200.json`
- rollout video:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen/rollout_steps_200_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`
- matched `embodik` stage-1 control chunks:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/embodik_stage1_ep0_step200.npz`
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/embodik_stage1_ep1_step200.npz`
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/embodik_stage1_ep10_step200.npz`
- matched `embodik` sweep:
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/embodik_step200_fullsweep_debug_ik.json`
  - `_workspace/starvla_runs/qwenoft_starvla_qwen_shorttrain1/embodik_step200_fullsweep_debug_sim.json`
  - `_workspace/validation_videos/qwenoft_embodik_step200_fullsweep`

### Reference DHB short-train run

- run dir:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1`
- checkpoint:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- direct rollout summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/libero_rollout_smoke_summary_step200_normfix.json`
- predict summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/predict_smoke_summary_step200.json`

## First comparison

### Predict smoke on one real sample

| model | chunk MAE | predict time (ms) | peak memory (MB) |
| --- | ---: | ---: | ---: |
| `QwenDHB` | `0.1545` | `683.41` | `4193.97` |
| `QwenOFT` | `0.2626` | `616.81` | `4161.64` |

Observations:

- The matched `QwenOFT` control is clearly worse on sample-level action
  prediction error under the same short-train budget.
- The runtime footprint is similar, so this is not a trivial memory / latency
  artifact.

### Direct LIBERO rollout smoke

| model | success | steps | rollout time (ms) |
| --- | ---: | ---: | ---: |
| `QwenDHB` direct `predict_action()` path | `0/1` | `220` | `9644.59` |
| `QwenOFT` direct `predict_action()` path | `0/1` | `220` | `12479.34` |

Observations:

- Neither short-trained direct-action path solves the task.
- This is consistent with the broader StarVLA findings so far:
  - the direct controller path is not the strongest execution path
- the DHB trajectory path remains the more promising route for embodied
  validation

### Matched `embodik` control

The direct-action comparison above was still weaker than the current DHB
baseline, because DHB is strongest through:

- `predict_trajectory()`
- `embodik` IK
- `JOINT_POSITION` sim

So `QwenOFT` was also run through the same downstream evaluator by:

- unnormalizing its predicted delta-action chunk
- integrating that chunk into EE waypoints
- feeding those waypoints into the same `embodik` Stage 2/3 stack

Bridge files:

- `scripts/starvla_oft_embodik_stage1.py`
- `src/dhb_xr/integration/vla/libero.py`

Result under the same offline `0 cm / 3 cm / 3-seed` evaluator:

| model | path | `0 cm` | `3 cm` |
| --- | --- | ---: | ---: |
| `QwenDHB` | canonical `predict_trajectory() -> embodik` | `1/3` | `1/3` |
| `QwenOFT` | integrated delta-action chunk -> `embodik` | `0/3` | `0/3` |

Important details:

- Stage 2 was clean for the `QwenOFT` control:
  - all `6/6` cases were IK-feasible
  - mean FK error stayed about `0.48 mm`
- so the `0/3, 0/3` result is not an IK-contract artifact
- it is a real matched execution-side baseline

One qualitative nuance:

- `offset=0 cm, seed 0` and `seed 2` each reached transient `reward=1.00`
  during sim
- but all runs still finished as `success=False`
- so the honest benchmark summary remains `0/3` at both offsets

### First-action behavior on the rollout task

`QwenDHB` direct rollout first action:

```json
[0.0014783, 0.0002695, -0.0028447, 0.0077643, -0.0127418, -0.0030956, 1.0]
```

`QwenOFT` direct rollout first action:

```json
[-0.0074953, 0.0006187, 0.0032836, -0.0131009, -0.0081004, -0.0022187, 1.0]
```

Interpretation:

- `QwenDHB` direct rollout is more conservative in translation.
- `QwenOFT` moves more immediately, but that did not translate into success.
- So the current evidence is **not** “regular head moves more, therefore better.”
  It is simply a different failure mode under the same short budget.

## Current conclusion

This closes one part of the gap:

- we now have a real StarVLA regular-head control path
- it trains on the same corrected dataset
- it can be rolled out and compared against the corrected DHB path

What it still does **not** close:

- final claim strength over a regular action head at larger scale
- live / online matched comparison
- whether the DHB margin comes from representation quality alone or also from
  the current execution design

## Immediate next step

Keep using this `QwenOFT` setup as the regular-head reference while continuing
to improve the canonical DHB -> `embodik` path. The next honest comparison
target is no longer “make the control fairer”; that is now done. The next gap
is benchmark infrastructure quality:

- centralize the history/reference/gripper adapter contract
- then carry both `QwenDHB` and `QwenOFT` through the same broader benchmark
  plumbing
