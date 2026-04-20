# StarVLA Reach-Only Findings

Date: 2026-04-20

## Goal

Simplify the current StarVLA+DHB evaluation target from full pick-and-place to:

1. keep the gripper open
2. ignore close / reopen timing
3. evaluate whether the policy can approach the bowl smoothly under offsets

This removes the current learned-gripper phase-alignment bottleneck and makes
the next model-quality target much clearer.

## New Reach-Only Evaluation Contract

The live and offline evaluators now support a dedicated reach-only mode:

- [scripts/starvla_dhb_embodik_live_rollout.py](/home/andypark/Projects/repos/dhb-xr/scripts/starvla_dhb_embodik_live_rollout.py)
- [_workspace/eval_dhb_head_embodik.py](/home/andypark/Projects/repos/dhb-xr/_workspace/eval_dhb_head_embodik.py)

Key behavior:

- force gripper open (`-1.0`)
- ignore close / reopen transitions as a success signal
- define success as entering a bowl-relative reach region:
  - `bowl_xy <= 0.05 m`
  - `bowl_z <= 0.08 m`
- optionally stop as soon as that reach criterion is met
- log smoothness metrics from the executed EE trace via
  [compute_motion_metrics()](/home/andypark/Projects/repos/dhb-xr/src/dhb_xr/integration/vla/chunk_rollout.py)

Tracked metrics now include:

- `min_bowl_xy_m`
- `min_bowl_xyz_m`
- `ik_fail_total`
- `path_efficiency`
- `mean_accel_mm`
- `hold_fraction`
- `direction_flip_fraction`

## Reach-Only Checkpoint Sweep

Sweep script:

- [scripts/starvla_dhb_live_checkpoint_sweep.py](/home/andypark/Projects/repos/dhb-xr/scripts/starvla_dhb_live_checkpoint_sweep.py)

Run settings:

- config: `configs/libero_dhbxr_qwen.yaml`
- cases:
  - `episode 0`, `offset 0 cm`
  - `episode 1`, `offset 3 cm`
- loop:
  - `replan_every = 4`
  - `max_first_offset_mm = 35`
  - `max_step_mm = 20`
  - `smooth_window = 3`
  - `blend_overlap_alpha = 0.6`
- gripper:
  - `demo`
- reach:
  - `reach_only = true`
  - `reach_threshold_xy_m = 0.05`
  - `reach_threshold_z_m = 0.08`
  - `reach_stop_on_success = true`

Sweep artifacts:

- JSON:
  [summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.json)
- Markdown:
  [summary.md](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md)

Ranking:

| Rank | Step | Avg min bowl XY (m) | Avg min bowl XYZ (m) | Avg IK fails | Avg accel (mm) | Avg hold frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.1465 | 0.2954 | 75.5 | 0.2306 | 0.8346 |
| 2 | 100 | 0.2280 | 0.3672 | 76.0 | 0.2385 | 0.8418 |
| 3 | 50 | 0.2500 | 0.3739 | 80.5 | 0.1602 | 0.8836 |
| 4 | 150 | 0.2500 | 0.3740 | 79.5 | 0.1760 | 0.8700 |

Interpretation:

- `steps_200` is still the best current checkpoint even after gripper timing is
  removed from the benchmark
- the improvement is not coming from gripper luck; it is coming from better bowl
  approach geometry
- none of the current checkpoints actually enter the reach-success region yet

## Canonical vs Raw-ish Execution

To test whether the visible jitter is mostly caused by our post-processing and
execution stack rather than by the predicted trajectory, the best checkpoint was
rerun in a stripped-down execution mode:

- `max_first_offset_mm = 0`
- `max_step_mm = 0`
- `smooth_window = 0`
- `blend_overlap_alpha = 0`
- `substeps = 1`

Artifacts:

- [ep0 raw-ish summary](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_reach_rawish_step200_summary.json)
- [ep1 raw-ish summary](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep1_offset3_reach_rawish_step200_summary.json)

Comparison against canonical `steps_200`:

### Episode 0, offset 0 cm

Canonical:

- `min_bowl_xy_m = 0.1590`
- `min_bowl_xyz_m = 0.2974`
- `ik_fail_total = 82`
- `mean_accel_mm = 0.1851`
- `hold_fraction = 0.8390`
- `direction_flip_fraction = 0.0357`
- `path_efficiency = 0.9695`

Raw-ish:

- `min_bowl_xy_m = 0.1996`
- `min_bowl_xyz_m = 0.3300`
- `ik_fail_total = 94`
- `mean_accel_mm = 0.4263`
- `hold_fraction = 0.8924`
- `direction_flip_fraction = 0.2353`
- `path_efficiency = 0.7111`

### Episode 1, offset 3 cm

Canonical:

- `min_bowl_xy_m = 0.1339`
- `min_bowl_xyz_m = 0.2935`
- `ik_fail_total = 69`
- `mean_accel_mm = 0.2762`
- `hold_fraction = 0.8301`
- `direction_flip_fraction = 0.0189`
- `path_efficiency = 0.9640`

Raw-ish:

- `min_bowl_xy_m = 0.1480`
- `min_bowl_xyz_m = 0.3010`
- `ik_fail_total = 76`
- `mean_accel_mm = 0.8571`
- `hold_fraction = 0.8194`
- `direction_flip_fraction = 0.1667`
- `path_efficiency = 0.7944`

Interpretation:

- the stripped-down run is **worse** on both bowl proximity and smoothness
- the current clamp / smoothing / overlap settings are helping feasibility more
  than they are causing the main failure
- visible jitter is not primarily a post-processing artifact
- the remaining bottleneck is still the predicted trajectory itself

## Connection To `vla-ideas`

The most relevant idea in
[/home/andypark/Projects/repos/vla-ideas](/home/andypark/Projects/repos/vla-ideas)
is the async chunking work:

- stale-state async replanning is bad
- hard prefix commitment can introduce oscillation
- future-state-aware planning helps

That is directionally useful, but the reach-only results here show something
important:

- our current live loop already replans from live observation
- removing the execution-side smoothing does **not** improve approach quality

So the immediate lesson from `vla-ideas` is not “strip out all execution
processing.” It is:

- keep the current stabilizing execution loop as the baseline
- improve the model’s approach trajectory quality first
- then revisit more future-state-aware or committed-prefix-aware training if the
  reach-only plateau remains

## Current Conclusion

The correct next local objective is now:

- **reach the bowl smoothly under offsets**

and not:

- learn close / reopen timing immediately

This is the cleanest way to reduce the current failure to one thing at a time.

