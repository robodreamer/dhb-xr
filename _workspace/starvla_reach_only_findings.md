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

## 2026-04-20 Follow-Up: Corrective Reach Augmentation Failed

A first rollout-grounded corrective augmentation run was tested at:

- run dir:
  `_workspace/starvla_runs/dhb_starvla_qwen_reachcorr`
- config:
  `configs/libero_dhbxr_qwen_reachcorr.yaml`
- warm-start source:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`

Result:

| Step | Avg min bowl XY (m) | Avg min bowl XYZ (m) | Avg IK fails |
| ---: | ---: | ---: | ---: |
| 100 | 0.1970 | 0.3304 | 75.5 |
| 200 | 0.1736 | 0.3331 | 76.5 |
| 300 | 0.1989 | 0.3506 | 77.5 |

Interpretation:

- this did **not** beat the old reach-only baseline (`0.1465 / 0.2954 / 75.5`)
- the likely failure mode is that the corrective rewrite targeted the wrong
  phase:
  - it overemphasized close-phase chunks
  - it rewrote state and targets without changing the images
- so the model learned to emit late pre-grasp geometry too early rather than a
  better open-gripper approach

Conclusion:

- keep the idea documented, but do not treat this augmentation as the current
  path forward
- do not spend more budget on the same corrective-rewrite setup

## 2026-04-20 Follow-Up: Reach-Phase Oversampling Also Failed To Beat Baseline

A cleaner reach-focused warm-start was then tested:

- run dir:
  `_workspace/starvla_runs/dhb_starvla_qwen_reachphase1`
- config:
  `configs/libero_dhbxr_qwen_reachphase.yaml`
- intervention:
  oversample real open-gripper approach chunks so the close phase lands near the
  end of the horizon (`gripper_event_target_pos=6`)

Result:

| Step | Avg min bowl XY (m) | Avg min bowl XYZ (m) | Avg IK fails |
| ---: | ---: | ---: | ---: |
| 100 | 0.1913 | 0.3289 | 75.5 |
| 200 | 0.1963 | 0.3475 | 76.5 |
| 300 | 0.2498 | 0.3738 | 80.5 |

Interpretation:

- this is cleaner than the synthetic corrective rewrite
- but it still does **not** beat the original `shorttrain1/steps_200` reach-only
  checkpoint

## 2026-04-20 Follow-Up: Committed-Handoff Chunk Runner Improves Reach Without Retraining

A structural execution-side follow-up was tested instead of another warm-start:

- add ramped overlap blending in the shared chunk runner
- optionally commit the first `k` overlap steps to the previously predicted tail
- keep the same best checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`

Probe settings on the two reference live cases:

- `blend_mode = ramp`
- `commit_prefix_steps = 2`
- same loop otherwise:
  - `replan_every = 4`
  - `max_first_offset_mm = 35`
  - `max_step_mm = 20`
  - `smooth_window = 3`
  - `blend_overlap_alpha = 0.6`

Reference artifacts:

- probe JSONs:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/handoff_probe/*.json`

Per-case improvement against the old canonical reach-only loop:

### Episode 0, offset 0 cm

- old:
  - `min_bowl_xy_m = 0.1590`
  - `min_bowl_xyz_m = 0.2974`
  - `ik_fail_total = 82`
- new (`ramp + commit2`):
  - `min_bowl_xy_m = 0.1410`
  - `min_bowl_xyz_m = 0.2799`
  - `ik_fail_total = 82`

### Episode 1, offset 3 cm

- old:
  - `min_bowl_xy_m = 0.1339`
  - `min_bowl_xyz_m = 0.2935`
  - `ik_fail_total = 69`
- new (`ramp + commit2`):
  - `min_bowl_xy_m = 0.1223`
  - `min_bowl_xyz_m = 0.2862`
  - `ik_fail_total = 68`

Full reach-only checkpoint sweep under the new handoff:

- all checkpoint case files:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/step_*/*.json`

Updated ranking:

| Rank | Step | Avg min bowl XY (m) | Avg min bowl XYZ (m) | Avg IK fails | Avg accel (mm) | Avg hold frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0.1375 | 0.2867 | 75.5 | 0.2583 | 0.8346 |
| 2 | 100 | 0.2287 | 0.3672 | 76.0 | 0.2843 | 0.8388 |
| 3 | 50 | 0.2500 | 0.3739 | 80.5 | 0.1688 | 0.8834 |
| 4 | 150 | 0.2500 | 0.3740 | 79.0 | 0.1972 | 0.8652 |

Interpretation:

- this is the first post-training change that clearly improves the reach-only
  bowl metric on the best checkpoint
- the gain is modest but consistent across both reference cases
- it is strong enough to keep as the current best reach-only execution mode
- but it does **not** justify replacing the full integrated pipeline blindly

## 2026-04-20 Follow-Up: Integrated Full Task Regresses Under The Same Handoff

The improved reach-only handoff was then pushed back into the full offline
`StarVLA -> DHB -> embodik` pipeline with demo gripper timing:

- Stage 1:
  `scripts/starvla_dhb_embodik_stage1.py`
  with `blend_mode = ramp`, `commit_prefix_steps = 2`
- Stage 2:
  `_workspace/eval_dhb_head_embodik.py --stage ik`
- Stage 3:
  `_workspace/eval_dhb_head_embodik.py --stage sim`

Artifacts:

- IK:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_ramp_commit2_ik_ik.json`
- sim:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_ramp_commit2_sim_sim.json`
- videos:
  `_workspace/validation_videos/starvla_embodik_invsup_step200_replan4_clamp35_step20_sw3_blend06_ramp_commit2_fullsweep`

Result:

- `0 cm`: `0/3`
- `3 cm`: `0/3`

Compared to the old canonical integrated baseline:

- old:
  - `0 cm`: `1/3`
  - `3 cm`: `1/3`
- new handoff:
  - `0 cm`: `0/3`
  - `3 cm`: `0/3`

Interpretation:

- the committed-handoff improvement is **phase-local**
- it helps open-gripper bowl approach
- but once the full manipulation sequence is reintroduced, the same stronger
  commitment hurts downstream grasp / transport / placement behavior

Current conclusion:

- `ramp + commit2` is now the best **reach-only** execution mode
- it is **not** yet the new integrated default
- the right next integrated design is likely phase-split:
  - use the committed-handoff mode for pre-grasp reach
  - then hand off to the old canonical integrated loop for the rest
- more same-family warm-start training is therefore no longer the highest-value
  move

Current recommendation:

- keep `shorttrain1/steps_200` as the reach-only control baseline
- stop iterating on small sampling-only warm-starts
- prioritize a more structural next change:
  - future-state / committed-tail chunk handoff, or
  - history-aware continuation conditioning
