# StarVLA Embodik Seed Diagnostic

## Scope

This note compares the latest replanned StarVLA `Qwen+DHB -> predict_trajectory() -> embodik -> LIBERO sim`
path across seeds `0`, `1`, and `2` for the reference LIBERO spatial task.

Artifacts:

- Stage 1 summaries:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed0_replan6.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed1_replan6.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed2_replan6.json`
- IK debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_debug_ik.json`
- Sim debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_debug_sim.json`

## Main finding

The remaining failure split looks like **trajectory quality / feasibility**, not another hidden 180-degree orientation bug:

- `seed 0`: partially IK-feasible and succeeds in sim at `0 cm`, and at `3 cm` with higher gains
- `seed 1`: fails IK from the first waypoint on every step
- `seed 2`: partly IK-feasible but still fails the task

## What changed relative to the earlier broken runs

These execution bugs were already fixed before this comparison:

- rollout delta normalization / stale stats mismatch
- LIBERO quaternion convention mismatch
- gripper label threshold bug in the converter
- single short Stage-1 chunk bug

So the current seed split should not be interpreted as the old rollout bug resurfacing.

## Seed-wise Stage 1 behavior

All three seeds still overshoot the first demo waypoint, especially in `+x`, but `seed 1` is the worst:

- `seed 0`
  - first position error: `51.8 mm`
  - first delta: `[+51.5, +5.9, +1.5] mm`
  - first rotation delta: `2.63 deg`
  - mean predicted step size: `28.63 mm`
- `seed 1`
  - first position error: `71.4 mm`
  - first delta: `[+71.0, +7.9, -1.9] mm`
  - first rotation delta: `2.61 deg`
  - mean predicted step size: `35.46 mm`
- `seed 2`
  - first position error: `59.7 mm`
  - first delta: `[+31.1, +18.8, -23.5] mm`
  - first rotation delta: `3.95 deg`
  - mean predicted step size: `36.40 mm`

The first-step orientation deltas are modest. The dominant error is translational overshoot.

## IK consequences

At `0 cm`:

- `seed 0`: `25/98` IK failures
- `seed 1`: `84/84` IK failures
- `seed 2`: `16/90` IK failures

At `3 cm`:

- `seed 0`: `28/98` IK failures
- `seed 1`: `84/84` IK failures
- `seed 2`: `18/90` IK failures

So the distinctive failure is not that all seeds are bad. It is that `seed 1` is globally infeasible from the first target onward.

## Important execution-side note

The current Stage-2 `embodik` path in `_workspace/eval_dhb_head_embodik.py` is solving **position-only IK**:

- orientation tolerance is effectively disabled
- target orientation is set to the current TCP orientation for continuity

That means the current seed split is not being caused by quaternion targets in Stage 2. The saved predicted quaternions are not yet driving IK.

## Interpretation

The most likely reading is:

1. the StarVLA Stage-1 trajectory is still too aggressive, especially at the first waypoint
2. `seed 1` crosses the IK-feasibility boundary immediately
3. `seed 0` remains salvageable despite later failures
4. `seed 2` is partly reachable but still not task-correct

## Immediate next step

The next high-value change should target **Stage-1 feasibility**, not the execution stack:

- compare `seed 0` vs `seed 1` first-waypoint direction and magnitude during training/debugging
- add feasibility clamps or smoothing for early predicted waypoints
- retrain / re-evaluate with the same replanned `embodik` sweep

Separately, the harness server now fails fast if `use_embodik_ik=true`, because that path still advertises EEF-delta actions while returning joint targets.

## Clamp follow-up

A lightweight feasibility pass was added in `scripts/starvla_dhb_embodik_stage1.py`:

- `max-first-offset-mm = 40`
- `max-step-mm = 20`
- `smooth-window = 3`

Artifacts:

- Stage 1 summaries:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed0_replan6_clamp40_step20_sw3.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed1_replan6_clamp40_step20_sw3.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed2_replan6_clamp40_step20_sw3.json`
- IK debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_ik.json`
- Sim debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_sim.json`
- Videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan6_clamp40_step20_sw3_sweep`

### What improved

IK feasibility improved substantially:

- `offset 0cm`
  - `seed 0`: `25/98 -> 21/98` IK failures
  - `seed 1`: `84/84 -> 19/84` IK failures
  - `seed 2`: `16/90 -> 8/90` IK failures
- `offset 3cm`
  - `seed 0`: `28/98 -> 26/98`
  - `seed 1`: `84/84 -> 20/84`
  - `seed 2`: `18/90 -> 11/90`

This confirms the earlier diagnosis: a large part of the seed split was caused by oversized early translational jumps, especially on `seed 1`.

### What did not improve

Closed-loop task success did **not** increase:

- still `1/3` at `0 cm`
- still `1/3` at `3 cm`

The successful cases are unchanged:

- `seed 0`, `0 cm` succeeds in the default controller run
- `seed 0`, `3 cm` succeeds in the high-gain retry

### Updated interpretation

The clamp removes the worst gross infeasibility, but it does not solve the downstream manipulation error. So the current bottleneck has shifted again:

1. first-waypoint feasibility is now much better
2. execution is now meaningful for all seeds, especially `seed 1`
3. the remaining failure is task quality, not just immediate IK collapse

That means the next useful work should focus on trajectory quality near grasp / placement, not only on generic feasibility smoothing.

## Receding-horizon overlap blend check

A more VLA-like receding-horizon test was run for the previously problematic `seed 1`:

- `replan_every = 4`
- `max_first_offset_mm = 40`
- `max_step_mm = 20`
- `smooth_window = 3`
- `blend_overlap_alpha = 0.6`

Artifacts:

- Stage 1:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed1_replan4_clamp40_step20_sw3_blend06.json`
- IK:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_seed1_replan4_clamp40_step20_sw3_blend06_offset0_debug_ik.json`
- Sim:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_seed1_replan4_clamp40_step20_sw3_blend06_offset0_debug_sim.json`
- Video:
  - `_workspace/validation_videos/starvla_embodik_seed1_replan4_clamp40_step20_sw3_blend06_offset0/offset_0.0cm_seed_0.mp4`

### Why this is conceptually better

This matches the expected VLA chunk loop more closely:

1. predict a future chunk
2. execute only a short prefix
3. replan from the new observation
4. blend the overlapping future steps instead of hard-switching at chunk boundaries

This is analogous to action-chunk ensembling used in StarVLA's BEHAVIOR interface.

### Result

For this targeted run, the behavior improved enough to succeed:

- `seed 1`, `0 cm`
  - IK failures: `19/84 -> 17/84`
  - sim result: `1/1 = 100%`
  - first success at step `74`

### Updated interpretation

The apparent `go-stop-go-stop` pattern was not just controller noise. It was largely caused by:

- hard chunk-boundary switching
- repeated hold segments after IK failures

Using shorter replanning plus overlap blending makes the closed-loop behavior more VLA-like and can materially improve execution.
