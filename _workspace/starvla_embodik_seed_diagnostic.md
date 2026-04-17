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
