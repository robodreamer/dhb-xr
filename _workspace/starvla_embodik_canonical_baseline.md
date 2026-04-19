# StarVLA Embodik Canonical Baseline Selection

Date: 2026-04-19

## Purpose

This note records the bounded calibration sweep used to freeze the canonical
StarVLA -> DHB -> `embodik` closed-loop configuration.

The goal was not to maximize one hand-tuned seed, but to choose one shared
setting that is:

- conceptually correct for a VLA chunked loop
- visibly continuous
- strong enough to carry forward as the default StarVLA baseline

## Candidates Compared

All candidates used:

- `max_step_mm = 20`
- `smooth_window = 3`
- `blend_overlap_alpha = 0.6`
- demo gripper timing
- current release semantics in `_workspace/eval_dhb_head_embodik.py`

The bounded sweep was:

1. `replan_every = 4`, `max_first_offset_mm = 35`
2. `replan_every = 4`, `max_first_offset_mm = 40`
3. `replan_every = 6`, `max_first_offset_mm = 35`
4. `replan_every = 6`, `max_first_offset_mm = 40`

## Artifacts

### `replan4 + clamp35`

- IK:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_fullsweep_ik.json`
- sim:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_fullsweep_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan4_clamp35_step20_sw3_blend06_fullsweep`

### `replan4 + clamp40`

- IK:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp40_step20_sw3_blend06_currentsim_ik.json`
- sim:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp40_step20_sw3_blend06_currentsim_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan4_clamp40_step20_sw3_blend06_currentsim`

### `replan6 + clamp35`

- IK:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp35_step20_sw3_blend06_fullsweep_ik.json`
- sim:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp35_step20_sw3_blend06_fullsweep_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan6_clamp35_step20_sw3_blend06_fullsweep`

### `replan6 + clamp40`

- IK:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_ik.json`
- sim:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan6_clamp40_step20_sw3_sweep`

## Comparison Summary

All four candidates reached the same coarse success summary:

- `0 cm`: `1/3`
- `3 cm`: `1/3`

So the selection had to be based on:

- motion continuity
- which seed carried the success
- how aggressive the loop was
- whether the setting still looked stable at `3 cm`

### `replan4` vs `replan6`

`replan4` is the better shared VLA-style loop.

Why:

- it is visibly smoother and less stop-go
- it follows the expected short-horizon receding-horizon pattern better
- it succeeds on `seed 1` at both `0 cm` and `3 cm`
- it avoids leaning on a high-gain retry for the `3 cm` success

`replan6` is still useful diagnostically, especially for `seed 0`, but it is
more pulse-like and less suitable as the canonical shared baseline.

### `clamp35` vs `clamp40`

`clamp35` is the better default clamp.

Why:

- it preserves the same gross success rate as `clamp40`
- it is less aggressive
- it avoids the large `seed 2`, `3 cm` overshoot seen with `replan4 + clamp40`
  where final `ee_minus_plate ~= [x=?, y=+0.035, z=+0.127]`
- it is the cleaner default if we want the baseline to favor stability over
  larger first-step freedom

## Selected Canonical Baseline

Use this as the default StarVLA -> DHB -> `embodik` closed-loop baseline:

- `replan_every = 4`
- `max_first_offset_mm = 35`
- `max_step_mm = 20`
- `smooth_window = 3`
- `blend_overlap_alpha = 0.6`
- demo gripper timing for now
- current release semantics from `_workspace/eval_dhb_head_embodik.py`

## What This Means

The loop-search phase is done.

The next bottleneck is no longer:

- gross orientation bugs
- hard chunk-boundary discontinuity
- loop cadence ambiguity

The next bottleneck is:

- grasp / placement phase quality under this frozen loop
- especially how to lift `seed 2`
- and how to replace demo gripper timing with a learned gripper branch later

## Immediate Next Step

Keep this canonical baseline fixed and improve:

1. bowl-relative close gating
2. plate-relative release behavior
3. an explicit placement tail if needed

Do not reopen the loop-search unless a later model change invalidates the
current baseline.

## Follow-up After Freezing

The next two low-cost execution-side follow-ups were tested against this
canonical baseline:

1. add bowl-relative close gating in the Stage 3 sim loop
2. add one minimal late placement tail that reuses the lowest recent
   plate-near joint target before opening

Artifacts:

- bowl close gate:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_geomgates_sim.json`
- bowl close gate + minimal placement tail:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_geomgates_tail2_sim.json`

Outcome:

- both still land at:
  - `0 cm`: `1/3`
  - `3 cm`: `1/3`
- the close gate is still worth keeping because it delays obviously bad early
  closes, especially for `seed 2`
- the minimal placement tail can change timing for an already-working seed, but
  did not create a new shared success

Conclusion:

- the frozen canonical loop remains the right baseline
- but small sim-side phase heuristics are now close to exhausted
- the next meaningful lift should come from a stronger phase-aware training or
  supervision change, not more loop-side knob tuning
