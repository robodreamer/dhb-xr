# StarVLA + DHB-XR Gap Closure Plan

## Purpose

This note turns the current StarVLA + DHB-XR findings into a concrete execution
plan for closing the remaining gaps before broader benchmark scaling.

It is meant to answer:

1. What is already solid enough to keep?
2. What is still blocking a strong DHB-XR claim?
3. What is the shortest high-value sequence to close those gaps?

## Current Position

### What is already working

- The StarVLA `QwenDHB` path now trains with:
  - `state`
  - `dhb.linear_invariants`
  - `dhb.angular_invariants`
- The earlier major execution bugs are fixed:
  - wrong quaternion convention
  - stale normalization / absolute-vs-delta rollout mismatch
  - gripper threshold bug in the LIBERO converter
  - single-short-chunk Stage-1 bug
  - seed-handoff mismatch in the `embodik` pack path
- The most promising execution path is now:
  - `predict_trajectory()`
  - `embodik` IK
  - `JOINT_POSITION` rollout
- StarVLA now has nonzero closed-loop success through `embodik`.
- The closed-loop motion is substantially more continuous with short replanning
  plus overlap blending.

### What is already validated

- DHB invariants are actually present in the real StarVLA training loop.
- State conditioning is really used and matters.
- The direct `predict_action()` path is currently weaker than the DHB trajectory
  path for embodied execution.
- Representation-level SE(3) consistency claims are much stronger than rollout-
  level superiority claims at the current stage.

### What is still missing

- A single shared StarVLA closed-loop config that clearly matches or beats the
  SmolVLA smoke baseline.
- Robust success across seeds, especially at `3 cm`.
- Correct learned gripper behavior without leaning on demo timing.
- A matched regular-action-head baseline under the same evaluator.
- Harness / benchmark integration using the correct StarVLA->DHB->`embodik`
  action contract.

## Current Baselines

### SmolVLA executable baseline to beat

- `0 cm`: `2/3`
- `3 cm`: `1/3`
- evaluator: `_workspace/eval_dhb_head_embodik.py`
- current best checkpoint: `_workspace/dhb_head_e2e_fix_smoke.pt`

### Current StarVLA status

- direct raw LIBERO controller rollout:
  - useful as a smoke test only
  - not the main DHB evaluation path
- StarVLA -> `embodik`:
  - now the main path
  - best shared configs still around `1/3`
  - targeted seed-specific tuning can do better than the shared baseline

## Remaining Gaps

| Gap | Current evidence | Why it matters |
|------|------------------|----------------|
| Canonical loop not frozen | `replan4` and `replan6` each help different seeds | We still cannot say which exact closed-loop policy is the baseline |
| Seed robustness | `seed 0` and `seed 1` prefer different settings; `seed 2` still misses completion | We need a shared configuration, not hand-tuned per-seed wins |
| Gripper behavior | demo timing still acts as a crutch | We cannot claim the action head handles full manipulation cleanly yet |
| Phase quality | grasp / placement logic is still fragile | Current failures are no longer just feasibility failures |
| Matched baseline missing | no clean `QwenDHB` vs regular-head comparison yet | We cannot honestly claim superiority over standard action heads |
| Harness contract mismatch | harness `use_embodik_ik=true` was invalid and now fails fast | Benchmark scale-out is not ready until the action contract is explicit |

## Strategy

The right order is:

1. Freeze the correct closed-loop semantics.
2. Improve manipulation quality under that fixed loop.
3. Run a matched baseline comparison.
4. Only then scale to broader benchmark tools.

This avoids repeating the SmolVLA pattern where more training improved loss but
did not improve execution.

## Phase 1: Freeze The Canonical Closed-Loop

### Goal

Pick one StarVLA -> DHB -> `embodik` loop that is:

- semantically correct
- smooth enough to inspect
- shared across seeds
- good enough to serve as the official baseline for further tuning

### Canonical path

Use this as the default architecture:

```text
Qwen VL hidden states
-> DHB invariant prediction
-> decoded EE waypoints
-> embodik IK
-> JOINT_POSITION execution
-> short-horizon replanning
```

Treat raw direct LIBERO controller rollout as smoke-only.

### Work items

- Run one bounded calibration sweep over the current best loop knobs:
  - `replan_every in {4, 6}`
  - `max_first_offset_mm in {35, 40}`
  - `max_step_mm = 20`
  - `smooth_window = 3`
  - `blend_overlap_alpha = 0.6`
- Keep video recording on for every candidate.
- Compare each candidate on:
  - success at `0 cm` and `3 cm`
  - IK failure rate
  - visible motion continuity
  - close / reopen timing
- Choose one shared config and freeze it in the notes and scripts.

### Exit criteria

- One shared config is selected as the StarVLA baseline.
- That config has:
  - no 180-degree EE snap
  - no gross stop-go artifacts
  - nonzero success at both `0 cm` and `3 cm`

## Phase 2: Improve Manipulation Quality Under The Frozen Loop

### Goal

Move from “trajectory is feasible” to “trajectory performs the task.”

### Focus

The current bottleneck is no longer the old orientation bug. It is:

- late-phase grasp / placement quality
- seed-to-seed trajectory robustness
- gripper semantics

### Work items

#### 2A. Close / release phase logic

- Add bowl-relative close gating.
- Keep release gating over the plate with height checks.
- Add a short explicit placement tail where needed:
  - descend
  - release
  - small lift

#### 2B. Keep gripper evaluation honest

- Use demo gripper timing as the control branch while trajectory quality is
  still being tuned.
- In parallel, fix the learned gripper branch separately instead of conflating
  the two problems.

#### 2C. Tighten seed robustness

- Compare `seed 0`, `seed 1`, and `seed 2` on:
  - first close pose
  - first reopen pose
  - final `ee_minus_plate`
  - final `ee_minus_bowl`
- Target `seed 2` as the “almost there” improvement case after the frozen-loop
  baseline is chosen.

### Exit criteria

- Shared StarVLA config matches the SmolVLA smoke baseline:
  - at least `2/3` at `0 cm`
  - at least `1/3` at `3 cm`
- Stretch goal:
  - `2/3` at `3 cm`

## Phase 3: Close The Scientific Comparison Gap

### Goal

Establish what can honestly be claimed over a regular action head.

### Required comparison

Run a matched baseline where only the action-head design changes.

Preferred comparison:

- `QwenDHB`
- `QwenOFT` or equivalent regular/direct action head

under the same:

- dataset
- frozen-Qwen backbone
- evaluator
- offsets
- seeds
- video protocol

### Metrics to log

- train losses:
  - action loss
  - invariant loss
- prediction quality:
  - chunk MAE
  - first-waypoint error
- execution quality:
  - IK failure rate
  - first close pose
  - final `ee_minus_plate`
  - success rate

### Claims this phase should answer

- Does DHB reduce IK infeasibility?
- Does DHB improve final placement geometry?
- Does DHB improve closed-loop success under the same execution stack?

### Exit criteria

- A matched baseline table exists with videos.
- We can say one of the following honestly:
  - DHB beats the regular head on success
  - DHB matches success but improves geometry / feasibility
  - DHB is still not better and needs more work

## Phase 4: Scale To Benchmark Infrastructure

### Goal

Only after the canonical StarVLA baseline is stable, connect it to the broader
benchmark stack.

### Work items

- Fix the harness action contract so `embodik` execution is represented
  honestly.
- Add a StarVLA model-server path that explicitly uses:
  - `predict_trajectory()`
  - `embodik`
  - the correct downstream action contract
- Re-run harness LIBERO smoke first.
- Only then broaden to:
  - VLA-Arena smoke tasks
  - RoboCasa smoke tasks

### Exit criteria

- Harness smoke uses the same canonical StarVLA->DHB->`embodik` path as the
  manual validation scripts.
- At least one smoke benchmark outside the custom script path runs with
  recorded video and matching semantics.

## Phase 5: Package The Evidence

### Goal

Convert the work into reusable evidence for internal reviews, the project page,
and the paper.

### Work items

- Curate a small video set:
  - one best `0 cm` StarVLA success
  - one best `3 cm` StarVLA success
  - one instructive failure contrast
- Create one comparison table:
  - SmolVLA smoke baseline
  - StarVLA canonical baseline
  - matched regular-head baseline
- Update the writeup with:
  - what is proven
  - what remains only suggestive

### Exit criteria

- We can present the DHB story without overclaiming:
  - representation-level consistency is strong
  - embodied gains are promising
  - matched head-to-head evidence is explicit

## Phase 6: Realize The Stronger Scaling Story

Reference note:
- `notes/supplementary/17_dhbxr_scaling_story_note.md`

### Goal

Move from:

- "DHB-XR works as a better structured action head"

to:

- "DHB-XR is the right interface for scaling VLAs from small real datasets via
  simulator or world-model augmentation."

### Why this is a separate phase

This is the larger intended story, but it should only be attempted after the
canonical StarVLA baseline and matched regular-head comparison are already
stable.

### Program A: Matched action-space ordering study

Explicitly test the intended ordering:

- `DHB-XR`
- `relative`
- `delta`
- `absolute`

using:

- representation consistency
- compression
- matched embodied evaluation

### Program B: Small-seed-data study

Train from very small teleop seed sets, for example:

- `N = 1`
- `N = 3`
- `N = 5`
- `N = 10`

and compare:

- DHB-XR action head
- regular action head

to test whether DHB-XR is more data-efficient.

### Program C: Augmentation study

Use a hybrid data strategy:

1. small real seed dataset
2. simulator or world-model-generated observation diversity
3. DHB-XR-adapted trajectories paired to those observations

Compare against non-DHB augmentation baselines.

### Program D: Cross-embodiment study

Test whether the same invariant representation supports better reuse across:

- embodiment shifts
- initial pose shifts
- scene-layout changes

### Exit criteria

The full stronger story is ready only if we can support at least one of these
with clear evidence:

- less real data for equal performance
- better performance with the same small real dataset
- stronger offset / embodiment robustness after augmentation

## Things Not To Spend Time On Right Now

- Do not switch back to the heavy Cosmos/WM4A path for iterative debugging.
- Do not scale training epochs alone before the canonical loop is frozen.
- Do not use raw direct LIBERO controller rollout as the primary DHB metric.
- Do not claim superiority over regular action heads until the matched baseline
  exists.

## Immediate Execution Order

1. Freeze one canonical StarVLA->`embodik` closed-loop config from the bounded
   calibration sweep.
2. Improve phase-specific behavior under that frozen loop:
   - close gate
   - release gate
   - placement tail
3. Match or beat the SmolVLA smoke baseline under one shared config.
4. Run the matched regular-head baseline.
5. Only then wire the canonical path into the harness and broader benchmarks.

## Decision Rule

If Phase 2 cannot match the SmolVLA smoke baseline under one shared config,
then do not broaden benchmarks yet. At that point the next effort should go
back into model/training changes, not more evaluation plumbing.
