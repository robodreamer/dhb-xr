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
- A reach-only benchmark now cleanly isolates bowl approach from gripper timing.
- A phase-split execution bridge now recovers the integrated baseline after the
  globally committed-handoff mode regressed it.

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
- A model-side continuation mechanism that knows which chunk prefix is already
  committed / executed.

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
2. Improve bowl approach under that fixed loop.
3. Reintroduce integrated manipulation on top of the improved reach baseline.
4. Run a matched baseline comparison.
5. Only then scale to broader benchmark tools.

### 2026-04-21 update: `vla_foundry` and `TRI-LBM` point to the same missing model-side cue

Two external repos were checked against the current StarVLA + DHB plateau:

- `TRI-ML/vla_foundry`
- `lucidrains/TRI-LBM`

The strongest new clue comes from `vla_foundry`:

- their diffusion policy explicitly distinguishes committed / past actions from
  future actions with `past_mask` / `future_mask`
- their robotics inference stack uses a formal adapter for observation history,
  action history, and reference-relative conversion instead of spreading those
  assumptions across rollout scripts

This matches the current local evidence:

- execution-side committed handoff improves reach-only bowl approach
- but the current `QwenDHB` model does not know which prefix is already
  committed, so integrated performance does not improve with that execution hack
  alone

`TRI-LBM` adds a second useful clue:

- action generation can benefit from a lightweight auxiliary status signal
  predicted jointly with the action chunk

In that repo the status signal is a generic `task_status`, but for this project
the more relevant analogue is:

- reach / near-contact / transport / placement phase
- or, more minimally, a continuation / progress status for the current chunk

What does **not** look like the next lever from these external repos:

- framework replacement
- generic longer training
- direct-action-only evaluation simplification
- typed action normalization as the primary fix

The updated ranking is therefore:

1. add DHB-specific continuation conditioning with a committed-prefix mask
2. keep the reach-only benchmark as the main short-horizon target
3. keep learned `gripper_state` as the primary gripper output and treat
   event/phase as auxiliary
4. stabilize learned gripper inference with adapter-level hysteresis instead of
   demo timing
5. once reach improves, return to the integrated pipeline with the phase-split
   bridge
6. only then add a lightweight auxiliary phase/status head if integrated
   manipulation still stalls

### 2026-04-21 update: committed-prefix continuation path is now in code

The external-repo-inspired continuation step is no longer just a plan item.
The local DHB path now supports:

- committed invariant prefix conditioning
- suffix-only loss masking
- optional continuation-prefix inputs at inference time

The first experiment config is:

- `configs/libero_dhbxr_qwen_continuation.yaml`

Initial intended run policy:

- warm-start from the old best reach checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- keep the current corrected dataset and frozen-Qwen setup
- benchmark only with the reach-only live sweep first

This should be treated as the main model-side breakthrough attempt before
reopening integrated gripper/manipulation work.

### 2026-04-21 update: first continuation-conditioned warm-start is in flight

The continuation-conditioned path is now implemented end to end:

- `src/dhb_xr/integration/vla/action_head.py`
- `vendor/starVLA/starVLA/model/modules/action_model/DHBActionHeader.py`
- `vendor/starVLA/starVLA/model/framework/VLM4A/QwenDHB.py`
- `configs/libero_dhbxr_qwen_continuation.yaml`

What changed:

- the DHB head can condition on a committed invariant prefix
- training can mask the committed prefix out of the loss so only the suffix is
  supervised
- inference can optionally receive the committed prefix explicitly

The first real run is:

- run dir:
  `_workspace/starvla_runs/dhb_starvla_qwen_continuation_short1`
- warm-start source:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- training budget:
  `200` steps

Current interpretation:

- this is the first model-side attempt that actually matches the
  `vla_foundry`-style past-vs-future conditioning clue
- it is therefore more meaningful than:
  - another same-family warm-start
  - another clamp / blend sweep
  - another gripper-only tweak

Final reach-only live-sweep result:

- `steps_100`:
  - `avg_min_bowl_xy_m = 0.2207`
  - `avg_min_bowl_xyz_m = 0.3551`
  - `avg_ik_fail_total = 13.5`
- `steps_200`:
  - `0.2320 / 0.3578 / 21.0`
- `steps_50`:
  - `0.2498 / 0.3739 / 78.0`
- `steps_150`:
  - `0.2502 / 0.3740 / 81.0`

Comparison to the current reach-only control:

- best existing control:
  - `avg_min_bowl_xy_m = 0.1375`

Interpretation:

- continuation conditioning did not beat the current reach-only control
- it did reduce IK failures sharply at the best checkpoint (`steps_100`), which
  means the idea is not useless
- but the main failure is still bowl approach quality, not just feasibility

So the next most impactful technical step now shifts to:

1. a more centralized adapter-style benchmark contract for:
   - history buffering
   - reference handling
   - gripper-state debouncing
2. only then reopen more ambitious DHB-specific continuation / phase work if
   the control comparison still shows a clear DHB-side opportunity

### 2026-04-19 update: first live-robustness augmentation result

A first training-side robustness check was added after the live checkpoint sweep
showed that more warm-start training on the same objective does not improve the
online baseline. The intervention was intentionally minimal:

- keep the canonical `StarVLA -> DHB -> embodik` loop fixed
- add small, consistent state-input drift during training
- warm-start from the old live-best checkpoint

Result:

- the best augmented checkpoint (`liveaug1/steps_300`) slightly reduced average
  IK failures
- but it still did not beat the old live-best checkpoint on bowl proximity
- it also produced `0` closes and `0` live successes

Interpretation:

- the main diagnosis is unchanged: the gap is still online robustness
- but simple state-noise augmentation is too weak on its own
- the next training-side iteration should therefore be more physically grounded,
  such as rollout-state perturbation or stronger observation/state augmentation,
  rather than just more warm-start training on the same objective

This avoids repeating the SmolVLA pattern where more training improved loss but
did not improve execution.

### 2026-04-19 update: low-risk phase-aware training knobs are now wired

Two low-risk training-side changes have now been implemented and smoke-tested:

- terminal-step weighting in the DHB invariant loss
- end-anchored chunk oversampling in the LeRobot mixture loader

Files:

- `src/dhb_xr/integration/vla/action_head.py`
- `vendor/starVLA/starVLA/model/modules/action_model/DHBActionHeader.py`
- `vendor/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py`
- `configs/libero_dhbxr_qwen_phaseaug.yaml`

Sanity checks:

- the same invariant error placed at the last timestep now gets about `2.5x`
  the loss of the same error at the first timestep
- with `terminal_chunk_prob=1.0`, sampled training steps snap to the last valid
  chunk of each trajectory as intended
- the new config passes a real `QwenDHB` one-step train smoke on the laptop GPU

Interpretation:

- these are ready to use in the next training run
- they are better aligned with the current failure mode than another round of
  generic state-noise warm-starting

### 2026-04-19 update: phase-aware warm-start still does not beat the old live baseline

The first real run using those knobs is now complete:

- run dir:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1`
- warm-start source:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- budget:
  `300` steps

Live-sweep best checkpoint:

- `steps_200`
  - `avg_min_bowl_xyz_m = 0.3182`
  - `avg_min_bowl_xy_m = 0.1894`
  - `avg_ik_fail_total = 75.5`

Comparison:

- old live best (`shorttrain1/steps_200`):
  - `0.2916 / 0.1431 / 75.5`
- state-noise warm-start (`liveaug1/steps_300`):
  - `0.3245 / 0.1986 / 73.5`
- phase-aware warm-start (`phaseaug1/steps_200`):
  - `0.3182 / 0.1894 / 75.5`

Interpretation:

- phase-aware weighting + terminal oversampling is directionally better than
  the earlier state-noise-only warm-start
- but it still does not recover the original live baseline on bowl proximity
- this reinforces the current ranking:
  - keep these knobs available
  - stop expecting same-objective warm-starting to close the live gap
  - prioritize rollout-grounded corrective augmentation or matched execution
    controls next

### 2026-04-20 update: two reach-focused follow-ups still do not beat the original control

Two more reach-only interventions are now complete:

1. `reachcorr`
   - rollout-grounded corrective rewrite of state + action + invariants
   - best checkpoint: `steps_200`
   - live reach-only result:
     - `avg_min_bowl_xy_m = 0.1736`
     - `avg_min_bowl_xyz_m = 0.3331`
     - `avg_ik_fail_total = 76.5`
2. `reachphase`
   - no synthetic target rewrite
   - oversample real approach chunks so the close phase lands near the end of
     the horizon
   - best checkpoint: `steps_100`
   - live reach-only result:
     - `avg_min_bowl_xy_m = 0.1913`
     - `avg_min_bowl_xyz_m = 0.3289`
     - `avg_ik_fail_total = 75.5`

Reference control remains:

- `shorttrain1/steps_200`
  - `avg_min_bowl_xy_m = 0.1465`
  - `avg_min_bowl_xyz_m = 0.2954`
  - `avg_ik_fail_total = 75.5`

Interpretation:

- the original `shorttrain1/steps_200` checkpoint is still the reach-only
  control to beat
- same-family warm-start training is no longer the highest-confidence path
- the next promising move should be more structural than small data
  reweighting, specifically:
  - future-state / committed-tail chunk handoff, or
  - history-aware continuation conditioning

This is now the most credible next branch before returning to the full
gripper-integrated task.

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

Current status:

- bowl-relative close gating has been tested under the frozen canonical
  `replan4 + clamp35 + blend06` loop
- one minimal placement-tail heuristic has also been tested
- both preserve cleaner phase semantics, but neither lifts the shared baseline
  above `1/3` at `0 cm` / `3 cm`

Interpretation:

- this confirms the bottleneck is no longer another small execution bug
- one more tiny sim-side heuristic is unlikely to be the highest-value move
- the next lift should come from a stronger phase-aware model / supervision
  change or rollout-grounded robustness intervention

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

### Current status

An initial matched regular-head control now exists:

- config: `configs/libero_qwenoft_qwen.yaml`
- same base VLM: `Qwen2.5-VL-3B-Instruct`
- same corrected dataset: `libero_spatial_dhbxr`
- same frozen-Qwen short-train budget: `200` steps

Tracked note:

- `_workspace/starvla_qwenoft_control_baseline.md`

What it already shows:

- `QwenOFT` is now runnable as a fair StarVLA control under the corrected setup
- on one matched sample, its chunk MAE is worse than `QwenDHB`
- both direct-action rollouts still fail on the reference LIBERO task

2026-04-21 update: the execution-level control is now done too.

`QwenOFT` has now been run through the same downstream evaluator by:

- unnormalizing the predicted delta-action chunk
- integrating that chunk into EE waypoint poses
- feeding those waypoints into the same `embodik` Stage 2/3 stack

Bridge files:

- `scripts/starvla_oft_embodik_stage1.py`
- `src/dhb_xr/integration/vla/libero.py`

Matched result under the same offline `0 cm / 3 cm / 3-seed` evaluator:

| model | path | `0 cm` | `3 cm` |
| --- | --- | ---: | ---: |
| `QwenDHB` | canonical `predict_trajectory() -> embodik` | `1/3` | `1/3` |
| `QwenOFT` | integrated delta-action chunk -> `embodik` | `0/3` | `0/3` |

Important detail:

- the `QwenOFT` control was not blocked by IK
- all `6/6` cases were IK-feasible with mean FK error around `0.48 mm`

So this is now a real matched execution-side result, not only a direct-rollout
comparison.

### Claims this phase should answer

- Does DHB reduce IK infeasibility?
- Does DHB improve final placement geometry?
- Does DHB improve closed-loop success under the same execution stack?

Current answer:

- yes, DHB currently improves closed-loop success over the matched `QwenOFT`
  control under the same offline `embodik` stack
- no, this is still not enough to claim broad superiority yet, because:
  - the margin is still small (`1/3` vs `0/3`)
  - it is not yet tested in a live / online matched comparison

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

## 2026-04-20 Execution Update

Recent evidence narrows the gap-closure order further:

- The best reach-only control is still the old checkpoint plus the committed
  handoff:
  - `shorttrain1/steps_200`
  - `ramp + commit2`
  - `avg_min_bowl_xy_m = 0.1375`
- A fresh corrected-dataset run with `state_history_window = 4` did **not**
  beat that baseline:
  - best fresh checkpoint:
    - `0.2381 / 0.3681`
- A phase-split execution loop is worth keeping as the last execution-side
  bridge:
  - committed handoff for pre-grasp approach
  - canonical chunking after the close phase
  - this recovers the integrated `1/3, 1/3` baseline after the global
    handoff had regressed it to `0/3, 0/3`

So the next model-side step should now shift from:

- plain fresh retraining

to:

- continuation conditioning
- committed-prefix-aware chunk prediction
- or short action-history / prefix-conditioned continuation

In other words:

- execution-side tuning is close to exhausted
- same-family warm-starts are not the lever
- the next real breakthrough is more likely to come from teaching the policy
  how to continue a partially committed trajectory than from more clamp/blend
  or more plain state-history retraining

## 2026-04-21 Gripper Path Update

Two practical decisions are now clearer:

1. The reusable rollout modules should stay object-agnostic.
   - We kept shared `gripper_state` hysteresis / temporal debounce in the
     general path.
   - We did **not** move bowl / plate geometry vetoes into the shared module.
     Those remain evaluator-specific LIBERO scaffolding.

2. The correct learned gripper hierarchy is:
   - primary output: dense `gripper_state`
   - auxiliary signals: sparse `gripper_event`, dense `gripper_phase`

Focused evidence on `dhb_starvla_qwen_gripper_multitask4/steps_200`:

- With `gripper_transition_debounce_steps=2`, the stage-1 preview for the
  reference episode begins with the first `12` decoded commands all open.
- The corresponding live no-gate case still fails, but early chatter is
  reduced:
  - previous: `close_step=4`, `reopen_step=8`
  - debounced: `close_step=5`, `reopen_step=13`
- Bowl approach does not improve:
  - `min_bowl_xy_m ≈ 0.2097`
  - `ik_fail_total = 82`

So the debounce path is worth keeping as benchmark infrastructure, but it does
not close the remaining manipulation gap on its own. The next real gain still
has to come from better phase alignment and better bowl approach, not from more
decode-only gripper smoothing.
