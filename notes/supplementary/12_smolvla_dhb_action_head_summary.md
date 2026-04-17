# SmolVLA DHB Action Head Summary

## Scope

This note summarizes the SmolVLA-based DHB action head validation performed before moving the main effort to StarVLA. The goal was to determine whether the DHB action head plumbing, inference path, and `embodik` execution stack were sound, and whether additional SmolVLA training was likely to improve closed-loop performance.

## Setup

The validated pipeline was:

`SmolVLA VLM -> DHBActionHead -> DHB invariants -> decode -> embodik IK -> LIBERO JOINT_POSITION sim`

Key implementation changes during this phase:
- all SmolVLA-era inference was routed through `SmolVLADHBAdapter`
- regression tests were added to ensure current robot state is forwarded into the DHB head
- the `embodik` evaluator was updated to accept fresh checkpoints, emit debug JSON, and record videos
- the end-to-end trainer was updated to use `action_horizon = 100`
- the end-to-end trainer was later patched to support resume / warm-start while preserving the evaluation checkpoint format

## Main Findings

### 1. The original integration bug was fixed

The previous failure mode in which the DHB action head ignored the current robot state was removed. Adapter-based inference now forwards the full 8D state, and regression tests lock this behavior down.

### 2. Persistent rotation runaway was removed

The intermediate smoke checkpoint no longer exhibited the previous runaway end-effector rotation behavior. Predict-stage debug metrics stayed in a small range, and later checkpoints remained stable enough for consistent execution.

### 3. The `embodik` execution path is valid

Across smoke, 50-epoch, and 200-epoch checkpoints, the IK stage remained highly accurate:
- mean FK error around `0.47 mm`
- max FK error around `0.50 mm`
- similar failure counts across demos

This indicates that `embodik` IK is not the main bottleneck in the current pipeline.

### 4. Longer SmolVLA training improved fit but not closed-loop success

Three main checkpoints were evaluated:

| Checkpoint | Best loss | `0cm` | `3cm` |
|-----------|-----------|-------|-------|
| Smoke (`10ep`) | `0.94095` | `2/3` | `1/3` |
| Full (`50ep`) | `0.69913` | `1/3` | `1/3` |
| Longer (`200ep`) | `0.06337` | `1/3` | `1/3` |

Training loss improved dramatically from `10` to `200` epochs, but closed-loop success did not improve. This is the clearest evidence that additional optimization alone is not the right next step.

### 5. The current bottleneck is per-demo placement robustness

The multi-seed evaluation is better interpreted as a per-demo evaluation:
- `seed 0` -> `demo_0`
- `seed 1` -> `demo_1`
- `seed 2` -> `demo_10`

`demo_0` remained the most forgiving and reproducibly succeeded. `demo_1` and `demo_10` remained the failure cases across checkpoints.

The final diagnostic run on the smoke checkpoint showed that the dominant failure mode is:
- under-reaching in `Y` at final placement
- with a smaller secondary bias toward slightly high final `Z`
- not primarily an incorrect grasp/release schedule

Representative final end-effector offsets relative to the plate at `0cm`:
- `demo_0` success: `ee_minus_plate ~= [0.018, -0.020, 0.072]`
- `demo_1` failure: `ee_minus_plate ~= [0.001, -0.047, 0.075]`
- `demo_10` failure: `ee_minus_plate ~= [-0.006, -0.055, 0.073]`

These numbers support the qualitative video read: the failing demos finish too far behind the placement region in `Y`.

## Interpretation

The SmolVLA-based phase established four useful facts.

First, the DHB action head integration is now functionally correct. Second, the `embodik` evaluation path is stable enough to use as the primary execution benchmark. Third, train loss is not a reliable proxy for closed-loop success in this setting. Fourth, the remaining limitation is not the DHB representation itself but the policy's ability to place robustly across specific demonstrations.

Because the main remaining error is structured and demo-specific, additional SmolVLA scaling without an explicit modeling or supervision change is low expected value.

## Decision

The SmolVLA phase should be treated as complete for debugging and validation purposes.

Recommended next step:
- keep the SmolVLA smoke checkpoint as the executable baseline
- carry the `embodik` pipeline, video recording, and per-demo diagnostics forward
- shift the main effort to StarVLA

SmolVLA should only be revisited if there is a concrete change to test, such as:
- different supervision or checkpoint selection
- stronger position conditioning
- explicit placement-focused losses
- a different replanning or execution strategy

Further scaling to `500+` or `1000` epochs is not justified by the current evidence.

## Key Artifacts

Checkpoints:
- `_workspace/dhb_head_e2e_fix_smoke.pt`
- `_workspace/dhb_head_e2e_fix_full.pt`
- `_workspace/dhb_head_e2e_fix_200ep.pt`

Evaluation summaries:
- `_workspace/dhb_head_e2e_fix_smoke_sweep_predict.json`
- `_workspace/dhb_head_e2e_fix_smoke_sweep_ik.json`
- `_workspace/dhb_head_e2e_fix_smoke_sweep_sim.json`
- `_workspace/dhb_head_e2e_fix_full_sweep_sim.json`
- `_workspace/dhb_head_e2e_fix_200ep_sweep_sim.json`
- `_workspace/dhb_head_e2e_fix_smoke_diag_sim.json`

Videos:
- `_workspace/validation_videos/smoke_sweep/`
- `_workspace/validation_videos/full_sweep/`
- `_workspace/validation_videos/200ep_sweep/`
- `_workspace/validation_videos/smoke_diag/`
