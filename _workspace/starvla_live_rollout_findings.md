# StarVLA DHB Live Rollout Findings

Date: 2026-04-19

## Goal

Replace the partially offline `StarVLA -> DHB -> embodik` Stage-1 path with a true
live-observation rollout loop:

1. reset LIBERO env
2. observe live images and EE state
3. run `model.predict_trajectory()` on the live observation
4. solve IK online for the next chunk
5. execute only the prefix
6. replan from the new live observation

This is the correct structural next step beyond the demo-conditioned replanning path.

## New Script

- [scripts/starvla_dhb_embodik_live_rollout.py](/home/andypark/Projects/repos/dhb-xr/scripts/starvla_dhb_embodik_live_rollout.py)

Key properties:

- live LIBERO observation, not demo frames
- same quantized `QwenDHB` checkpoint path
- same `embodik` IK backend
- same JOINT_POSITION stage-3 execution contract as
  [_workspace/eval_dhb_head_embodik.py](/home/andypark/Projects/repos/dhb-xr/_workspace/eval_dhb_head_embodik.py)
- same clamp / smoothing / overlap-blending ideas as the offline Stage-1 path

## Main Results

### Follow-up live cleanup: demo init-state + geometry-conditioned gripper

The live script was then tightened further to reduce two remaining interpretation
caveats:

- use the HDF5 demo's saved `init_state` by default
- gate close on bowl-relative geometry and release on plate-relative geometry

Representative rerun:

- summary:
  [live_rollout_ep0_offset0_geomgrip_summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_geomgrip_summary.json)

Important observations:

- `init_state_l2_diff ≈ 0.357` between the benchmark-provided init state and the
  HDF5 demo init state, so this alignment fix was necessary
- even after switching to the demo init state, the run still fails
- with bowl-relative close gating enabled, the gripper never closes at all
  (`close_step = null`)
- final EE remains far from the bowl:
  - `ee_minus_bowl ≈ [0.145, -0.119, 0.242]`

Interpretation:

- the init-state mismatch was real
- but fixing it does not recover online success
- the stronger conclusion is now that the live trajectory does not enter a valid
  grasp region, rather than “the arm closes at the wrong place”

### Live checkpoint sweep: online-feasibility ranking

To avoid selecting checkpoints only by train loss, the live rollout script now
supports no-video evaluation plus a compact feasibility summary, and
[scripts/starvla_dhb_live_checkpoint_sweep.py](/home/andypark/Projects/repos/dhb-xr/scripts/starvla_dhb_live_checkpoint_sweep.py)
aggregates those results across representative live cases.

Sweep artifacts:

- markdown summary:
  [_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md)
- json summary:
  [_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.json)

Representative cases:

- `episode_idx=0`, `offset=0 cm`
- `episode_idx=1`, `offset=3 cm`

Ranking:

| Rank | Step | Successes | Closes | Avg min bowl XYZ (m) | Avg min bowl XY (m) | Avg IK fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200 | 0 | 0 | 0.2916 | 0.1431 | 75.5 |
| 2 | 100 | 0 | 0 | 0.3672 | 0.2280 | 76.0 |
| 3 | 50 | 0 | 0 | 0.3739 | 0.2500 | 80.5 |
| 4 | 150 | 0 | 0 | 0.3740 | 0.2500 | 79.5 |

Interpretation:

- none of the current checkpoints are yet good enough for true live success
- none reach a valid live close event under the stricter bowl-relative close gate
- but the live metrics are still informative and clearly favor `steps_200`
- `steps_200` should therefore be treated as the current online candidate
  checkpoint until a stronger live metric or checkpoint overtakes it

### First live-robustness training check: state-input drift augmentation

To test the main diagnosis without changing the frozen execution loop, a first
training-side robustness experiment was added:

- config:
  [configs/libero_dhbxr_qwen_liveaug.yaml](/home/andypark/Projects/repos/dhb-xr/configs/libero_dhbxr_qwen_liveaug.yaml)
- dataset patch:
  [vendor/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py](/home/andypark/Projects/repos/dhb-xr/vendor/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py)

This experiment applies a **small, consistent perturbation to the state input**
during training:

- `state_input_noise_prob = 0.7`
- `state_input_noise_pos_std_m = 0.01`
- `state_input_noise_rot_std_rad = 0.05`

The target trajectory is unchanged. This is intentionally only a light
robustness augmentation, meant to reduce brittleness to small executed-state
drift rather than simulate full online rollouts.

Short warm-start run:

- run dir:
  [_workspace/starvla_runs/dhb_starvla_qwen_liveaug1](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_liveaug1)
- warm-start checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- train budget: `300` steps

Live sweep artifacts:

- markdown summary:
  [_workspace/starvla_runs/dhb_starvla_qwen_liveaug1/live_checkpoint_sweep/summary.md](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_liveaug1/live_checkpoint_sweep/summary.md)

Ranking:

| Rank | Step | Successes | Closes | Avg min bowl XYZ (m) | Avg min bowl XY (m) | Avg IK fails |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 300 | 0 | 0 | 0.3245 | 0.1986 | 73.5 |
| 2 | 100 | 0 | 0 | 0.3265 | 0.1975 | 75.5 |
| 3 | 200 | 0 | 0 | 0.3360 | 0.1966 | 76.0 |

Interpretation:

- the light state-input perturbation did **not** beat the old online candidate
  (`steps_200` from `dhb_starvla_qwen_invsup_shorttrain1`, which was
  `0.2916 / 0.1431 / 75.5`)
- it did slightly improve IK-feasibility relative to the old baseline
  (`73.5` avg IK fails vs `75.5`)
- but bowl proximity remained clearly worse
- so this first-pass augmentation helps smoothness / feasibility more than
  actual grasp-region targeting

Conclusion:

- the central diagnosis still holds: online robustness is the main gap
- but this lightweight state-noise augmentation alone is **not** enough to close
  it
- the next training-side change should be more physically grounded than simple
  state-input noise, e.g. rollout-state perturbation, stronger observation/state
  augmentation, or phase-aware supervision

### Phase-aware warm-start follow-up

- config:
  `configs/libero_dhbxr_qwen_phaseaug.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1`
- sweep summary:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1/live_checkpoint_sweep/summary.md`

Result:

- best phase-aware checkpoint: `steps_200`
  - `avg_min_bowl_xyz_m = 0.3182`
  - `avg_min_bowl_xy_m = 0.1894`
  - `avg_ik_fail_total = 75.5`

Comparison:

- old live best: `shorttrain1/steps_200`
  - `0.2916 / 0.1431 / 75.5`
- first state-noise warm-start: `liveaug1/steps_300`
  - `0.3245 / 0.1986 / 73.5`
- new phase-aware warm-start: `phaseaug1/steps_200`
  - `0.3182 / 0.1894 / 75.5`

Interpretation:

- terminal-step weighting plus end-anchored chunk oversampling is better than
  `liveaug1` on bowl proximity
- but it still does **not** beat the original live baseline from
  `shorttrain1/steps_200`
- `steps_100` is notable for low IK failures (`31.5`) but clearly worse bowl
  approach, which reinforces that better reachability alone is not the same as
  better task geometry

Conclusion:

- these low-risk phase-aware knobs are worth keeping available
- but they are not sufficient to close the live replanning gap by themselves
- the next training-side change should be more rollout-grounded than another
  same-objective warm-start

### Canonical live test: `episode_idx=1`, `offset=3 cm`

Command shape:

```bash
PYTHONPATH=vendor/starVLA WANDB_MODE=disabled MUJOCO_GL=egl \
./.pixi-home/bin/pixi run python scripts/starvla_dhb_embodik_live_rollout.py \
  --config_yaml configs/libero_dhbxr_qwen.yaml \
  --checkpoint _workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --episode_idx 1 \
  --offset-cm 3 \
  --replan_every 4 \
  --max-first-offset-mm 35 \
  --max-step-mm 20 \
  --smooth-window 3 \
  --blend-overlap-alpha 0.6 \
  --gripper-source demo
```

Artifacts:

- summary:
  [live_rollout_ep1_offset3_summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep1_offset3_summary.json)
- video:
  [live_offset_3.0cm_ep_1_replan4_clamp35.mp4](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen/live_rollout_videos/live_offset_3.0cm_ep_1_replan4_clamp35.mp4)

Result:

- `success = false`
- `close_step = 42`
- `reopen_step = 80`
- total IK failures across chunks: `68`
- first failure chunk index: `4` (`t_start=16`)

Interpretation:

- the first `4` replans are online-feasible
- after that, the executed state drifts far enough from the demo manifold that the
  next predicted chunks become increasingly bad
- the loop then degrades into repeated IK holds with almost no meaningful motion

### Easier live test: `episode_idx=0`, `offset=0 cm`

Artifacts:

- summary:
  [live_rollout_ep0_offset0_summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_summary.json)
- video:
  [live_offset_0.0cm_ep_0_replan4_clamp35.mp4](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen/live_rollout_videos/live_offset_0.0cm_ep_0_replan4_clamp35.mp4)

Result:

- also `success = false`
- same qualitative failure mode as the `3 cm` case
- the arm initially moves plausibly, then drifts and stalls

This matters because it rules out the explanation that live failure is only about
the offset perturbation.

### Conservative live test: `episode_idx=0`, `offset=0 cm`, tighter online control

Settings:

- `replan_every=1`
- `max_first_offset_mm=25`
- `max_step_mm=10`
- `smooth_window=1`

Artifacts:

- summary:
  [live_rollout_ep0_offset0_replan1_clamp25_step10_summary.json](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_replan1_clamp25_step10_summary.json)
- video:
  [live_offset_0.0cm_ep_0_replan1_clamp25.mp4](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_runs/dhb_starvla_qwen/live_rollout_videos/live_offset_0.0cm_ep_0_replan1_clamp25.mp4)

Result:

- still `success = false`
- early chunk errors are smaller
- but the same qualitative drift happens anyway
- the close snapshot remains far from the bowl

This is important because it shows the live failure is **not** just a chunk
cadence / smoothing problem.

## Comparison To Demo-Conditioned Canonical Baseline

The earlier offline-style canonical baseline still shows:

- `1/3` at `0 cm`
- `1/3` at `3 cm`

under:

- `replan_every = 4`
- `max_first_offset_mm = 35`
- `max_step_mm = 20`
- `smooth_window = 3`
- `blend_overlap_alpha = 0.6`

Reference:

- [starvla_embodik_canonical_baseline.md](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_embodik_canonical_baseline.md)

The live rollout results are therefore a stricter but more honest signal:

- the current `QwenDHB` checkpoint can support the partially offline
  demo-conditioned replanning path
- but it is **not yet strong enough** for true live receding-horizon execution

## Current Diagnosis

The execution contract in the live script matches the older successful stage-3
`embodik` evaluator. So the new failure is not best explained by another obvious
controller contract bug.

The more plausible explanation is:

1. early live chunks are good enough to move the arm
2. the executed state then deviates from the demo manifold
3. the next live replans are not robust to that deviation
4. IK feasibility collapses and the arm freezes

This means the remaining gap is now primarily a **model / data robustness** gap,
not another obvious rollout-plumbing bug.

## Remaining Caveats

Two caveats still matter when interpreting the live result:

1. The live loop still reuses demo-timed gripper commands.

   - The arm replans from live observations.
   - But the gripper timing still comes from `demo["gripper_cmd"][t:t+n_use]`.

   Once the live arm drifts away from the demo manifold, close/open timing can
   desynchronize from the actual live pose.

2. The live script currently runs a single controller setting only.

   The older offline evaluator retries failed runs with a higher-gain fallback.
   So a live failure is stricter than the canonical offline baseline, not yet a
   perfectly execution-matched comparison.

## Practical Conclusion

Safe statement:

- true live-observation replanning is implemented now
- the current checkpoint fails under that stricter online setting
- the offline-style canonical baseline remains useful for controlled progress
  tracking, but should not be presented as full live closed-loop performance

Most likely next steps:

1. keep `steps_200` as the online candidate based on the checkpoint sweep, not
   just train loss
2. training with state / observation perturbations closer to rollout conditions
3. later, augmentation studies using simulation or world-model spatial diversity
   to improve online replanning robustness

## Learned Gripper Token Follow-up

The first corrected dense-state gripper run changes the diagnosis again.

Checkpoint:

- `_workspace/starvla_runs/dhb_starvla_qwen_gripper_multitask4/checkpoints/steps_200_pytorch_model.pt`

What changed before this probe:

- `dhb.gripper_state` is now rebuilt from geometry-aligned close/open events
  instead of raw demo occupancy
- event / phase chunk sampling no longer hides a midpoint-centering bias
- live and offline sim evaluators now support `--disable-gripper-gates`
  so a learned gripper token can be evaluated without the legacy bowl / plate
  veto overriding it at runtime

Key findings:

1. The learned gripper branch is no longer dead.

   Under the old runtime vetoes, the canonical offline sweep looked always-open.
   That turned out to be misleading. The learned close commands survive through
   Stage 1, pack, and IK.

2. The learned token is still phase-misaligned.

   In the offline Stage-1 reference demos:

   - `episode 0`: closes at `20-23`
   - `episode 1`: closes at `52-55`
   - `episode 10`: never closes

   These are all misaligned relative to the geometry-conditioned target phases.

3. Removing the runtime vetoes does not recover success yet.

   Offline no-gate canonical sweep:

   - `0 cm`: `0/3`
   - `3 cm`: `0/3`

   But it does reveal real learned transitions:

   - `0 cm seed 0`: close at `20`, reopen at `24`
   - `0 cm seed 1`: close at `52`, reopen at `56`

4. True live learned-gripper rollout also fails.

   Artifact:

   - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_multitask4/live_rollout_ep0_offset0_step200_modelgrip_nogate_summary.json`

   Result:

   - `success = false`
   - `close_step = 4`
   - `reopen_step = 8`
   - `ik_fail_total = 82`
   - `min_bowl_xy_m ≈ 0.210`

Interpretation:

- The current gap is no longer “runtime gating suppressed the learned token.”
- The learned token now acts, but it acts at the wrong phase under live
  replanning.
- The older offline demo-conditioned Stage-1 loop is also a biased evaluator
  for learned gripper timing, because an early learned close is followed by a
  replan on the demo’s still-open observation stream.

Current best read:

- learned gripper evaluation should prefer the live rollout path
- the next lift should come from phase-alignment training / checkpoint
  selection, not more execution-side heuristics
