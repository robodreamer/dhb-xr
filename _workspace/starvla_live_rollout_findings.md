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
