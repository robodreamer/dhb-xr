# DHB Action Head Validation Plan

- [x] Route SmolVLA-era inference through `SmolVLADHBAdapter`
- [x] Add unit regressions for state-forwarding and observation conversion
- [x] Confirm missing historical checkpoints and switch plan to fresh retraining
- [x] Patch `embodik` evaluation to accept fresh checkpoints and emit debug summaries
- [x] Verify trainer output format matches `embodik` evaluator
- [x] Run smoke retrain in Pixi env at `action_horizon=100` and produce `_workspace/dhb_head_e2e_fix_smoke.pt`
- [x] Run `embodik` predict stage on smoke checkpoint with debug JSON
- [x] Confirm intermediate checkpoint no longer shows persistent EE rotation / orientation drift
- [x] Run `embodik` IK stage on smoke predictions
- [x] Run `embodik` sim stage on smoke IK output
- [x] Review debug artifacts for wrong-Y signature, IK failures, and gripper-close position
- [x] Run recorded `embodik` sweep on smoke checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Run recorded `embodik` sweep on 50-epoch full checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Decide next iteration: continue e2e training toward the historically used `200` epochs
- [x] Add resume / warm-start support to `_workspace/train_dhb_head_e2e.py`
- [x] Continue e2e retraining from the current full checkpoint to `200` total epochs
- [x] Run recorded `embodik` sweep on the `200`-epoch checkpoint (`0cm`, `3cm`, `3` seeds)
- [x] Compare smoke vs `50ep` vs `200ep` checkpoints on loss and closed-loop success
- [ ] Add trainer logging hooks for future `wandb` integration
- [ ] Decide whether to keep smoke checkpoint as deployment baseline or redesign training / checkpoint selection
- [x] Implement terminal-step loss weighting for DHB invariant supervision
- [x] Implement end-anchored chunk oversampling in the StarVLA LeRobot loader
- [x] Add a dedicated `phaseaug` training config that enables the new low-risk robustness knobs
- [x] Verify the new `phaseaug` config with a real one-step Qwen+DHB train smoke
- [x] Run a short `phaseaug` warm-start from the old live-best checkpoint
- [x] Re-run the live checkpoint sweep on the `phaseaug` run and compare against `shorttrain1` / `liveaug1`
- [x] Add a reach-only live evaluation mode that ignores gripper timing and scores bowl approach + smoothness
- [x] Run a reach-only live checkpoint sweep on `dhb_starvla_qwen_invsup_shorttrain1`
- [x] Compare canonical execution against a stripped-down “raw-ish” execution on the best reach checkpoint
- [ ] Use the reach-only baseline as the main short-horizon target before revisiting learned gripper timing

## SmolVLA Exit Criteria

Use SmolVLA as a debugging baseline only if all of these remain true:
- [x] State-forwarding regression tests pass
- [x] Persistent EE rotation / runaway orientation is gone
- [x] `embodik` IK remains stable and low-error across demos
- [x] At least one checkpoint is reproducibly executable with recorded videos
- [x] Failures are consistent enough to diagnose per demo

Stop scaling SmolVLA training if all of these are true:
- [x] Longer training improves train loss substantially
- [x] Closed-loop `embodik` success does not improve with that extra training
- [x] Failures remain concentrated in the same demos (`demo_1`, `demo_10`)
- [x] IK is not the bottleneck
- [x] Higher-gain retries do not rescue the failing demos

## Final SmolVLA Diagnostic Pass

- [x] Re-run the `embodik` sim stage with per-demo close/reopen and object-relative diagnostics
- [x] Review `demo_0`, `demo_1`, and `demo_10` videos side by side
- [x] Classify the dominant failure mode for the failing demos: under-reaching in `Y`, too high in `Z`, or gripper timing
- [x] Save that failure signature as the reference baseline for future frameworks

## StarVLA Handoff Gate

Switch the main effort to StarVLA once the final SmolVLA diagnostic pass is complete and the failure signature is clear.

Carry forward:
- [x] Keep `embodik` as the primary evaluation path
- [x] Keep video recording enabled for every validation run
- [x] Keep the same `0cm` / `3cm` / `3`-seed comparison protocol
- [x] Keep per-demo diagnostics in the evaluation pipeline
- [x] Keep `_workspace/dhb_head_e2e_fix_smoke.pt` as the best SmolVLA executable baseline

Do not continue SmolVLA training with `500+` or `1000` epochs unless there is a specific architecture or supervision change to test.

## StarVLA Qwen+DHB Laptop Baseline

- [x] Add a lightweight `Qwen2.5-VL-3B` + DHB StarVLA config for laptop-class GPUs
- [x] Add forward-only smoke validation for custom StarVLA+DHB
- [x] Add one-step training smoke validation for custom StarVLA+DHB
- [x] Add prediction-side smoke validation with chunk-level action error
- [x] Fix mixed-dtype OAT nested-dropout failure exposed by StarVLA training
- [x] Track vendor-side StarVLA changes on a dedicated local branch
- [x] Add a minimal direct LIBERO rollout smoke with recorded video for the Qwen+DHB path
- [ ] Compare the first StarVLA rollout smoke qualitatively against the SmolVLA smoke baseline
- [ ] Decide whether to wire the Qwen+DHB path into `vla-evaluation-harness` as the main StarVLA-side evaluation path

## StarVLA To EmbodiK Bridge

- [x] Confirm the direct LIBERO rollout path still suffers from controller / nullspace effects even after fixing quaternion convention bugs
- [x] Verify that direct GT replay is a control-path issue, not only a DHB model-quality issue
- [x] Add a Stage-1 StarVLA→DHB trajectory export path that emits decoded SE(3) chunks from live LIBERO observations
- [x] Expose DHB trajectory prediction from the `QwenDHB` framework instead of only normalized controller actions
- [x] Fix missing state-conditioning in the StarVLA `QwenDHB` path when examples provide live `state`
- [x] Validate one saved StarVLA Stage-1 artifact for the `steps_200` checkpoint
- [x] Reuse the working SmolVLA `embodik` Stage-2 IK path on the saved StarVLA Stage-1 chunk
- [x] Run one full StarVLA→`embodik`→JOINT_POSITION rollout with recorded video
- [ ] Compare StarVLA→`embodik` nullspace behavior against the direct LIBERO controller rollout

Current Stage-1 artifact:
- script: `scripts/starvla_dhb_embodik_stage1.py`
- checkpoint: `_workspace/starvla_runs/dhb_starvla_qwen_longtrain1/checkpoints/steps_200_pytorch_model.pt`
- summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_stage1_step200_summary.json`
- saved chunk:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_stage1_step200.npz`

Interpretation:
- StarVLA can now produce DHB-decoded EE pose chunks directly from live LIBERO observations on the laptop GPU.
- The next step is no longer DHB decoding itself; it is handing those chunks into the already-working `embodik` IK / JOINT_POSITION execution stack.

## Reach-Only Baseline

- [x] Add a reach-only mode to live and offline `StarVLA -> DHB -> embodik` evaluators
- [x] Define reach-only success from bowl-relative EE geometry instead of full task reward
- [x] Log executed-trajectory smoothness metrics (`mean_accel_mm`, `hold_fraction`, `direction_flip_fraction`, `path_efficiency`)
- [x] Re-rank `shorttrain1` checkpoints on reach-only live metrics
- [x] Compare canonical execution vs stripped-down “raw-ish” execution on `shorttrain1/steps_200`
- [x] Test a rollout-grounded corrective augmentation warm-start (`reachcorr`) against the reach-only baseline
- [x] Test a cleaner reach-phase oversampling warm-start (`reachphase`) against the reach-only baseline
- [ ] Improve smooth bowl approach under offsets before returning to learned close/open timing
- [ ] Try a more structural next change than same-family warm-start training:
  - DHB continuation conditioning with a committed chunk prefix, or
  - a lightweight auxiliary phase/status head once reach improves
- [ ] Re-run the reach-only live checkpoint sweep after continuation-conditioned training
- [ ] Only after reach improves, reintroduce the integrated benchmark through the phase-split bridge
- [ ] Factor rollout buffering / reference adaptation into a tested adapter layer so live and offline loops share one contract

Tracked note:
- `_workspace/starvla_reach_only_findings.md`

Current reach-only ranking on the representative live cases:

| Step | Avg min bowl XY (m) | Avg min bowl XYZ (m) | Avg IK fails | Avg accel (mm) | Avg hold frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.1465 | 0.2954 | 75.5 | 0.2306 | 0.8346 |
| 100 | 0.2280 | 0.3672 | 76.0 | 0.2385 | 0.8418 |
| 50 | 0.2500 | 0.3739 | 80.5 | 0.1602 | 0.8836 |
| 150 | 0.2500 | 0.3740 | 79.5 | 0.1760 | 0.8700 |

Interpretation:

- `shorttrain1/steps_200` remains the best current checkpoint even without any
  gripper timing in the loop
- removing clamp / smoothing / overlap / repeated controller substeps makes the
  same checkpoint worse on both bowl proximity and smoothness
- so the current bottleneck is still trajectory quality, not mainly execution
  heuristics

Follow-up results:

- `reachcorr` did not beat the old reach-only baseline:
  - best checkpoint: `steps_200`
  - `0.1736 / 0.3331 / 76.5`
- `reachphase` also did not beat the old reach-only baseline:
  - best checkpoint: `steps_100`
  - `0.1913 / 0.3289 / 75.5`

Current interpretation:

- both new runs underperform the original `shorttrain1/steps_200` control
- more same-family warm-start training is therefore not the most credible next
  move
- the next likely improvement has to be more structural than:
  - synthetic target rewrite
  - small sampling-only rebiasing
  - plain state-history retraining

External repo checks now point in the same direction:

- `vla_foundry` suggests explicit past-vs-future conditioning plus a tested
  inference adapter for history / reference handling
- `TRI-LBM` suggests an auxiliary status signal can be trained jointly with the
  action generator, but this should come *after* reach-only continuation gains
  are real

## First StarVLA→EmbodiK End-To-End Result

- [x] Convert the Stage-1 StarVLA chunk into the existing `POSES_NPZ` contract expected by `_workspace/eval_dhb_head_embodik.py`
- [x] Run Stage-2 `embodik` IK on that converted chunk
- [x] Run Stage-3 `JOINT_POSITION` sim execution with recorded videos
- [ ] Improve Stage-1 trajectory feasibility so `embodik` no longer returns `NO_PROGRESS` on all waypoints

Current result:
- prepare summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_prepare_step200_summary.json`
- IK debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_step200_debug_ik.json`
- sim debug:
  - `_workspace/starvla_runs/dhb_starvla_qwen/embodik_step200_debug_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_step200/offset_0.0cm_seed_0.mp4`
  - `_workspace/validation_videos/starvla_embodik_step200/offset_0.0cm_seed_0_high_gains.mp4`

Interpretation:
- The `embodik` bridge is structurally working end to end.
- The current bottleneck is now the Stage-1 StarVLA trajectory itself:
  - `embodik` returned `NO_PROGRESS` on all `6/6` waypoints
  - Stage-3 videos show essentially no meaningful arm motion
- So the next improvement target is not nullspace control first; it is producing an IK-feasible Stage-1 EE trajectory chunk.

## StarVLA Training-Data Parity With SmolVLA

- [x] Trace why StarVLA Qwen+DHB training batches were missing live `state`
- [x] Enable `include_state: true` in the Qwen+DHB StarVLA config
- [x] Verify StarVLA batches now contain `state` with shape `(T, 8)`
- [x] Trace why StarVLA batches were missing DHB invariant supervision
- [x] Patch the LIBERO→LeRobot converter to export `dhb.linear_invariants` and `dhb.angular_invariants`
- [x] Patch the StarVLA LeRobot dataset loader to pass `dhb.*` supervision fields through to training samples
- [x] Verify a real StarVLA sample now contains `state`, `dhb.linear_invariants`, and `dhb.angular_invariants`
- [x] Verify `QwenDHB._extract_gt_invariants(...)` now returns a valid `(1, 8, 8)` target tensor on a real sample
- [ ] Regenerate the full `/home/andypark/Projects/data/libero_lerobot` dataset with inline DHB invariants
- [ ] Restart Qwen+DHB training on the regenerated full dataset before judging Stage-1 / `embodik` quality again

Interpretation:
- The earlier StarVLA→`embodik` failure was not comparable to SmolVLA yet.
- StarVLA had been training with the action branch only; the invariant decoder path used for `predict_trajectory()` was effectively unsupervised.
- That mismatch is now fixed in code and validated on a one-demo smoke export.

## First Direct StarVLA Rollout Result

- [x] Run a direct local LIBERO rollout smoke on `libero_spatial/task_id=0/episode_idx=0`
- [x] Save the first rollout video and JSON summary
- [x] Confirm the direct rollout path is executable on the laptop GPU without a policy server
- [ ] Review the first rollout video against the SmolVLA baseline and identify whether the failure mode is already the same `Y` / `Z` placement-margin signature

Current result:
- rollout path: `scripts/starvla_dhb_libero_rollout_smoke.py`
- config: `configs/libero_dhbxr_qwen.yaml`
- outcome: `failure` after `220` steps
- runtime: about `66.1 s`
- peak GPU memory: about `4286.9 MB`
- video: `_workspace/starvla_runs/dhb_starvla_qwen/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`

Interpretation:
- This is the first behavior-level StarVLA+DHB baseline, not a trained result worth judging on task success yet.
- The main milestone is that the direct rollout stack now works end to end with recorded video on the laptop path.

## Short Qwen+DHB Training Check

- [x] Run a short frozen-Qwen StarVLA training job on the laptop path
- [x] Save at least one intermediate checkpoint from that run
- [x] Re-run the same direct LIBERO rollout from the intermediate checkpoint
- [ ] Decide whether orientation issues now look like model quality or remaining action/dataset semantics mismatch

Current result:
- training run: `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2`
- checkpoint: `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2/checkpoints/steps_50_pytorch_model.pt`
- comparison rollout summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_shorttrain2/libero_rollout_smoke_summary_step50.json`
- outcome after `50` steps: still `failure` on the same `libero_spatial/task_id=0/episode_idx=0` rollout

Interpretation:
- A short run is enough to verify the checkpointing / reload path.
- It is not enough to make the Qwen+DHB policy behavior credible yet.
- Remaining orientation problems are no longer obviously caused by the original rollout bug; they now look more like model quality and/or residual dataset-action convention issues.

## Longer Qwen+DHB Checkpoint Comparison

- [x] Fix the StarVLA trainer checkpoint-save path for single-process laptop runs
- [x] Run a longer frozen-Qwen training job and keep intermediate checkpoints
- [x] Add checkpoint-specific rollout video naming to avoid overwriting artifacts
- [x] Re-run the same rollout from a later checkpoint (`steps_200`)
- [ ] Decide whether the remaining orientation issue is dominated by model quality or by quantized-checkpoint reload semantics

Current result:
- training run: `_workspace/starvla_runs/dhb_starvla_qwen_longtrain1`
- checkpoints reached: `steps_100_pytorch_model.pt`, `steps_200_pytorch_model.pt`
- rollout summary from `steps_200`:
  - `_workspace/starvla_runs/dhb_starvla_qwen_longtrain1/libero_rollout_smoke_summary_step200.json`
- video from `steps_200`:
  - `_workspace/starvla_runs/dhb_starvla_qwen/rollout_steps_200_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_failure.mp4`
- outcome after `200` steps: still `failure`

Interpretation:
- Longer training changes the action outputs and makes the checkpoint comparison more meaningful than the `50`-step smoke.
- It still does not produce task success on the reference rollout.
- The run required `llm_int8_enable_fp32_cpu_offload=true` for reload, which produced meta-parameter warnings during `load_state_dict()`. That means the next cleanup target is the quantized checkpoint reload path itself, not just more training time.

## StarVLA Embodik Seed Robustness Follow-up

- [x] Compare replanned StarVLA→`embodik` behavior across seeds `0`, `1`, and `2`
- [x] Confirm the current `1/3` success is real and not another rollout-orientation bug
- [x] Add a diagnostic note for the seed split
- [x] Fix the seed-identity handoff so pack/IK preserve the originating `episode_idx`
- [x] Add a fail-fast guard for the harness `use_embodik_ik=true` contract mismatch
- [ ] Improve Stage-1 trajectory feasibility for the failing seeds before scaling training or benchmarks again

Current result:
- diagnostic note:
  - `_workspace/starvla_embodik_seed_diagnostic.md`
- repacked multiseed bundle:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_pack_step200_replan6_sweep_after_seedfix.json`
- IK rerun after seed-fix:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_after_seedfix_debug_ik.json`

Interpretation:
- The remaining failure split is now credible: `seed 0` is workable, `seed 1` fails IK from the first waypoint, and `seed 2` is partly feasible but not task-correct.
- The old orientation and gripper pipeline bugs were real, but they are no longer the dominant issue here.
- The next bottleneck is Stage-1 trajectory quality, especially the large first-waypoint translational overshoot seen across seeds and worst on `seed 1`.

## True Live-Observation Replanning Check

- [x] Add a true live-observation StarVLA→DHB→`embodik` rollout script that replans from the current env observation instead of demo frames
- [x] Verify the live rollout uses the same JOINT_POSITION execution contract as the older stage-3 `embodik` evaluator
- [x] Run the canonical live test on `episode_idx=1`, `offset=3 cm`
- [x] Run an easier live test on `episode_idx=0`, `offset=0 cm`
- [x] Run a more conservative live test (`replan_every=1`, tighter clamp) to check whether the online failure is just a chunk-tuning issue
- [x] Add an online-feasibility checkpoint sweep over the short-train checkpoints
- [x] Rank checkpoints by live metrics instead of train loss alone
- [x] Select the current online candidate checkpoint from that sweep
- [x] Decide whether the current model is strong enough for true live replanning, or whether the offline-style canonical baseline must remain the main progress metric for now
- [x] Add a first-pass live-robustness training variant with state-input drift augmentation
- [x] Run a short warm-start (`300` steps) from the old live-best checkpoint under that augmentation
- [x] Re-run the live checkpoint sweep on the augmented run and compare it against the old live-best checkpoint
- [ ] Decide whether to continue with stronger rollout-grounded augmentation or pivot to phase-aware supervision first

Current result:
- live rollout note:
  - `_workspace/starvla_live_rollout_findings.md`
- canonical live `episode 1 / 3 cm` summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep1_offset3_summary.json`
- canonical live `episode 0 / 0 cm` summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_summary.json`
- conservative live `episode 0 / 0 cm` summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_rollout_ep0_offset0_replan1_clamp25_step10_summary.json`
- checkpoint sweep summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.md`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/live_checkpoint_sweep/summary.json`

Interpretation:
- True live replanning is now implemented and executable on the laptop GPU.
- The current `QwenDHB` checkpoint fails under that stricter online setting, even on the easier zero-offset case.
- Tighter receding-horizon control reduces the earliest chunk errors but does not recover task success.
- The remaining gap is therefore better explained by online model / data robustness than by another obvious rollout-plumbing bug.
- Follow-up cleanup confirmed a real init-state mismatch (`demo` vs `benchmark` init state L2 about `0.357`), but switching to the demo init state plus bowl-relative close gating still did not recover success.
- Under the stricter geometry-conditioned gripper logic, the live run never reaches a valid grasp region at all (`close_step = null`), which sharpens the diagnosis: the current live trajectory quality is still insufficient for online grasp acquisition.
- A live checkpoint sweep across `steps_{50,100,150,200}` now provides a more honest selection signal than train loss alone.
- `steps_200` is the best current online candidate:
  - `avg_min_bowl_xyz_m = 0.2916`
  - `avg_min_bowl_xy_m = 0.1431`
  - `avg_ik_fail_total = 75.5`
- none of the current checkpoints achieve live success yet, so the offline-style canonical baseline still remains the main progress metric for now.

### First live-robustness augmentation follow-up

- config:
  `configs/libero_dhbxr_qwen_liveaug.yaml`
- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_liveaug1`
- sweep summary:
  `_workspace/starvla_runs/dhb_starvla_qwen_liveaug1/live_checkpoint_sweep/summary.md`

Result:

- best augmented checkpoint: `steps_300`
  - `avg_min_bowl_xyz_m = 0.3245`
  - `avg_min_bowl_xy_m = 0.1986`
  - `avg_ik_fail_total = 73.5`

Interpretation:

- this lightweight state-input drift augmentation improves IK-feasibility
  slightly, but it still loses clearly on bowl proximity to the old online
  candidate (`steps_200`)
- so simple state-noise training is not enough by itself to close the live gap
- the next training-side move should be a stronger, more rollout-grounded
  robustness intervention rather than more same-objective warm-start training

### Phase-aware warm-start: `dhb_starvla_qwen_phaseaug1`

Run:

- directory:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1`
- warm-start checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/checkpoints/steps_200_pytorch_model.pt`
- train budget:
  `300` steps

Live sweep summary:

- markdown:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1/live_checkpoint_sweep/summary.md`
- json:
  `_workspace/starvla_runs/dhb_starvla_qwen_phaseaug1/live_checkpoint_sweep/summary.json`

Ranking:

- `steps_200`:
  - `avg_min_bowl_xyz_m = 0.3182`
  - `avg_min_bowl_xy_m = 0.1894`
  - `avg_ik_fail_total = 75.5`
- `steps_300`:
  - `0.3347`
  - `0.2087`
  - `75.5`
- `steps_100`:
  - `0.3415`
  - `0.2029`
  - `31.5`

Comparison against earlier runs:

- old live best: `shorttrain1/steps_200`
  - `0.2916 / 0.1431 / 75.5`
- first state-noise warm-start: `liveaug1/steps_300`
  - `0.3245 / 0.1986 / 73.5`
- new phase-aware warm-start: `phaseaug1/steps_200`
  - `0.3182 / 0.1894 / 75.5`

Interpretation:

- terminal-step weighting plus end-anchored chunk oversampling is better than
  `liveaug1` on bowl approach
- but it still does not beat the original `shorttrain1` live baseline
- the next training-side intervention should therefore be more rollout-grounded,
  not another same-objective warm-start on the current demo distribution

## 2026-04-19 Gap-Closure Execution Plan

Reference note:
- `notes/supplementary/16_starvla_dhb_gap_closure_plan.md`

### Phase 1: Freeze the canonical StarVLA->`embodik` loop

- [x] Run a bounded calibration sweep over `replan_every in {4, 6}` and `max_first_offset_mm in {35, 40}` with `max_step_mm=20`, `smooth_window=3`, `blend_overlap_alpha=0.6`
- [x] Compare candidates on:
  - success at `0cm` / `3cm`
  - IK failure rate
  - visible motion continuity
  - close / reopen timing
- [x] Select one shared StarVLA closed-loop config as the canonical baseline

Selected canonical baseline:
- reference note:
  - `_workspace/starvla_embodik_canonical_baseline.md`
- chosen setting:
  - `replan_every = 4`
  - `max_first_offset_mm = 35`
  - `max_step_mm = 20`
  - `smooth_window = 3`
  - `blend_overlap_alpha = 0.6`
  - demo gripper timing
  - current release semantics from `_workspace/eval_dhb_head_embodik.py`

### Phase 2: Improve manipulation quality under the frozen loop

- [ ] Add bowl-relative close gating
- [ ] Keep plate-relative release gating and explicit placement-tail logic where needed
- [ ] Continue using demo gripper timing as the control branch while trajectory quality is tuned
- [ ] Compare `seed 0`, `seed 1`, `seed 2` close/reopen geometry under the frozen loop
- [ ] Match the SmolVLA smoke baseline with one shared config:
  - `2/3` at `0cm`
  - `1/3` at `3cm`

### Phase 3: Close the matched-baseline gap

- [x] Establish the first matched regular-head control on the corrected StarVLA setup:
  - `QwenOFT`
  - same `Qwen2.5-VL-3B` backbone
  - same `libero_spatial_dhbxr` data mix
  - same frozen-Qwen `200`-step short-train budget
  - same direct LIBERO rollout smoke (`task_id=0`, `episode_idx=0`)
  - reference note: `_workspace/starvla_qwenoft_control_baseline.md`
- [ ] Run a matched regular-head baseline (`QwenOFT` or equivalent) under the same evaluator and offsets
- [ ] Build a comparison table for:
  - success rate
  - IK failure rate
  - final `ee_minus_plate`
  - close / reopen geometry
- [ ] Decide which DHB claims are now strong enough to make over a regular action head

Current status:
- The first short-train `QwenOFT` control is now runnable and tracked.
- On one matched sample, `QwenOFT` has worse chunk MAE than the corrected `QwenDHB`
  checkpoint:
  - `QwenDHB`: `0.1545`
  - `QwenOFT`: `0.2626`
- Both direct-action rollout smokes still fail on the reference LIBERO task, so
  this only partially closes the claim gap.

### Phase 4: Scale out only after the canonical baseline is stable

- [ ] Fix the harness-side action contract for `embodik` execution
- [ ] Wire the canonical StarVLA->DHB->`embodik` path into the harness server
- [ ] Re-run LIBERO harness smoke with the canonical path
- [ ] Only then broaden to VLA-Arena / RoboCasa smoke validation

### Phase 5: Prepare the stronger DHB-XR scaling story

Reference note:
- `notes/supplementary/17_dhbxr_scaling_story_note.md`

- [ ] Define the matched action-space comparison plan:
  - `DHB-XR`
  - `relative`
  - `delta`
  - `absolute`
- [ ] Add a small-seed-data study plan for `N in {1, 3, 5, 10}`
- [ ] Define the simulator / world-model augmentation study contract:
  - diverse observations
  - DHB-XR-adapted trajectories
  - matched non-DHB baselines
- [ ] Define the cross-embodiment reuse study once the canonical StarVLA baseline is stable

## Stage-1 Feasibility Clamp Pass

- [x] Add a lightweight Stage-1 feasibility pass to clamp the first waypoint and per-step motion
- [x] Rerun the same `replan6` multiseed StarVLA→`embodik` sweep with recorded videos
- [x] Compare IK feasibility before vs after the clamp
- [ ] Decide whether the next change should target grasp / placement quality rather than generic feasibility

Current clamp settings:
- `max_first_offset_mm = 40`
- `max_step_mm = 20`
- `smooth_window = 3`

Artifacts:
- stage1 summaries:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_stage1_step200_seed{0,1,2}_replan6_clamp40_step20_sw3.json`
- IK summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_ik.json`
- sim summary:
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp40_step20_sw3_debug_sim.json`
- videos:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan6_clamp40_step20_sw3_sweep`

Key outcome:
- IK feasibility improved strongly, especially for `seed 1`
  - `0 cm seed 1`: `84/84 -> 19/84` failures
  - `3 cm seed 1`: `84/84 -> 20/84` failures
- Closed-loop success did **not** improve
  - still `1/3` at `0 cm`
  - still `1/3` at `3 cm`

Interpretation:
- The clamp confirms that early translational overshoot was a real bottleneck.
- But once that is reduced, the main remaining problem becomes trajectory quality around grasp / placement, not just immediate IK feasibility.

## Non-zero Offset Follow-up

- [x] Test a tighter first-offset cap (`35 mm`) on the smoother `replan4 + blend`
  execution path
- [x] Fix sim-stage gripper semantics so settling no longer force-recloses the
  hand
- [x] Add a release gate so reopen can be delayed until the TCP is actually
  over the plate and low enough to release
- [x] Test a `replan6 + clamp35 + blend06` compromise to see whether longer
  commit horizons help the failing `3 cm` seeds
- [x] Add bowl-relative close gating to the canonical `replan4 + clamp35 + blend06`
  sim loop and rerun the full `0 cm / 3 cm / 3 seed` sweep
- [x] Try one minimal late-phase placement tail on top of the geometry-gated
  canonical loop

Current best `3 cm` artifacts:
- `seed 1` success under the smoother `replan4` setting:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan4_clamp35_step20_sw3_blend06_offset3_sweep/offset_3.0cm_seed_1.mp4`
- `seed 0` success under the `replan6` compromise with high gains:
  - `_workspace/validation_videos/starvla_embodik_invsup_step200_replan6_clamp35_step20_sw3_blend06_offset3_releasegate_sweep/offset_3.0cm_seed_0_high_gains.mp4`

Current summaries:
- `replan4 + clamp35 + blend06`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_offset3_debug_sim.json`
- `replan6 + clamp35 + blend06 + release gate`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan6_clamp35_step20_sw3_blend06_offset3_releasegate_debug_sim.json`
- `replan4 + clamp35 + blend06 + bowl close gate`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_geomgates_sim.json`
- `replan4 + clamp35 + blend06 + bowl close gate + minimal placement tail`
  - `_workspace/starvla_runs/dhb_starvla_qwen_invsup_shorttrain1/embodik_step200_replan4_clamp35_step20_sw3_blend06_geomgates_tail2_sim.json`

Interpretation:
- The loop semantics are now substantially cleaner:
  - no more forced gripper re-close during settling
  - first close now respects live bowl-relative geometry
  - reopen can be held until the TCP is actually over the plate
- The bowl-relative close gate is mechanically correct, but it still leaves the
  canonical baseline at `1/3` for both `0 cm` and `3 cm`
- The minimal placement tail is also not enough to raise the shared baseline:
  - it only accelerated the existing `0 cm seed 1` success
  - it forced one late reopen for `3 cm seed 2`, but still did not create a
    new success
- So the remaining gap is no longer best attacked with more small sim-side
  heuristics.
- Different seeds prefer different replanning cadences:
  - `seed 1` is best under `replan4`
  - `seed 0` is best under `replan6`
  - `seed 2` remains the clearest “almost there” case, but now points toward a
    stronger phase-aware model or supervision change rather than another
    execution-only tweak

## Joint DHB + Contact Signal

- [x] Add a jointly trained DHB contact head and export close/open supervision into the LeRobot dataset
- [x] Verify `QwenDHB.predict_trajectory()` exposes learned contact probabilities
- [x] Route Stage-1 and live rollout `gripper_source=model` through the learned contact head instead of the old raw-action gripper branch
- [x] Run an initial contact retrain and probe close/open predictions on real dataset chunks
- [x] Run a stronger weighted-contact retrain from the best initial contact checkpoint
- [ ] Replace demo-timed gripper replay only after the learned close signal becomes usable in rollout

Artifacts:
- initial contact run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_contact1`
- weighted contact run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_contact_weighted1`
- checkpoint probes:
  - `_workspace/starvla_runs/dhb_starvla_qwen_contact1/contact_checkpoint_probe.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_contact_weighted1/contact_checkpoint_probe.json`
- model-gripper stage-1 check:
  - `_workspace/starvla_runs/dhb_starvla_qwen_contact1/embodik_stage1_ep0_modelgrip_step200.json`

Interpretation:
- The joint token+gripper path is now real end to end:
  - data export emits `dhb.close_contact` / `dhb.open_contact`
  - StarVLA batches carry those targets
  - the DHB head predicts a parallel contact signal
  - rollout code can consume learned contact outputs directly
- Current learned behavior is asymmetric:
  - open-event prediction is strong
  - close-event prediction is still effectively collapsed
- On canonical stage-1 replay with `gripper_source=model`, the learned contact
  branch keeps the gripper open throughout the episode, so the model-only
  gripper path is not yet a valid replacement for demo timing.
- The next gap is therefore not rollout plumbing. It is the close-event
  learning objective itself.

## Gripper Event Token

- [x] Replace the 2-logit close/open contact head with a 3-way event token:
  `hold / close / open`
- [x] Export `dhb.gripper_event` labels into the LeRobot dataset
- [x] Add event-centered chunk oversampling in the StarVLA loader
- [x] Retrain from the invariant-supervised checkpoint with perception-conditioned
  event prediction
- [x] Run the learned-gripper checkpoint through the frozen offline canonical
  `StarVLA -> DHB -> embodik` loop

Artifacts:
- config:
  - `configs/libero_dhbxr_qwen_gripper_token.yaml`
  - `configs/libero_dhbxr_qwen_gripper_token_focus.yaml`
- run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token1`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_centered1`
- stage-1 model-gripper sweeps:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1/embodik_stage1_ep0_modelgrip_canonical.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1/embodik_stage1_ep1_modelgrip_canonical.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1/embodik_stage1_ep10_modelgrip_canonical.json`
- offline canonical evaluation:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1/embodik_modelgrip_canonical_ik.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_token_focus1/embodik_modelgrip_canonical_sim.json`
  - `_workspace/validation_videos/starvla_embodik_gripper_token_focus1_canonical`

Interpretation:
- The 3-way event token fixes the previous degenerate failure mode:
  - the model no longer stays open forever
  - the learned gripper now produces real transitions in Stage 1
- But the new event head is phase-wrong:
  - it closes from the first chunk on all three seeds
  - `episode 1` and `episode 10` also show an early reopen / re-close pattern
- Under the same frozen offline canonical loop, learned gripper makes the
  overall result worse:
  - `0 cm`: `0/3`
  - `3 cm`: `0/3`
  - every run starts effectively closed, so Stage 3 records `close_step = null`
- So the current learned event token has crossed the threshold from “dead” to
  “active,” but it is still not phase-correct enough to replace demo timing in
  benchmark claims.
- This narrows the next model gap:
  - not “make the gripper event head exist”
  - but “align close/open to perceptual task phase rather than overfitting an
    always-close prior”
- A concrete loader-side bias was present and is now fixed:
  - among close-event chunks from the first 20 parquet files, the previous
    oversampler placed the first close label at relative position `0` in
    `105 / 252` candidates
  - the centered-event sampler now targets relative position `4` for the 8-step
    chunk, which removes that front-loading bias in the training examples
- But the centered retrain still does not fix rollout-phase alignment:
  - `dhb_starvla_qwen_gripper_token_centered1` still predicts immediate close on
    `episode 0` and `episode 1`, and stays closed throughout the preview on
    `episode 10`
  - so the remaining blocker is no longer just chunk selection; it is that the
    close/open labels still are not object/perception-conditioned enough

### Geometry-Conditioned Gripper Relabeling

- [x] Regenerate `libero_lerobot` with geometry-conditioned close/open labels
- [x] Warm-start the focused gripper-token checkpoint on the regenerated data
- [x] Re-run the frozen offline canonical `StarVLA -> DHB -> embodik` loop
- [x] Fix the Stage-1 learned-gripper decode bug that was partially resetting
  each chunk from the demo gripper state instead of the model's last decoded
  state

Artifacts:
- run:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1`
- regenerated dataset:
  - `/home/andypark/Projects/data/libero_lerobot`
- stage-1 model-gripper sweeps:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1/stage1_ep0_modelgrip_summary.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1/stage1_ep1_modelgrip_summary.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1/stage1_ep10_modelgrip_summary.json`
- offline canonical evaluation:
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1/embodik_modelgrip_geom_canonical_ik.json`
  - `_workspace/starvla_runs/dhb_starvla_qwen_gripper_geom1/embodik_modelgrip_geom_canonical_sim.json`
  - `_workspace/validation_videos/starvla_embodik_gripper_geom1_canonical`

Interpretation:
- The geometry relabeling itself is sane:
  - full regenerated dataset event counts are:
    - hold: `53724`
    - close: `3575`
    - open: `4951`
  - reference `demo_0` close labels shift to a geometry band around steps
    `37-42`, and open labels to `77-88`
- A real evaluation bug was present in the offline Stage-1 path:
  - model gripper decoding was being re-initialized from the demo's previous
    gripper command at each chunk boundary
  - this created a false `closed -> open -> closed` oscillation in
    `episode 1` and `episode 10`
  - the Stage-1 script now carries the model's last decoded gripper command
    across chunks instead
- After fixing that decode bug, the geometry-labeled checkpoint still fails
  under the frozen canonical offline loop:
  - `0 cm`: `0/3`
  - `3 cm`: `0/3`
- The corrected diagnosis is stronger than before:
  - the current learned event token is not merely being mis-decoded
  - it is genuinely biased toward early close from the first chunk
  - after the decoder fix, all three reference episode previews start closed
    for the first three chunks under the canonical `replan4 + clamp35 +
    blend06` settings
- So the next gap is no longer rollout plumbing or chunk-boundary reset.
  It is the event supervision itself:
  - the model still learns an always-close prior even when trained on
    geometry-conditioned close/open labels
  - the next model-side change should target phase semantics more directly,
    likely by conditioning the contact head on object/perception cues more
    explicitly or by changing the event target structure rather than only
    relabeling the same 3-way token

### Sparse Geometry Event Tokens

- [x] Fix the Stage-3 learned-gripper initialization contract
- [x] Change geometry-labeled `dhb.gripper_event` from a dense in-gate band to
  a sparse edge-trigger token
- [ ] Regenerate `/home/andypark/Projects/data/libero_lerobot` with the sparse
  event token
- [ ] Warm-start the gripper-token run again on the regenerated dataset

Interpretation:
- A second real evaluation bug remained in Stage 3:
  - `_workspace/eval_dhb_head_embodik.py` was seeding the effective gripper
    state from `gripper_cmds[0]` instead of the actual env/demo initial gripper
    qpos
  - that contaminated the learned-gripper canonical sweep because the first
    predicted close could bypass the live bowl gate entirely
- The bigger contract issue was in the exporter:
  - `dhb.gripper_event` was still being written as a dense occupancy band over
    every timestep inside the bowl-ready / plate-ready regions
  - but the loader, centered event sampler, 3-way cross-entropy head, and
    decode path all treat it as a sparse transition token
- The exporter now keeps:
  - `dhb.close_contact` and `dhb.open_contact` as dense geometry masks
  - `dhb.gripper_event` as a sparse edge-trigger token:
    - close on first entry into the bowl-ready region
    - open on first entry into the plate-ready region
- Reference `demo_0` after the sparse-event patch:
  - `close_contact`: steps `37-42`
  - `open_contact`: steps `77-88`
  - `close_event`: step `37`
  - `open_event`: step `77`
- The first rebalanced retrain on the old dense-band target still stays closed
  from chunk 0 on `episode 0`, so this sparse-event relabeling is now the
  highest-leverage model-side change to test before another round of sampler or
  class-weight tuning.

### Dense Gripper State, Geometry-Aligned

- [x] Rebuild `dhb.gripper_state` from geometry-aligned close/open events
- [x] Remove hidden midpoint anchoring from event / phase chunk sampling
- [x] Tighten `QwenDHB` so gripper-focused runs fail fast if DHB labels are
  missing
- [x] Probe corrected dense-state checkpoints on reference demos before full
  rollouts
- [x] Run the best corrected dense-state checkpoint through the frozen offline
  canonical `StarVLA -> DHB -> embodik` loop
- [x] Remove runtime bowl / plate vetoes from learned-gripper evaluation via an
  explicit `--disable-gripper-gates` flag
- [x] Recheck live rollout with learned gripper and no runtime vetoes

Interpretation:
- The corrected dense-state path is materially better than the earlier event
  token runs:
  - `multitask4/steps_100` is the healthiest startup checkpoint
  - `multitask4/steps_200` is the first checkpoint that both stays open at
    chunk start and later predicts real closes on reference demos
- Reference Stage-1 close windows for `multitask4/steps_200` are:
  - `episode 0`: closed at steps `20-23`
  - `episode 1`: closed at steps `52-55`
  - `episode 10`: no close
- Under the old Stage-3 geometry vetoes, these learned closes were silently
  suppressed and the rollout looked always-open. That was an evaluation
  contract bug for learned-gripper runs, not evidence that the head was dead.
- With the vetoes disabled:
  - offline canonical sweep is still `0/3` at `0 cm` and `0/3` at `3 cm`
  - but real close / reopen events now appear in sim for seeds `0` and `1`
- The stricter live rollout with learned gripper and no vetoes also still fails:
  - `episode 0`, `offset 0 cm`, `steps_200`
  - `close_step = 4`
  - `reopen_step = 8`
  - `ik_fail_total = 82`
  - `min_bowl_xy_m ≈ 0.210`
- So the current bottleneck is no longer “gripper branch is dead.” It is phase
  alignment under live replanning:
  - seed `0` closes far too early
  - seed `1` closes too late and reopens too quickly
  - seed `2` never closes
- This also exposes a limitation of the offline demo-conditioned Stage-1 loop
  for learned gripper evaluation: if the learned policy closes earlier than the
  demo, the next replan still sees the demo’s open-gripper observation, which
  biases it back toward reopening. Live evaluation is therefore the more honest
  signal for learned gripper timing than the frozen offline canonical loop.

### 2026-04-21 update: external gripper references favor state + debounce, not pure event tokens

External repo checks sharpen the gripper direction:

- `vla_foundry` treats gripper as a standard learned action/state dimension and
  applies a small hysteresis debouncer at inference time
- `TRI-LBM` does not add gripper-specific event structure; its closest useful
  analogue is a lightweight auxiliary status signal

The updated gripper-side plan is therefore:

- use learned `gripper_state` as the primary benchmark-time gripper output
- keep `gripper_event` / `gripper_phase` as auxiliary supervision only
- stabilize learned gripper inference with adapter-level hysteresis, not demo
  timing or script-local ad hoc logic

The shared rollout code now supports this directly:

- `src/dhb_xr/integration/vla/libero.py`
- `src/dhb_xr/integration/vla/chunk_rollout.py`
- `scripts/starvla_dhb_embodik_stage1.py`
- `scripts/starvla_dhb_embodik_live_rollout.py`

Current default hysteresis for learned `gripper_state`:

- close threshold: `0.6`
- open threshold: `0.4`

This is not the main reach breakthrough. It is a benchmark-contract cleanup that
should reduce chatter once the model-side gripper-state path is revisited.

### Reach-Only Handoff Baseline

- [x] Add a committed-handoff mode to the shared chunk runner
- [x] Probe `ramp` overlap blending with `commit_prefix_steps = 1/2`
- [x] Re-rank the full reach-only checkpoint sweep under the best handoff mode
- [x] Push the same handoff back into the full offline integrated pipeline

Interpretation:
- A structural execution-side change finally improved the reach-only control
  baseline without retraining:
  - `blend_mode = ramp`
  - `commit_prefix_steps = 2`
- On the best checkpoint (`shorttrain1/steps_200`), both reference cases
  improved:
  - `episode 0`, `0 cm`:
    - bowl XY `0.1590 -> 0.1410`
    - bowl XYZ `0.2974 -> 0.2799`
  - `episode 1`, `3 cm`:
    - bowl XY `0.1339 -> 0.1223`
    - bowl XYZ `0.2935 -> 0.2862`
- Full reach-only checkpoint sweep under this handoff now ranks:
  - `steps_200`: `avg_min_bowl_xy_m = 0.1375`
  - `steps_100`: `0.2287`
  - `steps_50`: `0.2500`
  - `steps_150`: `0.2500`
- So this is the new best **reach-only** execution mode.

Integrated consequence:
- The same handoff was then pushed through the full offline
  `StarVLA -> DHB -> embodik` pipeline with demo gripper timing.
- Result:
  - old integrated canonical:
    - `0 cm`: `1/3`
    - `3 cm`: `1/3`
  - new handoff integrated:
    - `0 cm`: `0/3`
    - `3 cm`: `0/3`
- So the gain is real, but phase-local:
  - good enough to revisit the integrated pipeline experimentally
  - not good enough to replace the integrated canonical loop globally
- Current best interpretation:
  - keep `ramp + commit2` as the best pre-grasp / reach-only mode
  - keep the old `replan4 + clamp35 + blend06` loop as the integrated default
  - next integrated move should be phase-split rather than a whole-loop swap

### Fresh Corrected Training + State History: Negative Result

- [x] Add a fresh corrected-dataset training config with `state_history_window = 4`
- [x] Run a from-scratch frozen-Qwen training job on the regenerated dataset
- [x] Sweep the fresh checkpoints under the current best reach-only handoff

Interpretation:
- This did **not** beat the reach-only control baseline.
- Fresh run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_statehist4`
- Best checkpoint:
  - `step 200`
  - `avg_min_bowl_xy_m = 0.2381`
  - `avg_min_bowl_xyz_m = 0.3681`
  - `avg_ik_fail_total = 42.0`
- Current reach-only control baseline remains:
  - `shorttrain1/steps_200`
  - `avg_min_bowl_xy_m = 0.1375`
  - `avg_min_bowl_xyz_m = 0.2867`
  - `avg_ik_fail_total = 75.5`
- So the plain “fresh corrected train + state history” path is not the next
  breakthrough by itself.

### Phase-Split Integrated Bridge

- [x] Add a phase-split chunk mode:
  - `ramp + commit2` only before the first close phase
  - canonical chunking after that
- [x] Re-run the full offline integrated `StarVLA -> DHB -> embodik` sweep

Interpretation:
- Global `ramp + commit2` had regressed the integrated sweep to:
  - `0 cm`: `0/3`
  - `3 cm`: `0/3`
- Phase-split recovers the old aggregate baseline:
  - `0 cm`: `1/3`
  - `3 cm`: `1/3`
- This is useful, but not a true breakthrough:
  - keep canonical as the integrated default
  - keep phase-split as the best integrated bridge for reach-first behavior

### Current Priority

The next highest-value move is now:

1. stop spending the next cycle on more same-family fresh training
2. stop broadening execution heuristics beyond the new phase-split bridge
3. move the model side toward continuation conditioning:
   - committed-prefix continuation input, or
   - short action-history / prefix-conditioned chunk continuation
