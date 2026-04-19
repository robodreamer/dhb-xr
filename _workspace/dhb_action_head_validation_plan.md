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

## 2026-04-19 Gap-Closure Execution Plan

Reference note:
- `notes/supplementary/16_starvla_dhb_gap_closure_plan.md`

### Phase 1: Freeze the canonical StarVLA->`embodik` loop

- [ ] Run a bounded calibration sweep over `replan_every in {4, 6}` and `max_first_offset_mm in {35, 40}` with `max_step_mm=20`, `smooth_window=3`, `blend_overlap_alpha=0.6`
- [ ] Compare candidates on:
  - success at `0cm` / `3cm`
  - IK failure rate
  - visible motion continuity
  - close / reopen timing
- [ ] Select one shared StarVLA closed-loop config as the canonical baseline

### Phase 2: Improve manipulation quality under the frozen loop

- [ ] Add bowl-relative close gating
- [ ] Keep plate-relative release gating and explicit placement-tail logic where needed
- [ ] Continue using demo gripper timing as the control branch while trajectory quality is tuned
- [ ] Compare `seed 0`, `seed 1`, `seed 2` close/reopen geometry under the frozen loop
- [ ] Match the SmolVLA smoke baseline with one shared config:
  - `2/3` at `0cm`
  - `1/3` at `3cm`

### Phase 3: Close the matched-baseline gap

- [ ] Run a matched regular-head baseline (`QwenOFT` or equivalent) under the same evaluator and offsets
- [ ] Build a comparison table for:
  - success rate
  - IK failure rate
  - final `ee_minus_plate`
  - close / reopen geometry
- [ ] Decide which DHB claims are now strong enough to make over a regular action head

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
- [ ] Add phase-specific close / release logic and an explicit placement tail

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

Interpretation:
- The loop semantics are now substantially cleaner:
  - no more forced gripper re-close during settling
  - reopen can be held until the TCP is actually over the plate
- But there is still no single shared configuration above `1/3` at `3 cm`
- Different seeds prefer different replanning cadences:
  - `seed 1` is best under `replan4`
  - `seed 0` is best under `replan6`
  - `seed 2` now looks like the best remaining target for a phase-logic fix
