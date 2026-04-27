# DHB-XR Internal Notes Index

Use these top-level notes as the current source of truth:

- `dhbxr_story_gap_closure_plan.md`: paper story, safe claims, and remaining
  evidence gaps.
- `dhbxr_next_10h_execution_plan.md`: execution order for the next active work
  block.
- `dhbxr_matched_head_comparison_v1.md`: current matched `QwenDHB` versus
  `QwenOFT` evidence and next comparison matrix.
- `dhbxr_unattended_benchmark_runbook.md`: commands and gates for benchmark
  runs.

Latest active result to preserve:

- `dhb_starvla_qwen_fresh_actionxyz_from2000/steps_400` with
  `--trajectory-source direct_action` and
  `--direct-action-xyz-scale 1.0 1.35 1.15` reaches `5/5` across
  `0,+3,-3 cm` offset diagnostics with `0` IK failures under the current
  benchmark-init protocol when the rollout uses raw LIBERO images and a
  longer closed-loop cap (`--max-policy-steps 140`). Current artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`.
  The average minimum dynamic bowl XY is `39.2 mm`; delayed +3 cm cases reach
  after the original demo horizon, so non-zero-offset evaluation must not cap
  execution at the demo length.
- Older `live_reach_step400_calib_y135_z115_offsets.json` and the first
  residual reach artifacts used demo init and/or stale evaluator settings.
  Keep them for provenance, but do not use them as the paper-safe benchmark
  protocol.
- Representative successful non-zero-offset video:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3.mp4`.
- Negative branch to preserve:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionfull_grip_from400`
  did not preserve reach after joint full-action/gripper fine-tuning. The
  two-case reach sweep stayed at `0/2` for all tested checkpoints; do not use
  it for paper videos.
- Active follow-up:
  `configs/libero_dhbxr_qwen_fresh_gripper_heads_only.yaml` freezes the
  validated reach/action modules and trains only generic learned
  contact/gripper heads from demo labels.
- New diagnostic follow-up:
  `configs/libero_dhbxr_qwen_fresh_gripper_heads_calibrated.yaml` trained only
  the generic gripper/contact heads to step `2000`. It improved close
  probability calibration but did not solve full pick-place. Best diagnostic
  contact run reached `min_bowl_xyz=22.1 mm` while closed, but still failed,
  so the remaining blocker is contact-quality grasp/transport rather than
  gripper command polarity or missing gripper labels.
- New negative branch:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionhead_contact_from400`
  trained only the shallow direct-action/contact/gripper heads from the
  validated reach checkpoint. It failed reach preservation (`0/2`, average
  minimum bowl XY about `23 cm`), so do not continue direct-action-head
  fine-tuning without a reach-preserving residual/adaptor design.
- Residual branch diagnostic:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400`
  trains a zero-initialized additive residual adapter while freezing the
  original direct-action head. The earlier `2/2` reach gate was measured under
  an easier demo-init/raw-image protocol and should be treated as diagnostic,
  not paper-safe. Under the current benchmark-init/raw-image/full-horizon
  protocol, the full task still fails because the learned close signal fires
  early on the delayed +3 cm rollout: close at step `75`, about `100 mm` XY
  from the dynamic bowl. Current artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/full_task_step400_benchmark_raw_y135_z135_thr0395_open020_deb2_ep1_offset3_model_gripper_nogates_trace.json`.
  A diagnostic-only object-distance gate delays close to step `165`, but still
  fails transport/release (`min_bowl_xyz=53.2 mm`, final bowl-plate XY about
  `128 mm`), so the remaining gap is not just threshold timing. Gated artifact:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/full_task_step400_benchmark_raw_y135_z135_thr0395_open020_deb2_ep1_offset3_model_gripper_gated_trace.json`.
- New negative continuation:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_fromres400`
  keeps the residual reach behavior (`2/2` in the two-case gate) but worsens
  gripper timing. The no-gate full-task diagnostic closes at step `45`, around
  `116 mm` XY and `147 mm` Z away from the bowl. Do not use this continuation
  for claims; it shows that more gripper/contact class weighting alone creates
  spurious early close peaks.
- Active diagnostic improvement:
  `scripts/starvla_dhb_embodik_live_rollout.py --record-step-trace` now stores
  per-step dynamic EE/object snapshots in JSON. Use it to distinguish contact
  reach from actual object motion/lift before choosing the next contact-quality
  training target.
- Latest traced result:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contact_from400/full_task_step400_calib_y135_z135_thr0395_open020_deb2_ep1_offset3_model_gripper_nogates_trace.json`
  shows the residual branch can move/lift the bowl (`64.5 mm` total bowl
  displacement, `37.5 mm` lift) but fails post-contact placement/release
  (`127.7 mm` final bowl-plate XY offset, gripper still closed). The remaining
  full-task blocker should be treated as post-contact transport/release, not
  gripper polarity or total grasp absence.
- New negative transport branch:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_transport_fromres400`
  preserves reach (`2/2` at both `steps_300` and `steps_600`, `0` IK failures)
  but suppresses learned close probabilities. Full-task traces at threshold
  `0.395` never close (`max close prob 0.282` at `steps_300`, `0.346` at
  `steps_600`) and only push the bowl about `53 mm` with `13-14 mm` lift. Do
  not continue this scalar transport fine-tune for claims; the next model
  change should separate post-contact transport adaptation from the pre-contact
  close signal.
- Latest implementation fix:
  command-state gripper conditioning is now supported in rollout and dataset
  loading. Use `--state-gripper-source commanded` and
  `datasets.vla_data.state_gripper_source: commanded` for branches that train
  or execute closed-residual behavior. This avoids the observed LIBERO qpos lag
  disabling closed residuals after a close command.
- Latest branch status:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_transport_fromrelease600`
  preserves the corrected 5-case reach gate at `5/5`, average dynamic bowl XY
  `40.3 mm`, and `0` IK failures. The follow-up XY closed residual branch
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_transport_xy_fromcommand800`
  removes full-task IK failures in the best diagnostic, but still closes at
  about `29.2 mm` XY from the bowl after a best approach of about `19.2 mm`.
  Treat the remaining gap as contact-centered approach and learned close timing
  before returning to release/placement.
- Latest contact-close status:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_fromxy500`
  improves the contact-centered action field while preserving the corrected
  5-case reach gate: `5/5`, average dynamic bowl XY `37.3 mm`, average dynamic
  bowl XYZ `78.7 mm`, and `0` IK failures. The follow-up gripper timing branch
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_grip_fromcenter500`
  reaches the current no-gate contact-close target: close at step `103`, close
  dynamic bowl XY `18.0 mm`, minimum dynamic bowl XY `8.2 mm`, and `0` IK
  failures. It still fails full pick-place because the bowl is pushed/slid
  rather than lifted and transported. The active remaining gap is
  post-contact transport/release from this contact-close checkpoint.
- Latest root cause:
  the task `0` bowl grasp is a side-grasp affordance, not a center-contact
  event. `scripts/libero_demo_event_geometry_probe.py` reports physical close
  over 50 demos at mean EE-minus-bowl `[-1.5, -49.9, +26.3] mm` with
  `52.1 mm` lateral offset. The current post-contact branch closes near bowl
  center (`[4.5, -5.3, 19.6] mm` in the best trace), which explains why it
  pushes/slides the bowl instead of lifting it. Replace center-close gates with
  affordance-relative grasp/release diagnostics and offline supervision.
- Latest diagnostic tool:
  `scripts/libero_rollout_affordance_score.py` scores rollout traces against
  the demo close/release event geometry. The current score artifact
  `_workspace/paper_eval/2026-04-26_affordance_scores_postcontact.json` shows
  that center-contact branches fail the side-grasp close-vector gate, while a
  diagnostic `-0.3 mm/step` Y bias can pass close affordance but still fails
  object-target placement. Use this scorer as the full-task gate for future
  branches and avoid more scalar execution-bias sweeps.
- Latest dataset/training path:
  `/home/andypark/Projects/data/libero_lerobot_affordance` was regenerated
  with demo-affordance close labels from simulator replay. It has `500`
  episodes and `62,250` frames. Use StarVLA data mix
  `libero_spatial_dhbxr_affordance` and configs
  `libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp.yaml` then
  `libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_grip.yaml` for
  the next branch.
- Latest 2026-04-27 branch status:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_fromxy500`
  preserves the corrected 5-case reach gate at `5/5`, average dynamic bowl XY
  `38.2 mm`, and `0` IK failures. The gripper-only continuation
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_grip_fromaff500`
  makes the learned gripper-state signal actionable without object-distance
  gates: `steps_800`, threshold `0.55`, closes at step `83` with `0` IK
  failures and minimum bowl XY `6.7 mm`. It still misses the demo side-grasp
  affordance (`51.8 mm` close-vector error; best possible `25.0 mm` at step
  `113`), never reopens, and ends about `116 mm` bowl-plate XY from target.
  A generic threshold `0.70` delays close to step `87` and improves close
  vector error to `42.3 mm`, but still fails. Demo gripper replay is much
  worse (`194.8 mm` close-vector error, `112` IK failures). Current summary:
  `_workspace/paper_eval/2026-04-27_affordance_gripper_retrain_summary.md`.
- Latest implementation fix:
  3-class contact-phase event decoding now respects configured close/open
  thresholds instead of toggling by argmax alone. The fix prevents weak
  close/release event logits from becoming hard gripper commands, and is
  covered in `tests/test_libero_orientation.py`. The current trained
  contact-phase head still closes too early after the fix, so the model needs
  better event calibration rather than another rollout heuristic.
- Latest event-BCE result:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_eventbce.yaml`
  trains thresholded close/open contact-phase logits from the gripper step-800
  checkpoint, with the action path and dense gripper-state head frozen. It
  trains cleanly, but no-gate contact-phase rollout is still negative:
  threshold `0.55` closes at step `3` with `339.3 mm` close-vector error;
  thresholds `0.80-0.90` delay close to steps `74-75` with `76-80 mm`
  close-vector error, `0` IK failures, and no release. This is now evidence
  that the event head is chunk-time-biased, not just decoder-bugged. Debug
  traces now also record the event close/open thresholds per chunk; see
  `tests/test_chunk_rollout.py`.
- Latest release-focused branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip.yaml`
  trains only generic contact/gripper heads from the affordance-gripper
  `steps_800` checkpoint with release-heavy sampling. It confirms that no-gate
  reopen is learnable (`steps_300`, threshold `0.55`, closes at `137` and
  reopens at `165`), but the close is still center-contact (`54.4 mm`
  close-vector error), release target XY is `119.9 mm`, and IK failures rise
  to `124`. Lowering close threshold to `0.45` improves close-vector error to
  `41.9 mm` but causes the close/open collapse (`87 -> 89`). Step `600`
  restores `0` IK failures but loses reopen. Treat this branch as evidence that
  the next model needs phase persistence plus post-contact transport coupling.
- Latest smooth gripper-state branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_releasegrip_smooth.yaml`
  adds a target-aware smoothness loss on dense gripper-state probabilities. It
  mechanically fixes the close/open flicker failure and preserves `0` IK
  failures, with step `200-400` rollouts reaching `0.2-0.6 mm` minimum bowl XY.
  It still does not reopen, and close-vector error remains `46.8-49.0 mm`.
  Keep this loss as a useful stabilizer, but do not treat the first weighting
  as a solved release policy. The next branch should couple side-affordance
  close, hold/transport persistence, and object-target release instead of only
  increasing release sampling or scalar thresholds.
- Latest coupled post-contact branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_coupled_postcontact.yaml`
  trains the closed-state residual and generic contact/gripper heads jointly
  from the affordance-gripper `steps_800` checkpoint. The gated diagnostic at
  step `600` improves close-vector error to `36.0 mm` with `0` IK failures, but
  still never reopens. The sweep harness was also fixed because it previously
  failed to forward `--disable-gripper-gates`; files labeled `nogates` from the
  sweep path may therefore have evaluator object gates enabled unless their JSON
  has `"disable_gripper_gates": true`. With true gates disabled and generic
  `0.55/0.40` gripper-state hysteresis, the branch closes at `81` and reopens at
  `93`, before transport (`57.6 mm` close-vector error, `129.2 mm` release
  target XY). This is the clearest remaining blocker: learned phase persistence
  after close, not missing release labels or gripper polarity.
- Latest phase-persistence branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_phasepersistent.yaml`
  adds dense gripper-state margin supervision. It is the current best
  learned-gripper sequence under true no-gate evaluation: close at step `80`,
  reopen at step `197`, `0` IK failures, and `2.3 mm` minimum bowl XY. It still
  fails full pick-place because close happens before physical side-grasp
  contact (`60.2 mm` close-vector error) and release remains far from the plate
  (`118.1 mm` bowl-plate XY). Raising close threshold to `0.85` only moves
  close to step `83`; contact-phase decoding closes at step `2`.
  A trace-scored rerun strengthens the diagnosis: actual close at step `79`
  has `63.2 mm` close-vector error, but the same no-gate trajectory reaches a
  `26.1 mm` best close-affordance opportunity at step `123` with `0` IK
  failures. The motion is close enough; the learned gripper/contact-ready
  calibration fires too early.
- Latest negative calibration branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_precontactopen.yaml`
  oversamples close labels at the horizon tail to teach open-before-contact
  negatives. It improves close-vector error to `42.1 mm` but immediately
  reopens (`87 -> 89`) and introduces `72` IK failures. Its heavy checkpoint
  and final-model artifacts were deleted to recover disk; keep the config and
  score summaries as a negative result. The next model change should add a
  learned contact-ready supervision path, not more threshold/sampler-only
  sweeps.
- Latest sparse contact-event branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_contactevent.yaml`
  trains the existing generic `contact_head` with 3-way sparse event labels and
  evaluates via the new sweep flag `--model-gripper-decode-source contact`. It
  is negative: close at step `3`, no reopen, `339.3 mm` close-vector error, and
  `160.3 mm` minimum bowl XY. Heavy artifacts were deleted; summaries/scores
  remain. This rules out simply switching to the existing sparse event head.
  The next breakthrough needs contact-ready supervision/calibration tied to
  current visual/proprioceptive state.
- Latest current-alignment implementation:
  `vendor/starVLA/starVLA/model/framework/VLM4A/QwenDHB.py` now supports
  `contact_target_alignment`, `contact_phase_target_alignment`, and
  `gripper_state_target_alignment` with default `future` behavior and optional
  `current` broadcast behavior. Coverage lives in
  `tests/test_qwendhb_target_alignment.py`. This is for training target
  alignment only; deployment still uses learned predictions without
  object-distance gates.
- Latest current-aligned gripper result:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentgrip.yaml`
  delays actual close to the bowl-contact region (`close_step=127`,
  `6.9 mm` minimum bowl XY, close snapshot about `[14.9, -3.1, 17.0] mm`
  EE-minus-bowl), but reopens two steps later and introduces `81` IK failures.
  Treat it as evidence that fully current-aligned dense gripper labels are too
  pulse-like for persistent hold/transport.
- Latest hybrid current-phase result:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentphase.yaml`
  keeps dense gripper-state labels future-aligned and current-aligns only
  contact/contact-phase targets. True no-gate rollout has close step `77`,
  reopen step `233`, `0` IK failures, `4.7 mm` minimum bowl XY, and `69.6 mm`
  actual close-vector error; the same trace reaches a `25.9 mm` best
  close-affordance opportunity at step `122`. Next target is a prefix-scoped
  current open/contact-ready margin plus future-aligned hold persistence, not
  another threshold sweep.

Historical diagnostics were moved to:

`_workspace/archive_notes/2026-04-historical/`

Do not treat archived notes as active guidance unless a current top-level note
explicitly references them. They are kept for provenance only.
