# DHB-XR Internal Notes Index

Use these top-level notes as the current source of truth:

- `dhbxr_story_gap_closure_plan.md`: paper story, safe claims, and remaining
  evidence gaps.
- `dhbxr_next_10h_execution_plan.md`: execution order for the next active work
  block.
- `dhbxr_gripper_contract_audit_2026-04-28.md`: latest stop-and-audit note
  comparing our gripper/action-loop assumptions against OpenVLA, StarVLA, VLA
  Foundry, and the current best attached-transport trace.
- `dhbxr_objectpair_transport_status_2026-04-28.md`: current object-pair
  transport attempt status, including the 6-D policy-trace export, bounded
  continuation result, and the hard-stop diagnosis for the small 20-chunk
  attached-transport objective.
- `dhbxr_matched_head_comparison_v1.md`: current matched `QwenDHB` versus
  `QwenOFT` evidence and next comparison matrix.
- `dhbxr_unattended_benchmark_runbook.md`: commands and gates for benchmark
  runs.

Latest active result to preserve:

- Current strategy correction:
  `_workspace/dhbxr_gripper_contract_audit_2026-04-28.md` supersedes
  center-contact selection as the active full-task objective. The guarded
  attached-transport step-100 checkpoint is useful evidence for smooth reach
  and physical close; the physical close event actually matches the demo
  side-grasp affordance, while target-directed transport still fails. The next
  gate is physical close-vector affordance error, object/EE coupling,
  bowl-plate target progress, and release proximity, not minimum bowl-center XY
  alone.
- Latest object-pair transport result:
  `_workspace/dhbxr_objectpair_transport_status_2026-04-28.md` records that
  object-pair supervision is wired and trainable, but the current small
  policy-only continuation does not produce reliable target transport. Do not
  extend this exact 20-chunk warm-start loop; generate broader target-reaching
  attached positives or revisit fresh/full dataset training. The latest
  lift-transport audit fixed the missing-Z attached-positive labels, but the
  small two-manifest continuation still failed the no-gate placement gate, so
  broad regenerated augmentation plus a matched regular-head control is the
  active next step.
- Latest community-code check:
  VLA Foundry and TRI-LBM support explicit action/proprio/gripper field
  contracts, chunk past/future masks, and action-type-aware normalization as
  useful infrastructure ideas. They do not suggest runtime object gates or more
  scalar threshold sweeps as the route through the current post-contact
  transport failure.
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
- Latest 2026-04-28 audit:
  the guarded attached-transport continuation improves smooth center reach but
  should not be scaled blindly. Step `100` closes with `0` IK failures and
  physically closes the gripper (`observed qpos` about `0.034 -> 0.0069`).
  Command-close is near bowl center, but physical close is close to the demo
  side-grasp affordance: `4.6 mm` close-vector error against the first-three
  demo mean. Minimum bowl-center XY is therefore misleading, but the active
  failure is target-directed transport/release after close: no reopen, best
  bowl-plate XY about `125 mm`, final about `130 mm`. Use physical
  close-vector pass, observed qpos closure, post-close object/EE coupling,
  target progress, and release proximity as the next gate. Detailed note:
  `_workspace/dhbxr_gripper_contract_audit_2026-04-28.md`.
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
- Latest current-prefix result:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentprefix.yaml`
  is negative. It keeps IK stable and reaches the bowl (`0.34 mm` minimum bowl
  XY), but dense gripper-state decoding still closes at step `80` with
  `60.2 mm` close-vector error and `112.1 mm` final target XY. Contact-phase
  decoding closes at step `74` and never reopens.
- Latest currentready result:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready.yaml`
  is the new best learned no-gate close-readiness evidence. The object-free
  `contact_gripper_state` decoder closes from the current contact-readiness
  head and reopens from the future dense gripper-state head. It closes at step
  `103`, has `0` IK failures, reaches `10.2 mm` minimum bowl XY, and scores
  `37.1 mm` close-vector error with `19.0 mm` close XY. It never reopens and
  leaves `118.0 mm` final target XY, so the remaining active blocker is
  post-close transport/release.
- Latest evaluator correction:
  `scripts/starvla_dhb_embodik_live_rollout.py` now supports
  `--rollout-seed` and records `config_seed` plus `rollout_seed`. Use this for
  all future claim-level rollouts. Previous currentready/currentready-release
  single-run comparisons were partially seed-confounded because config seeds
  also controlled the LIBERO environment. Under controlled `rollout_seed=70`,
  currentready `close=0.55/open=0.40` closes at step `91`, no reopen, `0` IK
  failures, actual close-vector error `36.4 mm`, and best possible
  close-affordance error `3.3 mm` at step `110`. `close=0.58` delays close to
  step `101` but reopens early at `121`. The state-only release continuation
  changes only `gripper_state_head`; under the same seed it closes at `87`,
  never reopens, and finishes `129 mm` from the plate. This confirms the
  active blocker is learned phase selection/release under a controlled
  benchmark, not trajectory reachability.
- Latest currentready transport-release branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_transport_release.yaml`
  freezes the currentready close-readiness head and trains only
  `action_closed_residual_adapter` plus `gripper_state_head` (`4.464M`
  trainable). Step `200` is the useful partial: close step `103`, reopen step
  `221`, `0` IK failures, `37.1 mm` close-vector error, and `114.4 mm` final
  target XY under `rollout_seed=70`. Step `400` overtrains into close/open
  flicker (`85 -> 89`) and `12` IK failures. The next branch should strengthen
  held-object transport and phase persistence, not extend this weighting.
- Latest action-only closed-transport branch:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_closedtransport_actiononly.yaml`
  starts from the useful transport-release `steps_200` partial and trains only
  `action_closed_residual_adapter` (`0.267M` trainable). It is negative:
  step `150` never reopens and ends `123.9 mm` from target; step `300` closes
  at `103`, reopens at `237`, keeps `0` IK failures, but final target XY is
  still `112.9 mm`. This rules out pure closed-residual action fitting as the
  next breakthrough. Use offline object-level transport/release supervision or
  post-contact relabeling next, while keeping rollout learned/no-gate.
- Latest object-geometry transport status:
  `/home/andypark/Projects/data/libero_lerobot_affordance_transport` now
  includes object-to-EE, target-to-EE, object-minus-target, and transport-phase
  labels. `QwenDHB` can train configurable object-target heads from one or more
  fields and live rollouts now record `pred_object_target` diagnostics. The
  simple object-minus-target branch and the frozen-feature object-pair branch
  both collapsed to near-constant live geometry predictions and did not improve
  placement. The separate cross-attention object-pair readout produces varying
  predictions, but still fails: step `150` flickers (`87 -> 89`), and step
  `300` closes at `103` with no release and `117.9 mm` final target XY. Treat
  these as negative result/provenance. The next active branch should train a
  learned release-ready signal and hold persistence from the same offline
  geometry rather than adding more scalar object-vector context.
- Latest object-target release branch:
  the converter now exports `dhb.object_release_*` and
  `dhb.object_hold_transport` labels from generic object-minus-target geometry.
  `QwenDHB` can route phase/gripper supervision to configured fields, and the
  new active config is
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease.yaml`.
  The refreshed dataset has `500` episodes / `62,250` frames, and the targeted
  regression suite plus one-step StarVLA smoke pass. The first evaluation
  exposed a full-task horizon bug: `--max-policy-steps 0` capped the
  episode-1 offset rollout at `84` steps before known delayed close events.
  The rollout scripts now warn/record `demo_length_cap_warning`; use
  `--max-policy-steps 240` for full-task offset claims.
  Under the corrected fixed `rollout_seed=70`, true-no-gate protocol, step
  `300` closes at `82`, reopens at `153`, and reaches `1.8 mm` minimum bowl
  XY, but still has `51.7 mm` close-vector error, `121.1 mm` reopen target XY,
  `116.1 mm` final target XY, and `11` IK failures. A generic
  `close=0.60/open=0.60/debounce=1` diagnostic improves actual close to
  `37.9 mm` close-vector error and `12.8 mm` close XY, but final target XY
  remains `110.6 mm` and IK failures rise to `64`. Direct `contact_phase`
  decoding is worse. Treat this as a validated no-gate learned close/reopen
  partial, not a solved full-task result.
- Latest current-phase release decoder:
  `src/dhb_xr/integration/vla/libero.py` now has
  `contact_release_phase`, which closes from the currentready contact head and
  opens from the learned object-release phase class. The current-phase
  continuation config is
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_currentready_objectrelease_currentphase.yaml`.
  Under fixed `rollout_seed=70`, true no-gate, h240, step `300` reaches
  `6.4 mm` minimum bowl XY with `close=0.55`, but closes too early/high
  (`49.1 mm` close-vector error), never reopens, and finishes `111.8 mm` from
  the target. Raising close threshold to `0.60` improves close geometry
  (`16.8 mm` XY, `35.8 mm` vector error) but worsens transport and IK. A trace
  shows release probability remains near zero while the bowl is still far from
  the plate, which is conceptually correct. The active blocker is now
  close-affordance precision plus held-object transport, not release-class
  plumbing.
- Latest close-contact recalibration negative:
  StarVLA BCE contact loss now honors `contact_pos_weight`, and
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_objectrelease.yaml`
  trains only `contact_head` from current-aligned `dhb.close_contact`. The
  training objective collapsed cleanly, but rollout collapsed to immediate
  close at step `1` for both `steps_150` and `steps_300`. Root cause is the
  recipe, not the deployment concept: close-positive sampling and high
  positive weight made the classifier always-close. Future close-affordance
  work needs balanced hard negatives and calibration, not positive-only
  close-window fitting.
- Latest balanced close-contact follow-up:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_balanced_objectrelease.yaml`
  avoids the immediate-close collapse but under-calibrates the learned close
  probability. At `close=0.55`, `steps_150` and `steps_300` never close despite
  reaching about `6.8 mm` minimum bowl XY; both accumulate `88` IK failures.
  A diagnostic `close=0.15` on `steps_150` closes at step `87` with `0` IK
  failures, but only scores `40.0 mm` close-vector error, `42.7 mm` close XY,
  no reopen, and `126.8 mm` final target XY. The contact-cross-attention
  variant,
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_xattn_objectrelease.yaml`,
  trains `7.347M` parameters but suppresses close confidence further and never
  closes at the default threshold. Conclusion: BCE close-contact retraining is
  bracketed and should stop. The next high-leverage path is rollout-state hard
  negatives or a pairwise/margin close-readiness objective that explicitly
  contrasts true side-grasp-ready states against near-bowl-but-wrong states.
- Latest contact mining/margin follow-up:
  `scripts/libero_rollout_contact_miner.py` now mines recorded traces for
  close-affordance positives and hard negatives. The currentready/current-phase
  trace has many hard negatives (`170`) and true positives (`45`), so the best
  partial is over-confident at wrong closed-loop states rather than simply
  under-confident near the bowl. A generic `lambda_contact_margin` path was
  added and tested via
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_margin_objectrelease.yaml`,
  but the retrain suppresses close entirely at the default threshold. Treat
  demo-only contact-head recalibration as exhausted. Next: collect/replay
  rollout-state observations and train against those hard negatives directly.
  The rollout harness now has policy-trace export (`--save-policy-trace-dir`
  and sweep `--save-policy-trace`) so the next dataset can use actual
  closed-loop observation/state tensors rather than geometry-only summaries.
  One policy-trace smoke artifact was generated for the currentready/current-
  phase `steps_300` checkpoint, `episode=0`, `offset=0cm`, fixed
  `rollout_seed=70`; it produced `60` NPZ chunks / `8.6M` total and mines to
  `191` hard negatives. The contact miner now emits a policy-manifest JSONL
  with `target_close`, `train_mask`, and `hard_negative_mask`, which is the
  next bridge into a closed-loop correction dataloader. A matching non-zero
  offset trace for `episode=1`, `offset=3cm` produced another `60` chunks and
  `171` hard negatives.
- Latest policy-trace hard-negative dataloader follow-up:
  StarVLA now has `policy_trace_contact_datasets`, which mixes LeRobot demo
  samples with saved closed-loop policy-trace NPZ chunks and their mined
  `dhb.close_contact`/`dhb.action_valid_mask` labels. The new config is
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_policytrace_hardneg.yaml`.
  Unit tests, action-head/contact-miner regressions, and an actual QwenDHB
  train-step smoke pass. The 300-step continuation is a controlled negative:
  both `steps_150` and `steps_300` never close on fixed-seed true-no-gate
  `0 cm`/`3 cm` h240 sweeps. Max close probability is suppressed to about
  `0.15` at step 150 and below `0.03` at step 300. This proves the loader
  works but the data is one-sided: current manifests contain `242`
  hard-negative supervised steps and `0` closed-loop positive steps. Next
  breakthrough work should generate positive closed-loop contact states through
  successful/near-success traces or simulator replay around demo affordance
  states, then reuse this dataloader with a balanced rollout-state set.

Historical diagnostics were moved to:

`_workspace/archive_notes/2026-04-historical/`

Do not treat archived notes as active guidance unless a current top-level note
explicitly references them. They are kept for provenance only.

2026-04-27 active contact-token status:

- Root cause fixed: future close labels were being erased by
  `contact_target_alignment: current`. The active balanced contact config now
  uses `future` alignment.
- New active files:
  `scripts/libero_demo_contact_manifest_export.py` and
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_demopos_policytrace_balanced.yaml`.
- Corrected run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_demopos_policytrace_balanced_futurealign_fromcurrentphase300`.
- Current best evaluation profile: `trajectory_source=direct_action`,
  `direct_action_xyz_scale=(0.85, 1.55, 1.15)`,
  `closed_direct_action_xyz_scale=(0.85, 1.55, 1.15)`,
  `model_gripper_decode_source=contact_release_phase`, true no-gate,
  `state_history_update_rate=policy`, `state_gripper_source=commanded`.
- Current evidence: calibrated reach is strong (`6.9 mm` XY on the `3 cm`
  offset case), learned close is active but under-calibrated (`0.45` threshold
  needed), and full placement is still unsolved.
- Next active gap: generate balanced rollout-positive contact states and then
  revisit post-grasp transport/release. Do not evaluate main examples with
  neutral direct-action scaling unless explicitly testing calibration.

2026-04-27 rollout-positive/shuffled contact-token status:

- New active config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_rolloutpos_tol45_balanced.yaml`.
- New active run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_rolloutposfirst_shuffle_tol45_fromfuturealign300`.
- StarVLA policy-trace contact datasets now honor
  `policy_trace_shuffle: true`; this fixed short continuation runs that could
  miss appended rollout-positive rows.
- `steps_400` makes the learned contact token active under true no-gate
  deployment with calibrated direct-action scale `0.85, 1.55, 1.15`.
  Threshold `0.55` gives the current best non-zero-offset close snapshot:
  about `6.8 mm` XY and `19.3 mm` Z from the bowl, but no full pick-place
  success.
- Current interpretation: the learned token path works, but final results need
  stricter side-grasp-ready offline positives plus near-bowl/high-or-center
  hard negatives, followed by post-grasp transport/release training. The
  deployment policy must remain learned-token/no-gate.

2026-04-27 side-grasp calibration status:

- `scripts/libero_rollout_contact_miner.py` now supports component-wise
  positive tolerances and near-object hard negatives. The first strict
  side-grasp manifest
  `_workspace/paper_eval/2026-04-27_policy_trace_sidegrasp_v1_contact_manifest.jsonl`
  has `9` positives and `601` hard negatives.
- Strict continuation config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1.yaml`.
  The strict checkpoint suppresses live close confidence and never closes at
  threshold `0.55` under true no-gate deployment.
- A lighter side-grasp calibration run restores no-gate close activation but
  still fails grasp/transport/release. Repro config:
  `configs/libero_dhbxr_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegraspv1_light.yaml`.
  Treat this as evidence that the next
  breakthrough requires balanced side-grasp-positive rollout data, not another
  deployment heuristic or scalar threshold sweep.
- Latest physical-close correction:
  `scripts/libero_rollout_affordance_score.py` now scores physical close using
  `observed_state_gripper <= 0.015` when available, while still reporting the
  command close score. This fixes the mismatch between rollout command timing
  and the demo `close_phys` reference.
- Latest best learned-token contact result:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegrasp_bestposv2_fromrolloutpos400`,
  `steps_200`, threshold `0.50`, episode `1,+3 cm`, true no-gate deployment.
  Command close is step `99`, physical close is step `101`, and physical
  side-grasp vector error is `9.2 mm`. Video:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_closecontact_sidegrasp_bestposv2_fromrolloutpos400/bestposv2_thr050_ep1_offset3.mp4`.
  Full pick-place is still unsolved because zero-offset contact is high and
  post-close transport/release fails.
- Active closed-transport follow-up:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_bestposv2_closedtransport_actiononly_frombestpos200`,
  `steps_300`, threshold `0.50`, fixed cases `0:0` and `1:3`, true no-gate
  deployment. This checkpoint keeps the non-zero-offset physical side-grasp
  pass (`10.6 mm` vector error), closes in both fixed cases, and has zero IK
  failures, but still never reopens/releases. Treat it as the smoother current
  contact/approach checkpoint, not as a solved full-task result.
- Release-aware diagnostic:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_bestposv2_releaseaware_fromclosedtransport300`.
  The run proves learned reopen can activate without deployment object gates,
  but the reopen is still far from target (`127-134 mm` XY) and raises IK
  failures. Model weights were removed after evaluation; keep this as evidence
  that release needs rollout-state positives and closed-far hard negatives.
- New release mining tool:
  `scripts/libero_rollout_release_miner.py`. Strict mining of fresh
  closed-transport policy traces produced `0` release positives and `270`
  closed-far hard negatives. Use it next as conservative hard-negative
  calibration mixed with demo release positives; do not treat the current
  rollout states as release positives because best target XY is still above
  `100 mm`.
- Release hard-negative calibration run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_releasehardneg_gripperstate_fromclosedtransport300`.
  `steps_200` delays reopen and preserves the `3 cm` physical side-grasp pass
  (`10.6 mm` vector error), but still reopens over `100 mm` from the plate.
  Treat this as diagnostic. The active baseline remains closed-transport
  `steps_300`; the next data gap is post-close transport-to-target positives
  before release positives can be mined honestly.
- Object-transport conditioning diagnostic:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_affordancegrasp_objecttransport_fromclosedtransport300`.
  `steps_300` preserves the `3 cm` physical close pass (`9.4 mm` vector error)
  but still reopens at `136.5 mm` target XY with `34` IK failures. Treat this
  as evidence that nominal demo-side object-target supervision is insufficient;
  the next breakthrough needs generated or replayed post-close target-reaching
  positives. Transport mining shows only `10.4 mm` target progress at `0 cm`
  and `6.5 mm` at `3 cm`.
- Transport correction miner:
  `scripts/libero_rollout_transport_miner.py`. On active closed-transport
  `steps_300`, post-close target progress is only about `10 mm` in both fixed
  cases, with no target-reached steps under a `45 mm` window. Correction
  requests are exported locally at
  `_workspace/paper_eval/2026-04-27_closedtransport_transport_correction_requests.jsonl`.
  Use these requests as the interface for simulator/world-model post-contact
  augmentation.
- Trainable transport correction status:
  `scripts/libero_rollout_transport_miner.py` now also exports policy-trace
  manifests with normalized corrective actions, object-minus-target targets,
  and closed gripper-state targets; StarVLA `PolicyTraceContactDataset`
  consumes all three fields. Aggressive v1 labels caused flicker/IK regression.
  Conservative v2 labels with dense gripper-state supervision fixed that
  mechanical issue at `steps_200`, but still failed object transport and
  weakened `0 cm` close. Treat both as negative diagnostics; the non-best
  weights were deleted after evaluation. Next step is successful post-contact
  target-reaching positive generation, not another scalar warm-start sweep.
- Post-contact reset/replay status:
  Live rollouts and checkpoint sweeps now support `--record-sim-state-trace`.
  `scripts/libero_postcontact_correction_plan.py` creates reset/replay plans
  from mined correction requests, and
  `scripts/libero_postcontact_correction_rollout.py` restores captured LIBERO
  states and executes closed-gripper correction actions. The transport miner now
  requires command-closed plus observed-closed candidates and reports object/EE
  coupling. First replay attempts were negative: earliest `+3 cm` reset states
  give `0.0 mm` target progress, and a later reset gives only `2.6 mm`. The
  next gap is verified grasp-attached state generation/selection, not more
  direction-label tuning.
- Coupling-aware replay status:
  Transport mining now supports local coupling-aware correction selection with
  minimum EE/object motion thresholds. Coupled `+3 cm` starts improve target XY
  by about `26-33 mm` under replay, versus `0 mm` for earliest physical-close
  starts, but still stall around `97-100 mm` and do not create release/placement
  positives. Use coupling-aware mining for diagnostics; the next breakthrough
  still requires simulator-assisted grasp-attached target-reaching positives.
- Disk cleanup status:
  On 2026-04-28, superseded model weights were removed from old exploratory
  runs while preserving configs, logs, summaries, notes, and selected active
  checkpoints. Free disk increased from `43G` to `712G`. Cleanup manifests are
  local at `_workspace/disk_cleanup_2026-04-28_weights_tranche1.md` and
  `_workspace/disk_cleanup_2026-04-28_weights_tranche2.md`.
- Attached transport positive status:
  `scripts/libero_postcontact_correction_rollout.py` now has an offline
  `--attach-object-to-ee` augmentation mode and can export policy-trace NPZ
  chunks plus manifest rows. The first attached `+3 cm` coupling-selected replay
  reaches the `45 mm` target window in `5/5` rows with zero IK failures and
  about `87-88 mm` target XY progress. Trainable manifest:
  `_workspace/paper_eval/2026-04-28_attached_transport_policy_manifest_ep1_offset3.jsonl`
  (`30` rows). Next step is a narrow closed-transport continuation from active
  closed-transport `steps_300`; keep runtime deployment object-gate-free.
