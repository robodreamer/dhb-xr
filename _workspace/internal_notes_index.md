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

Historical diagnostics were moved to:

`_workspace/archive_notes/2026-04-historical/`

Do not treat archived notes as active guidance unless a current top-level note
explicitly references them. They are kept for provenance only.
