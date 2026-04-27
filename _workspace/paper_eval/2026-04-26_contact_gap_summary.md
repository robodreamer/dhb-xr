# 2026-04-26 Contact Gap Summary

## Purpose

This note consolidates the current full-task diagnostics after the robust
reach result. The goal is to keep the claim boundary precise: the system can
reach under offsets, but full pick-place still fails at contact precision,
post-contact transport, and learned release.

## Current Best Reach Evidence

The canonical reach evidence remains:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`

It reports `5/5` reach success, average dynamic bowl XY `39.2 mm`, and `0` IK
failures under benchmark init, raw LIBERO images, direct-action execution,
`1.0,1.35,1.15` XYZ scale, policy-rate state history, and a `140`-step closed
loop horizon.

## Full-Task Diagnostics

All rows below are task `0`, episode `1`, offset `+3 cm`, no object-specific
rollout gates.

| Branch | Gripper source/settings | Close step | Close EE-bowl offset | Min bowl XY | Min bowl XYZ | Bowl motion | IK fails | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Post-contact residual | gripper-state threshold `0.35` | `101` | `[13.9, -38.8, 25.2] mm` | `39.3 mm` | `43.2 mm` | `80.1 mm` | `16` | no reopen, no success |
| Release/gripper branch | threshold `0.35` | `65` | `[-42.6, -117.0, 114.8] mm` | `58.1 mm` | `70.2 mm` | `83.4 mm` | `0` | closes too early |
| Release/gripper branch | threshold `0.90` | `93` | `[1.0, -56.0, 44.9] mm` | `40.1 mm` | `43.5 mm` | `82.0 mm` | `14` | no reopen, no success |
| Release/gripper branch | contact-phase decoder | none | none | `39.2 mm` | `44.2 mm` | `64.7 mm` | `28` | no close |
| Release/gripper branch | Y-scale `1.55`, threshold `0.90` | `93` | `[1.5, -25.5, 44.7] mm` | `24.7 mm` | `43.4 mm` | `95.7 mm` | `22` | closer contact, no release |
| Release/gripper branch | Y-scale `1.55`, Z-scale `1.30`, threshold `0.90` | `85` | `[-11.5, -45.8, 36.3] mm` | `34.2 mm` | `39.3 mm` | `82.5 mm` | `24` | worse XY than Y-only |
| Release/gripper branch | Y-scale `1.55`, threshold `0.94` | `177` | `[30.4, -14.0, 21.7] mm` | `17.1 mm` | `30.2 mm` | `88.4 mm` | `36` | contact-ready but too late |
| Constrained contact residual | Y-scale `1.55`, threshold `0.90` | `101` | `[20.2, -9.4, 25.1] mm` | `18.9 mm` | `33.3 mm` | `98.5 mm` | `29` | best contact-ready diagnostic, no reopen |
| Constrained contact residual | Y-scale `1.55`, threshold `0.92` | `129` | `[35.4, 5.5, 14.2] mm` | `18.9 mm` | `32.8 mm` | `82.4 mm` | `39` | delayed close worsens IK |
| Contact-ready transport residual | Y-scale `1.55`, threshold `0.90` | `93` | `[7.0, -22.0, 42.5] mm` | `23.1 mm` | `44.2 mm` | `94.7 mm` | `28` | slight plate improvement, worse contact, no reopen |

## Negative Branch

The pre-contact residual-only branch
`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_fromrelease600`
failed the corrected reach-preservation gate at `steps_400`: `2/5` reach
successes, average dynamic bowl XY `70.8 mm`, average dynamic bowl XYZ
`73.6 mm`, and `0` IK failures. It should not be used for full-task videos.

The constrained cumulative-base branch
`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_cumsum_fromrelease600`
fixes that reach-preservation failure. It passes the corrected reach gate at
`steps_400` with `5/5` successes, average dynamic bowl XY `39.5 mm`, average
dynamic bowl XYZ `78.3 mm`, and `0` IK failures.

The closed-state transport branch
`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactready_transport_fromcumsum400`
also preserves the corrected reach gate at `steps_500`: `5/5` successes,
average dynamic bowl XY `40.1 mm`, average dynamic bowl XYZ `78.2 mm`, and
`0` IK failures. Its full-task diagnostic is neutral/negative: it slightly
improves plate proximity and IK count relative to the constrained contact
branch, but contact quality regresses to `23.1 mm` XY / `44.2 mm` 3D, it never
reopens, and it still does not complete placement. Do not continue the same
closed-residual demo-chunk training blindly.

## Interpretation

The current failure is no longer basic arm motion, gripper polarity, missing
gripper labels, image orientation, or IK collapse. The policy needs a generic
contact-ready module that preserves the validated reach field while improving
the close pose to roughly `<=20 mm` dynamic bowl XY before the learned close
event, then continues with object transport and a learned reopen.

The most useful positive signal is now the constrained residual branch with
Y-scale `1.55`: it reaches `18.9 mm` minimum dynamic bowl XY, closes at a
reasonable step with `22.3 mm` close-time XY, and moves the bowl `98.5 mm`.
The most important negative signal is that the same run still does not reopen,
keeps the bowl about `140 mm` XY from the plate at the end, and accumulates
late IK failures. Contact precision is now close to the target; post-contact
transport and learned release are the next blockers.

The closed-transport follow-up confirms this boundary rather than closing it.
Training only a physical-gripper-gated residual from the same demo chunks keeps
open-gripper reach intact, but it does not provide the missing post-contact
state coverage or release timing needed for success.

## Gripper Streaming Decoder Check

The gripper decoder now carries hysteresis/debounce state across replanned
chunks, which is the correct streaming behavior for VLA execution. Targeted
tests cover the case where transition evidence begins at the end of one chunk
and completes in the next.

Replaying the saved cumsum and contact-ready transport traces through the
fixed streaming decoder does not create a hidden reopen. The learned
gripper-state head emits a repeated slot-position pattern: high closed
probability in the first three executed slots of each chunk, then a sharp open
spike in the last executed slot. The next replan returns immediately to high
closed probability, so a real hysteresis decoder correctly treats this as an
isolated tail artifact rather than a stable release decision.

Interpretation: the no-reopen failure is not an object-specific gate bug. It is
a model/training issue in the learned release signal, likely caused by
chunk-relative gripper supervision or insufficient post-contact/release state
coverage. The next release attempt should train a release signal that is stable
across replans, not rely on lowering debounce to accept one-slot tail spikes.

## Command-State Transport Update

The latest root-cause check found a training/deployment mismatch in the
closed-residual gate. The closed transport residual was conditioned on the
observed physical gripper qpos, but during live LIBERO rollout the commanded
close is delayed in the observed qpos for dozens of policy steps. This meant
the transport residual often did not activate during the intended grasp and
post-grasp window.

Two generic fixes were added:

1. The rollout can feed a commanded gripper-state proxy into the policy state
   with `--state-gripper-source commanded`.
2. The dataset loader can train the state history the same way with
   `datasets.vla_data.state_gripper_source: commanded`.

This is not object-specific. It aligns the policy state with the command stream
used by a receding-horizon VLA controller, rather than with delayed actuator
qpos.

New branch results:

| Branch | State gripper | Close threshold | Close step | Close EE-bowl offset | Min bowl XY | Min plate XY | IK fails | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Release-stable | commanded | `0.60` | `93` | near contact, no reopen | `~19 mm` | `~86 mm` | `48` | improves IK timing, no success |
| Commanded transport | commanded | `0.45` | `109` | `24.4 mm` XY / `18.3 mm` Z | `19.2 mm` | `80.1 mm` | `40` | better IK, no placement |
| Commanded transport XY | commanded | `0.45` | `121` | `29.2 mm` XY / `15.3 mm` Z | `19.2 mm` | `88.2 mm` | `0` | stable post-close, no success |
| Commanded transport XY + closed calibration | commanded | `0.45` | `121` | `29.2 mm` XY / `15.3 mm` Z | `19.2 mm` | `81.7 mm` | `16` | diagnostic only, no success |
| Commanded contact-center | commanded | old gripper, `0.45` | `149` | late close after best contact | `7.2 mm` | `82.3 mm` | `46` | excellent contact diagnostic, too-late close |
| Contact-center + learned gripper heads | commanded | `0.47-0.55` | `103` | `18.0 mm` XY / `22.4 mm` Z | `8.2 mm` | `95.0 mm` | `0` | contact-close gate passed, no lift/placement |

The commanded transport branch preserves reach:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_transport_fromrelease600`
- checkpoint: `steps_800_pytorch_model.pt`
- corrected 5-case reach gate:
  `5/5`, average dynamic bowl XYZ `78.1 mm`, average dynamic bowl XY
  `40.3 mm`, and `0` IK failures

The XY-only closed residual branch then removes the late IK collapse in the
full-task diagnostic, but it still does not complete the task. The learned
close fires after the best approach point: the best EE-bowl XY is about
`19.2 mm` near step `97`, while the learned close fires at step `121` at about
`29.2 mm` XY. LIBERO bowl grasping appears to require roughly `<=15 mm` XY
contact precision, so the current blocker has narrowed to contact-centered
approach and close timing. More post-close transport tuning is lower leverage
until the close pose becomes physically graspable.

## Contact-Centered Gripper Timing Update

The contact-centered residual branch is the strongest contact diagnostic so
far:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_fromxy500`
- checkpoint: `steps_500_pytorch_model.pt`
- corrected 5-case reach gate:
  `5/5`, average dynamic bowl XY `37.3 mm`, average dynamic bowl XYZ
  `78.7 mm`, and `0` IK failures
- diagnostic scale:
  `--direct-action-xyz-scale 0.85 1.55 1.15`

With the older gripper head, the same branch reaches physically plausible bowl
contact before close: minimum dynamic bowl XY is `7.2 mm`, minimum dynamic
bowl XYZ is `21.0 mm`, and the trace stays around `7-10 mm` XY / `19-23 mm`
Z over steps `104-116`. However, the old close fires too late at step `149`,
after the contact window and after IK degradation begins.

The follow-up gripper-head continuation fixes that timing enough to pass the
current contact-close gate:

- run:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_commanded_contactcenter_grip_fromcenter500`
- checkpoint: `steps_800_pytorch_model.pt`
- state convention:
  commanded gripper proxy
- threshold/open/debounce:
  `0.47-0.55 / 0.05 / 2`
- close step:
  `103`
- close geometry:
  `18.0 mm` dynamic bowl XY and `22.4 mm` vertical offset
- minimum dynamic bowl geometry:
  `8.2 mm` XY and `25.4 mm` 3D
- IK failures:
  `0`

This is a meaningful gate change. The active blocker is no longer
contact-centered reaching or a gross gripper timing bug. The policy now closes
near plausible bowl contact without object-specific gripper gates, but it
pushes/slides the bowl instead of lifting and transporting it to the plate.
Final bowl-plate XY remains about `116.8 mm` in the best trace. The next
branch should focus on post-contact grasp/transport/release from this
contact-close checkpoint.

## Next Claim Gate

Do not call a new branch promising unless it passes all of these checks:

1. Corrected 5-case reach gate remains `5/5` with `0` IK failures.
2. Full-task close occurs before step `120` with dynamic bowl XY `<=20 mm`.
   The current contact-center gripper branch now passes this gate.
3. Object lift/transport after close moves the bowl toward the plate rather
   than pushing/sliding it away from a graspable pose.
4. A learned reopen event occurs near the plate without object-specific gates.
