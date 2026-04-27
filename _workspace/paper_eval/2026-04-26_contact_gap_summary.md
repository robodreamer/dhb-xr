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

## Negative Branch

The pre-contact residual-only branch
`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionres_contactprecision_fromrelease600`
failed the corrected reach-preservation gate at `steps_400`: `2/5` reach
successes, average dynamic bowl XY `70.8 mm`, average dynamic bowl XYZ
`73.6 mm`, and `0` IK failures. It should not be used for full-task videos.

## Interpretation

The current failure is no longer basic arm motion, gripper polarity, missing
gripper labels, image orientation, or IK collapse. The policy needs a generic
contact-ready module that preserves the validated reach field while improving
the close pose to roughly `<=20 mm` dynamic bowl XY before the learned close
event, then continues with object transport and a learned reopen.

The most useful positive signal is the Y-scale `1.55` diagnostic: it brings the
full-task close pose to `25.5 mm` lateral error and the delayed threshold run
shows `17.1 mm` XY is reachable. The most important negative signal is that
the same improvements still do not produce release, so contact precision alone
will not be enough for a full manipulation claim.

## Next Claim Gate

Do not call a new branch promising unless it passes all of these checks:

1. Corrected 5-case reach gate remains `5/5` with `0` IK failures.
2. Full-task close occurs before step `120` with dynamic bowl XY `<=20 mm`.
3. Object motion after close exceeds the current `95.7 mm` diagnostic without
   increasing IK failures.
4. A learned reopen event occurs near the plate without object-specific gates.
