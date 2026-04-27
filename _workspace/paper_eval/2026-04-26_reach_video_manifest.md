# 2026-04-26 Reach Video Manifest

## Canonical Contract

These videos use the current paper-safe reach-only closed-loop contract:

- config: `configs/libero_dhbxr_qwen_fresh_actionxyz.yaml`
- checkpoint:
  `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/checkpoints/steps_400_pytorch_model.pt`
- task suite: `libero_spatial`
- task ID: `0`
- init: benchmark state with explicit bowl offset
- observation: raw LIBERO image orientation
- trajectory source: `direct_action`
- direct-action XYZ scale: `1.0,1.35,1.15`
- state history update rate: policy step
- reach-only stop: enabled
- max policy steps: `140`

## Videos

| Episode | Offset | Success step | Min dynamic bowl XY | Min dynamic bowl XYZ | IK failures | Video |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `-3 cm` | `80` | `23.1 mm` | `82.8 mm` | `0` | `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep0_offset-3.mp4` |
| `0` | `+3 cm` | `91` | `47.8 mm` | `70.9 mm` | `0` | `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep0_offset3.mp4` |
| `1` | `+3 cm` | `82` | `14.4 mm` | `79.4 mm` | `0` | `_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/reach_step400_calib_y135_z115_ep1_offset3.mp4` |

## Interpretation

Use these as visual evidence for robust non-zero-offset reaching, not for full
pick-place. They deliberately keep the gripper open and stop on reach success.
The corresponding 5-case sweep is:

`_workspace/starvla_runs/dhb_starvla_qwen_fresh_actionxyz_from2000/live_reach_step400_calib_y135_z115_5case_benchmark_raw_max140.json`

That sweep reports `5/5` reach success, average dynamic bowl XY `39.2 mm`, and
`0` IK failures.
