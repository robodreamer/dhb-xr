# Framework Validation Status

Date: 2026-04-17

## StarVLA base examples

- Released LIBERO checkpoint eval runs successfully and writes rollout videos.
- Quantized `8-bit` framework smokes now run on the laptop GPU for:
  - `QwenGR00T`
  - `QwenOFT`
- Quantized smoke details are in [_workspace/starvla_quantized_smoke_results.md](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_quantized_smoke_results.md).

## Harness

- Model server test passes:
  - `./.pixi-home/bin/pixi run vla-eval test -c configs/model_servers/dhbxr.yaml --server --timeout 180 -v`
- Real local benchmark run completes end-to-end:
  - `./.pixi-home/bin/pixi run vla-eval run --config configs/eval/smoke_test_dhb.yaml --no-docker --verbose`
- Result artifacts:
  - [results/harness/smoke/LIBEROBenchmark_libero_spatial_smoke_sync_1776447588.json](/home/andypark/Projects/repos/dhb-xr/results/harness/smoke/LIBEROBenchmark_libero_spatial_smoke_sync_1776447588.json)
  - [results/harness/smoke/LIBEROBenchmark_libero_spatial_smoke_sync_1776447678.json](/home/andypark/Projects/repos/dhb-xr/results/harness/smoke/LIBEROBenchmark_libero_spatial_smoke_sync_1776447678.json)

## RoboCasa

- Base import and task registration work in `/tmp/robocasa_venv`.
- A local smoke recorder was added at [_workspace/robocasa_smoke_video.py](/home/andypark/Projects/repos/dhb-xr/_workspace/robocasa_smoke_video.py).
- Current proof-of-life video:
  - [_workspace/starvla_example_videos/robocasa_smoke/gr1_lift_random.mp4](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_example_videos/robocasa_smoke/gr1_lift_random.mp4)
- The smoke script uses local monkeypatches for current upstream wrapper mismatches:
  - unsupported `seed`
  - unsupported `translucent_robot`
  - `egoview` to `agentview` camera remap

## VLA-Arena

- Base registry import works in `/tmp/vla_arena_venv` with `PYTHONPATH=/home/andypark/Projects/repos/VLA-Arena`.
- A local smoke recorder was added at [_workspace/vla_arena_smoke_video.py](/home/andypark/Projects/repos/dhb-xr/_workspace/vla_arena_smoke_video.py).
- Current proof-of-life video:
  - [_workspace/starvla_example_videos/vla_arena_smoke/safety_dynamic_obstacles_task0.mp4](/home/andypark/Projects/repos/dhb-xr/_workspace/starvla_example_videos/vla_arena_smoke/safety_dynamic_obstacles_task0.mp4)

## Current conclusion

- StarVLA base examples: working
- `vla-evaluation-harness`: working
- RoboCasa: working for smoke rollout with local compatibility patches
- VLA-Arena: working for smoke rollout with local `PYTHONPATH` workaround

## Next practical step

- Move back to custom StarVLA+DHB smoke validation using these baseline checks as controls.
