# StarVLA DHB Handoff Checklist

## SmolVLA Baseline To Beat

- Checkpoint: `_workspace/dhb_head_e2e_fix_smoke.pt`
- Evaluation path: `_workspace/eval_dhb_head_embodik.py`
- Validation protocol: offsets `0cm` and `3cm`, `3` seeds, recorded videos
- Current executable baseline:
  - `0cm`: `2/3`
  - `3cm`: `1/3`

## What Is Already Validated

- DHB state-forwarding bug is fixed.
- Persistent EE rotation / orientation runaway is gone.
- `embodik` IK is stable and not the bottleneck.
- The current bottleneck is closed-loop placement robustness.

## Failure Signature To Target

Primary failing demos:
- `seed 1` -> `demo_1`
- `seed 2` -> `demo_10`

Dominant failure mode:
- under-reaching in `Y` at final placement
- secondary effect: slightly high final `Z`
- not primarily a grasp/release timing issue

Representative final offsets relative to the plate from the SmolVLA smoke diagnostic:
- `demo_0` success at `0cm`: `ee_minus_plate ~= [0.018, -0.020, 0.072]`
- `demo_1` fail at `0cm`: `ee_minus_plate ~= [0.001, -0.047, 0.075]`
- `demo_10` fail at `0cm`: `ee_minus_plate ~= [-0.006, -0.055, 0.073]`

## Required Evaluation Settings

- Use `embodik` as the primary evaluation path.
- Record videos for every validation run.
- Keep per-demo diagnostics enabled in sim:
  - first close step
  - first reopen step
  - final EE pose
  - object-relative offsets to bowl and plate

## Go / No-Go Rule For StarVLA

StarVLA is worth continuing if it improves either:
- overall success rate over the SmolVLA smoke baseline, or
- final `Y` placement offset for `demo_1` and `demo_10`

Do not treat lower train loss alone as progress. SmolVLA already showed that train loss can improve substantially without improving closed-loop execution.

## First StarVLA Experiments

1. Reproduce the SmolVLA smoke validation protocol unchanged.
2. Compare `demo_0`, `demo_1`, and `demo_10` side by side.
3. Check whether final `ee_minus_plate[1]` improves on the failing demos.
4. Only then broaden to larger sweeps or longer training.

## Current StarVLA Readiness

- Vendor patch present:
  - `vendor/starVLA/starVLA/model/framework/WM4A/WM4A_DHB.py`
  - `vendor/starVLA/starVLA/model/modules/action_model/DHBActionHeader.py`
- Local LeRobot converter is patched for the current LIBERO HDF5 layout.
- StarVLA dataloader smoke passes on both:
  - single-demo export
  - multi-task smoke export
- Active dataset rebuild target:
  - `/home/andypark/Projects/data/libero_lerobot`

## Current Blocking Issues

1. StarVLA training does not yet launch on this machine.
   - `accelerate launch ... train_starvla.py` currently fails inside DeepSpeed before model loading.
   - Observed error: `MissingCUDAException: CUDA_HOME does not exist`.
2. This environment also does not currently expose a working NVIDIA driver.
   - `nvidia-smi` fails with driver communication error.
   - `nvcc` is not on `PATH`.
3. No local StarVLA world-model checkpoint was found under the repo or user caches.
   - `WM4A_DHB` defaults to `./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B`.
   - That path does not exist locally right now.

## Immediate Next Steps

1. Finish the full `libero_lerobot` dataset rebuild.
2. Re-run a dataloader smoke check against the real dataset path.
3. Restore GPU training prerequisites:
   - working NVIDIA driver
   - visible CUDA toolkit / `CUDA_HOME`
   - local StarVLA base world-model checkpoint or an approved download path
4. Only after that, launch the first StarVLA smoke train:
   - `max_train_steps=1`
   - `per_device_batch_size=1`
   - offline `wandb`
