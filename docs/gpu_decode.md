# GPU Decode

DHB-XR can use CusADi-generated CUDA libraries for fixed-horizon DHB-DR linear
decode. This is an optional acceleration path for large batches and for
`generate_trajectory(..., backend="cusadi")` when exact GPU decode is required.

## What Ships With dhb_xr

The package includes:

- CasADi artifacts in `dhb_xr.optimization.cusadi_functions`.
- Generated CUDA source in `dhb_xr.optimization.cusadi_codegen`.
- A local builder exposed as `dhb_xr-build-cusadi-decode`.
- A runtime loader that checks the dhb_xr cache before falling back to CPU.

Supported sample horizons are `50`, `80`, `100`, `150`, and `200`. These are
sample counts, not invariant counts. A trajectory with `traj_length=100` should
use horizon `100`.

## Build Libraries

Install the optional dependencies, then compile only the horizons you need:

```bash
python -m pip install "dhb_xr[cusadi]"
dhb_xr-build-cusadi-decode --horizons 50 80 100 150 200
```

The builder writes into `$DHB_XR_CUSADI_CACHE/build` when
`DHB_XR_CUSADI_CACHE` is set. Otherwise it uses
`~/.cache/dhb_xr/cusadi/build`.

Use `--dry-run` to verify paths and selected horizons without writing files:

```bash
dhb_xr-build-cusadi-decode --horizons 100 --dry-run
```

Use `--output-dir` for a project-local or deployment-managed cache:

```bash
dhb_xr-build-cusadi-decode --horizons 100 --output-dir /opt/dhb_xr/cusadi
```

## Runtime Selection

The high-level API keeps CPU decode as the default fallback:

```python
from dhb_xr.optimization import generate_trajectory

result = generate_trajectory(
    demo_positions,
    demo_quaternions,
    pose_target_init={"position": start_pos, "quaternion": start_quat},
    pose_target_final={"position": goal_pos, "quaternion": goal_quat},
    traj_length=100,
    backend="cusadi",
    cusadi_decode="auto",
)

print(result["cusadi_decode"])
print(result.get("cusadi_decode_fallback_reason"))
```

Require GPU decode when falling back would hide a deployment problem:

```python
result = generate_trajectory(
    demo_positions,
    demo_quaternions,
    pose_target_init={"position": start_pos, "quaternion": start_quat},
    pose_target_final={"position": goal_pos, "quaternion": goal_quat},
    traj_length=100,
    backend="cusadi",
    cusadi_decode="gpu_required",
    cusadi_decode_horizon=100,
)
```

If libraries live outside the default cache, pass
`cusadi_decode_library_dir="/path/to/build"`.

## When To Use It

Use FATROP first for single constrained trajectory generation. Use CusADi GPU
decode when decode throughput matters, when many fixed-horizon trajectories are
evaluated together, or when a deployment wants explicit CUDA artifacts built
ahead of time.

For unsupported horizons, either choose one of the shipped fixed sizes or use CPU
decode. Regenerating arbitrary CusADi kernels remains a developer task and is
not required for normal dhb_xr usage.

## Troubleshooting

If the runtime falls back to CPU, inspect the metadata:

```python
print(result["cusadi_decode_requested_horizon"])
print(result["cusadi_decode_artifact_path"])
print(result["cusadi_decode_library_path"])
print(result["cusadi_decode_fallback_reason"])
```

Common causes:

- The requested `traj_length` is not one of `50`, `80`, `100`, `150`, or `200`.
- The matching library has not been built.
- PyTorch is installed without CUDA support.
- `DHB_XR_CUSADI_CACHE` points at a different cache than the one used during
  compilation.
