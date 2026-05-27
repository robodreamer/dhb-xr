# Trajectory Optimization

The main entry point is `generate_trajectory`. It adapts a demonstration
trajectory to a requested start and goal pose, then reports which backend was
requested and which solver path actually ran.

## Backend Selection

Use `backend="auto"` for normal single-trajectory generation. It prefers FATROP
when the optional Rockit/FATROP stack is importable. Fallback to slower IPOPT or
interpolation is disabled by default; set `allow_fallback=True` only when an
exploratory run should continue after a preferred backend is unavailable.

| Backend | Use when | Notes |
| --- | --- | --- |
| `auto` | Production default | Uses FATROP when available. |
| `fatrop` | Require the structured OCP solver | Raises unless FATROP succeeds or `allow_fallback=True`. |
| `cusadi` | Boundary adaptation plus fixed-horizon CusADi decode | Exact GPU decode is optional and falls back to CPU by default. |
| `ipopt` | Legacy CasADi/IPOPT behavior | Useful for comparisons and fallback studies. |
| `interpolation` | Deterministic resample/decode baseline | No nonlinear solver dependency. |

```python
from dhb_xr.optimization import generate_trajectory

result = generate_trajectory(
    demo_positions,
    demo_quaternions,
    pose_target_init={"position": start_pos, "quaternion": start_quat},
    pose_target_final={"position": goal_pos, "quaternion": goal_quat},
    traj_length=100,
    backend="auto",
)

print(result["requested_backend"], result["solver"])
```

## CusADi Decode Policy

CusADi decode is fixed-horizon by design. DHB-XR ships artifacts and generated
CUDA source for sample horizons `50`, `80`, `100`, `150`, and `200`. Build
compiled libraries only when the machine needs GPU decode:

```bash
python -m dhb_xr.optimization.build_cusadi_decode --horizons 50 80 100 150 200
```

Runtime controls:

- `cusadi_decode="auto"`: use GPU decode when all assets are available,
  otherwise use CPU fallback.
- `cusadi_decode="gpu_required"`: require a matching compiled library and CUDA
  execution.
- `cusadi_decode_horizon="auto"`: select the exact sample horizon from
  `traj_length`.
- `cusadi_decode_horizon=100`: require a specific fixed sample horizon.
- `cusadi_decode_fallback="cpu"`: default CPU fallback.
- `cusadi_decode_fallback="error"`: raise instead of falling back.
- `cusadi_decode_library_dir=...`: load prebuilt libraries from a custom
  directory instead of the default dhb_xr cache.

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

print(result["cusadi_decode"])
print(result["cusadi_decode_library_path"])
```

The result metadata includes `requested_backend`, `solver`, `optimizer_solver`,
`cusadi_decode`, `cusadi_decode_requested_horizon`,
`cusadi_decode_artifact_horizon`, `cusadi_decode_artifact_path`,
`cusadi_decode_library_path`, and `cusadi_decode_fallback_reason` when that path
runs.

See [GPU Decode](../gpu_decode.md) for install, build, cache, and troubleshooting
details.

## API Surface

Public facade:

- `dhb_xr.optimization.generate_trajectory`
- `dhb_xr.optimization.get_optimizer`

FATROP path:

- `dhb_xr.optimization.fatrop_solver.generate_trajectory_fatrop`
- `dhb_xr.optimization.fatrop_solver.FatropTrajectoryGenerator`
- `dhb_xr.optimization.fatrop_solver.ConstrainedTrajectoryGenerator`

CusADi path:

- `dhb_xr.optimization.cusadi_solver.CusadiDecodeSpec`
- `dhb_xr.optimization.cusadi_solver.CusadiTrajectoryOptimizer`
- `dhb_xr.optimization.cusadi_solver.batched_decode_dhb_dr`
- `dhb_xr.optimization.build_cusadi_decode.build_cusadi_decode_libraries`
