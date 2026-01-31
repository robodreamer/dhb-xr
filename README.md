# dhb_xr

DHB Extended Representations — SE(3) invariant trajectory encoding for robotics, VLAs, and motion data management.

## Overview

This library implements the double-reflection (DHB-DR) and quaternion-relative (DHB-QR) invariant representations for rigid-body motion trajectories on SE(3), as described in the manuscript "Double-Reflection DHB Invariant Representation on SE(3)". It provides:

- **Encoding/Decoding**: DHB-DR (Euler) and DHB-QR (quaternion) invariant computation and reconstruction
- **DHB-TI (time-invariant)**: Reparameterize by geometric progress (translational arc-length, angular, or hybrid) and resample at uniform progress knots so invariants are approximately independent of execution speed and sampling rate; then encode with DHB-DR or DHB-QR
- **Trajectory adaptation**: Constrained optimization for retargeting demos to new start/goal poses
- **GPU acceleration**: PyTorch batched operations and optional Cusadi for large-scale optimization
- **VLA support**: VQ-VAE/RVQ tokenization for streaming action representation
- **Motion database**: Similarity search, DTW alignment, and retrieval
- **Imitation learning**: Invariant-space and geodesic losses

## Installation

### End users (pip)

```bash
# Basic installation
pip install dhb_xr

# With optimization (CasADi)
pip install dhb_xr[optimization]

# With GPU (PyTorch)
pip install dhb_xr[gpu]

# Full installation
pip install dhb_xr[all]
```

### Developers (pixi)

```bash
# Install pixi: https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
cd dhb_xr
pixi install              # installs default env (dev tools, jupyter, no CUDA)
# If solve fails (e.g. pytorch-cuda): install only default env:
#   pixi install -e default

# Run tests
pixi run test

# Editable install
pixi run build

# Run notebooks
pixi run notebook
```

## Quick start

```python
import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod

# Create or load trajectory: N poses (position + quaternion wxyz)
positions = np.cumsum(np.random.randn(50, 3) * 0.01, axis=0)
quaternions = np.tile(np.array([1.0, 0, 0, 0]), (50, 1))  # identity orientation

# Encode to invariants (DHB-DR: double reflection + Euler)
result = encode_dhb_dr(
    positions, quaternions,
    method="pos",
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
linear_inv = result["linear_motion_invariants"]
angular_inv = result["angular_motion_invariants"]

# Decode back to trajectory
decoded = decode_dhb_dr(
    linear_inv, angular_inv,
    result["initial_pose"],
    method="pos",
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    drop_padded=True,
)
print(decoded["positions"].shape, decoded["quaternions"].shape)
```

### Time-invariant reparameterization (DHB-TI)

To reduce sensitivity to execution speed and sampling rate, reparameterize by a geometric progress variable and resample at uniform progress knots before encoding:

```python
from dhb_xr.encoder.dhb_ti import compute_progress, resample_by_progress, encode_dhb_dr_ti

# Progress: translation (arc-length), angular, or hybrid σ = α||Δp|| + (1-α)||Δr||
progress = compute_progress(positions, quaternions, kind="hybrid", alpha=0.5)
pos_m, quat_m = resample_by_progress(positions, quaternions, M=30, progress_kind="hybrid", alpha=0.5)
# Time-invariant encode
out = encode_dhb_dr_ti(positions, quaternions, M=30, progress_kind="hybrid", alpha=0.5, ...)
```

See `examples/08_dhb_ti_time_invariant.py`.

## Documentation

API documentation is built with Sphinx. From the repo root (use the pixi environment; dev feature includes sphinx):

```bash
pixi run docs          # or: pixi run build-docs
# open docs/_build/index.html
```

Without pixi: `pip install sphinx sphinx-autodoc-typehints` then `sphinx-build docs docs/_build -b html`.

### Cusadi (optional)

- **pip**: `pip install dhb_xr[cusadi]`
- **Usage**: `batched_decode_dhb_dr(...)` and `CusadiTrajectoryOptimizer.forward(...)` always work (NumPy batched decode). For GPU batch: export a CasADi decode with `python -m dhb_xr.optimization.export_casadi_decode --out fn_dhb_decode_step.casadi`, move the `.casadi` into [cusadi](https://github.com/se-hwan/cusadi)'s `src/casadi_functions/`, run `run_codegen.py`, then use `CusadiFunction` in your code.

### C++ extension (optional)

- **Build** (from repo root, with pixi): `pixi run build-cpp` (requires nanobind in dev feature). This builds the nanobind module into `src/dhb_xr/` so `import dhb_xr._dhb_xr_cpp` works.
- **Use**: `from dhb_xr import cpp_version` (returns `None` if not built). See `src/dhb_xr/_cpp/README.md` for extending with encode/decode.

## Testing

### Full test suite (pixi)

```bash
cd dhb_xr
pixi install
pixi run test
```

Or without pixi: `PYTHONPATH=src pytest tests/ -v`.

### C++ extension (nanobind)

1. **Build** the extension. Nanobind must be available to CMake (e.g. conda: `conda install -c conda-forge nanobind`; or set `nanobind_DIR` to the nanobind install share path). With pixi (default env has nanobind from conda-forge):

   ```bash
   pixi run build-cpp
   ```

   If pixi solve fails (e.g. CUDA), use a minimal env: `conda install -c conda-forge python cmake ninja nanobind`, then from repo root:

   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build .
   cp src/dhb_xr/_cpp/_dhb_xr_cpp*.so ../src/dhb_xr/
   ```

2. **Run C++ tests** (skip if extension not built):

   ```bash
   pixi run test -- tests/test_cpp.py -v
   ```

   Or run the checks manually:

   ```bash
   PYTHONPATH=src python3 -c "
   from dhb_xr import cpp_version
   if cpp_version:
       print('C++ extension:', cpp_version())
       from dhb_xr import _dhb_xr_cpp
       print('add(1,2)=', _dhb_xr_cpp.add(1.0, 2.0))
   else:
       print('C++ extension not built (pixi run build-cpp)')
   "
   ```

### Cusadi implementation

Cusadi tests cover **batched_decode_dhb_dr** and **CusadiTrajectoryOptimizer** (NumPy fallback; no cusadi package required):

```bash
pixi run test -- tests/test_cusadi.py -v
```

- **batched_decode_dhb_dr**: batch decode; test compares with single `decode_dhb_dr` for consistency.
- **CusadiTrajectoryOptimizer.forward**: same batch decode via the optimizer interface.
- **export_casadi_decode** (optional): if `casadi` is installed (`pip install dhb_xr[optimization]`), one test exports a `.casadi` decode step.

To test the CasADi export script explicitly:

```bash
pip install dhb_xr[optimization]
python -m dhb_xr.optimization.export_casadi_decode --out /tmp/fn_dhb_decode_step.casadi
# Check: ls /tmp/fn_dhb_decode_step.casadi
```

## References

- D. Lee, R. Soloperto, M. Saveriano, "Bidirectional invariant representation of rigid body motions and its application to gesture recognition and reproduction", *Autonomous Robots*, 2018.
- R. Soloperto, M. Saveriano, D. Lee, "A Bidirectional Invariant Representation of Motion for Gesture Recognition and Reproduction", *ICRA*, 2015.
- W. Wang et al., "Computation of rotation minimizing frames", *ACM TOG*, 2008.

## License

MIT
