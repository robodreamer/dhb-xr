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
pixi install              # installs default env (dev tools, jupyter, casadi)

# Run tests
pixi run test

# Editable install
pixi run build

# Run notebooks (CPU-only PyTorch)
pixi run notebook
```

### CUDA Environment (GPU acceleration)

For GPU features (CusADi, VLA tokenization, faster PyTorch):

```bash
# Install the cuda environment (requires NVIDIA GPU with driver)
pixi install -e cuda

# Verify CUDA is available
pixi run -e cuda check-cuda
# Output: PyTorch 2.5.1, CUDA available: True, CUDA version: 12.4

# Run notebooks with CUDA
pixi run -e cuda notebook-cuda

# Run tests with CUDA
pixi run -e cuda test
```

**Performance (GPU position decode):**
- 1000 trajectories: **6.8 ms** (146k traj/s)
- Per-trajectory: **6.8 µs**

<details>
<summary><strong>Technical notes on pixi + PyTorch CUDA setup</strong></summary>

Getting CUDA-enabled PyTorch to work with pixi required careful configuration. Here are the key insights:

**Problem:** By default, pixi's dependency solver picks PyTorch from conda-forge, which is CPU-only (`pytorch-2.x.x-cpu_mkl_*`). Simply adding `pytorch-cuda` doesn't make it pick the CUDA build.

**Solution:** The `cuda` feature in `pyproject.toml` uses these techniques:

1. **Channel priority**: The cuda feature specifies `channels = ["pytorch", "nvidia", "conda-forge"]` with `channel-priority = "strict"` so PyTorch comes from the pytorch channel (which has CUDA builds), not conda-forge.

2. **Explicit channel specification**: Dependencies use `{ version = ">=2.0", channel = "pytorch" }` to force the pytorch channel:
   ```toml
   [tool.pixi.feature.cuda]
   channels = ["pytorch", "nvidia", "conda-forge"]
   channel-priority = "strict"

   [tool.pixi.feature.cuda.target.linux-64.dependencies]
   pytorch = { version = ">=2.0", channel = "pytorch" }
   pytorch-cuda = { version = ">=12.1", channel = "pytorch" }
   ```

3. **Platform-specific**: `pytorch-cuda` only exists for `linux-64`, so we use `target.linux-64.dependencies` to avoid solve failures on macOS.

4. **Separate solve group**: The cuda environment uses `solve-group = "cuda"` to avoid conflicts with the default CPU environment.

**Verification:**
```bash
# Check which pytorch package is installed
pixi list -e cuda | grep pytorch
# Should show: pytorch 2.x.x from pytorch channel (not conda-forge)
# Should show: pytorch-cuda 12.x from pytorch channel

# Verify CUDA is actually available
pixi run -e cuda python -c "import torch; print(torch.version.cuda)"
# Should print: 12.4 (not None)
```

**Common pitfalls:**
- Using `pytorch-cuda = "12.4"` fails (ambiguous version); use `"12.4.*"` or `">=12.1"`
- Not specifying channel priority causes conda-forge's CPU pytorch to be picked
- Forgetting `target.linux-64` causes solve failures on non-Linux platforms

</details>

## Quick start

```python
import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod

# Create or load trajectory: N poses (position + quaternion wxyz)
positions = np.cumsum(np.random.randn(50, 3) * 0.01, axis=0)
quaternions = np.tile(np.array([1.0, 0, 0, 0]), (50, 1))  # identity orientation

# Encode to invariants (DHB-DR: double reflection + Euler)
from dhb_xr.core.types import EncodingMethod
result = encode_dhb_dr(
    positions, quaternions,
    method=EncodingMethod.POSITION,
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)
linear_inv = result["linear_motion_invariants"]
angular_inv = result["angular_motion_invariants"]

# Decode back to trajectory
decoded = decode_dhb_dr(
    linear_inv, angular_inv,
    result["initial_pose"],
    method=EncodingMethod.POSITION,
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

📚 **[Read the Docs](https://dhb-xr.readthedocs.io/)** - Complete API documentation with examples

The documentation is built with MkDocs and automatically deployed to GitHub Pages on every push to main.

### Build locally

```bash
# Install development dependencies (includes MkDocs)
pixi install

# Build documentation
pixi run docs          # or: pixi run build-docs

# Serve locally for development
pixi run serve-docs    # opens http://127.0.0.1:8000/
```

### Without pixi

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
mkdocs build
mkdocs serve  # opens http://127.0.0.1:8000/
```

### CusADi GPU Acceleration (optional)

For batch processing thousands of trajectories, CusADi provides **up to 387x speedup** on GPU:

| Batch Size | CPU (ms) | GPU (ms) | Speedup |
|------------|----------|----------|---------|
| 100 | 34 | 0.8 | 43x |
| 1000 | 342 | 1.7 | **199x** |
| 2000 | 685 | 1.8 | **387x** |

**Requirements:**
- NVIDIA GPU with CUDA toolkit (nvcc)
- PyTorch with CUDA support

**Setup with pixi:**

```bash
cd dhb_xr

# 1. Install pixi environment
pixi install

# 2. Install PyTorch with CUDA (one-time)
pixi run install-cuda

# 3. Clone cusadi (if not already)
git clone https://github.com/se-hwan/cusadi /path/to/cusadi
cd /path/to/cusadi && pip install -e .

# 4. Build CasADi functions (use pixi python for version compatibility)
cd dhb_xr
pixi run python3 << 'EOF'
import casadi as ca
import numpy as np

def euler_to_rot(rx, ry, rz):
    cx, sx = ca.cos(rx), ca.sin(rx)
    cy, sy = ca.cos(ry), ca.sin(ry)
    cz, sz = ca.cos(rz), ca.sin(rz)
    return ca.vertcat(
        ca.horzcat(cy*cz, -cy*sz, sy),
        ca.horzcat(cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx),
        ca.horzcat(sx*sz - cx*cz*sy, cz*sx + cx*sy*sz, cx*cy))

T = 50
lin_inv = ca.SX.sym("lin_inv", T * 4)
init_pos = ca.SX.sym("init_pos", 3)
init_rot = ca.SX.sym("init_rot", 9)

rotm = ca.reshape(init_rot, 3, 3)
pos = init_pos
out = [init_pos]
for k in range(T):
    mag, rx, ry, rz = lin_inv[k*4], lin_inv[k*4+1], lin_inv[k*4+2], lin_inv[k*4+3]
    rotm = rotm @ euler_to_rot(rx, ry, rz)
    pos = pos + rotm @ ca.vertcat(mag, 0, 0)
    out.append(pos)

fn = ca.Function("fn_dhb_decode_linear", [lin_inv, init_pos, init_rot], [ca.horzcat(*out).T])
fn.save("/path/to/cusadi/src/casadi_functions/fn_dhb_decode_linear.casadi")
print(f"Saved: {fn}")
EOF

# 5. Compile CUDA kernel (use pixi python for same CasADi version)
cd /path/to/cusadi
/path/to/dhb_xr/.pixi/envs/default/bin/python3 run_codegen.py --fn=fn_dhb_decode_linear
```

**Usage:**

```python
import sys
sys.path.insert(0, "/path/to/cusadi")
sys.path.insert(0, "/path/to/cusadi/src")
sys.path.insert(0, "/path/to/cusadi/build")

import torch
import casadi as ca
from src.CusadiFunction import CusadiFunction

fn = ca.Function.load("/path/to/cusadi/src/casadi_functions/fn_dhb_decode_linear.casadi")
cusadi_fn = CusadiFunction(fn, batch_size=1000)

# GPU tensors (batch_size, features)
lin_inv_gpu = torch.from_numpy(invariants).cuda().contiguous()
init_pos_gpu = torch.from_numpy(positions).cuda().contiguous()
init_rot_gpu = torch.from_numpy(rotations).cuda().contiguous()

cusadi_fn.evaluate([lin_inv_gpu, init_pos_gpu, init_rot_gpu])
positions = cusadi_fn.getDenseOutput(0).cpu().numpy()  # (batch, T+1, 3)
```

**Important:** The CasADi `.casadi` files must be saved with the same CasADi version that loads them. Use pixi python for both building and running to ensure version compatibility.

See the [CusADi paper](https://arxiv.org/abs/2408.09662) for details on the parallelization framework.

### Fatrop Fast Optimization (optional)

For **single trajectory optimization** with constraints, [Fatrop](https://github.com/meco-group/fatrop) provides ~10x speedup over IPOPT:

| Solver | Use Case | Speed |
|--------|----------|-------|
| IPOPT | General NLP | ~50-100ms |
| **Fatrop** | Structured OCP | ~5-10ms |

**Installation:**

```bash
# Rockit (required for OCP formulation)
pip install rockit-meco
# or with pixi:
pixi run install-rockit

# Fatrop is bundled with conda casadi (no separate install needed)
# The pixi environment includes casadi with Fatrop support
```

**Usage:**

```python
from dhb_xr.optimization import generate_trajectory_fatrop

result = generate_trajectory_fatrop(
    demo_positions, demo_quaternions,
    start_pose={'position': start_pos, 'quaternion': start_quat},
    goal_pose={'position': goal_pos, 'quaternion': goal_quat},
    traj_length=50,
    use_fatrop=True,  # False for IPOPT fallback
)
print(f"Solved in {result['solve_time']*1000:.1f} ms")
```

**Use cases:**
- Real-time MPC (100+ Hz replanning)
- Constrained trajectory generation (obstacles, joint limits)
- Online trajectory adaptation

**CusADi vs Fatrop:**
- CusADi: Best for **batch evaluation** (1000 trajectories in 2ms)
- Fatrop: Best for **single optimization** with constraints (5-10ms)

### C++ extension (optional)

- **Build** (from repo root, with pixi): `pixi run build-cpp` (requires nanobind in dev feature). This builds the nanobind module into `src/dhb_xr/` so `import dhb_xr._dhb_xr_cpp` works.
- **Use**: `from dhb_xr import cpp_version` (returns `None` if not built). See `src/dhb_xr/_cpp/README.md` for extending with encode/decode.

## VLA Integration (LIBERO / RoboCASA)

DHB-XR includes adapters for loading trajectory data from popular VLA benchmarks.

### Quick Start: DHB Encoding Only

No simulation required - just load and process trajectory data:

```bash
# 1. Download LIBERO-Spatial dataset (smallest, ~2.8GB compressed)
mkdir -p ~/Projects/data/libero && cd ~/Projects/data/libero
wget -O libero_spatial.zip "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
unzip libero_spatial.zip

# 2. Test DHB encoding (works with pixi environment)
pixi run python examples/integration/test_libero_adapter.py
pixi run python examples/integration/test_libero_encoding.py

# 3. Run full demo (DHB-only mode, no simulation, saves plot to /tmp/dhb_demo_plot.png)
pixi run python examples/integration/libero_full_demo.py --dhb-only

# 4. Motion retrieval demo
pixi run python examples/integration/libero_full_demo.py --retrieval

# 5. View generated plot
xdg-open /tmp/dhb_demo_plot.png  # Linux
```

### Programmatic Usage

```python
from dhb_xr.integration.vla.libero import LiberoAdapter
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import EncodingMethod, DHBMethod

# Load episodes from LIBERO HDF5
adapter = LiberoAdapter()
for episode in adapter.load_dataset("/path/to/libero_task.hdf5"):
    positions = episode["positions"]      # (N, 3) end-effector positions
    quaternions = episode["quaternions"]  # (N, 4) quaternions (x, y, z, w)

    # Encode to SE(3)-invariant representation
    result = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    invariants = result["linear_motion_invariants"]  # Shape: (N+2, 4)
```

### Full LIBERO Simulation

For running LIBERO tasks in simulation with DHB-XR trajectory adaptation:

```bash
# 1. Install Miniforge (if conda/mamba not available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# 2. Create and configure libero environment
~/miniforge3/bin/mamba create -n libero python=3.10 -y
~/miniforge3/bin/mamba run -n libero pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0
~/miniforge3/bin/mamba run -n libero pip install future easydict hydra-core cloudpickle 'gym==0.25.2'

# 3. Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/Projects/repos/LIBERO
~/miniforge3/bin/mamba run -n libero pip install -e ~/Projects/repos/LIBERO --config-settings editable_mode=compat

# 4. Configure LIBERO paths (creates ~/.libero/config.yaml)
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: ~/Projects/repos/LIBERO/libero/libero
bddl_files: ~/Projects/repos/LIBERO/libero/libero/bddl_files
init_states: ~/Projects/repos/LIBERO/libero/libero/init_files
datasets: ~/Projects/data/libero
assets: ~/Projects/repos/LIBERO/libero/libero/assets
EOF

# 5. Install dhb_xr and visualization dependencies
~/miniforge3/bin/mamba run -n libero pip install dhb_xr opencv-python imageio imageio-ffmpeg

# 6. Run simulation demo
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py
```

### Viewing Simulations

```bash
# Option 1: Real-time display with OpenCV (requires X11 display)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render

# Option 2: Save video for later viewing (works headless)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --save-video demo.mp4

# Option 3: Both display and save
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render --save-video demo.mp4

# Play saved video
vlc demo.mp4  # or: ffplay demo.mp4
```

For remote servers without display, use `--save-video` and download the video locally.

**Key version requirements:**
- robosuite==1.4.0 (LIBERO is incompatible with robosuite 1.5+)
- Python 3.10 recommended
- bddl==1.0.1, robomimic==0.2.0

See [VLA Integration Guide](docs/integration/vla_integration.md) for full documentation.

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
