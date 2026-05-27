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

### End users

```bash
pip install dhb_xr

# Solver-backed trajectory generation
pip install "dhb_xr[fatrop]"

# Fixed-horizon CusADi GPU decode support
pip install "dhb_xr[cusadi]"

# Examples, notebooks, and optional features
pip install "dhb_xr[examples,tokenization,database]"
```

See the [installation guide](https://robodreamer.github.io/dhb-xr/installation/)
for the full extras matrix, CUDA decode build steps, and troubleshooting.

### Developers

```bash
# Install pixi: https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
cd dhb_xr
pixi install              # installs default env (dev tools, jupyter, casadi, examples, build tools)

# Run tests
pixi run test

# Editable install (includes examples package)
pixi run build

# Run notebooks (CPU-only PyTorch)
pixi run notebook

# Copy examples for local development
pixi run dhb_xr-examples --copy ./local_examples

# Run examples programmatically
pixi run python -c "import examples; examples.run_basic_encoding()"

# Build docs and distributions
pixi run docs
pixi run build-dist
```

Publishing and version-management commands live in
[docs/development.md](docs/development.md).

## CUDA Environment

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

## Examples Package

DHB-XR includes a comprehensive examples package in the PyPI distribution, plus notebooks in
the source repository.

### Option 1: Install With Example Dependencies

```bash
pip install dhb_xr[examples]
```

Then run examples programmatically:

```python
import examples

# Run basic encoding example
examples.run_basic_encoding()

# Or run individual examples
from examples.basic_encoding import run_example
run_example()
```

### Option 2: Copy Examples Locally

For development and experimentation, copy examples to a local directory:

```bash
# Copy to default location (./dhb_xr_examples)
dhb_xr-examples --copy

# Copy to specific directory
dhb_xr-examples --copy ~/my_dhb_examples

# List available examples
dhb_xr-examples --list

# Show examples location
dhb_xr-examples
```

This creates a local copy you can modify and experiment with.

The examples package includes:
- **Core examples**: Basic encoding/decoding, trajectory adaptation, DHB-DR vs QR
- **Advanced examples**: GPU batch optimization, VLA tokenization, motion databases
- **VLA integration**: Full LIBERO simulation, perturbation robustness demos
- **Research examples**: Imitation learning losses, time-invariant reparameterization
- **Tutorial notebooks**: Interactive Jupyter notebooks in the source repository

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

See `examples/dhb_ti_time_invariant.py`.

## Documentation

📚 **[GitHub Pages](https://robodreamer.github.io/dhb-xr/)** - Complete API documentation with examples

The documentation is built with MkDocs and can be deployed to GitHub Pages on pushes to `main`
when Pages is enabled and set to build using GitHub Actions.

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

For fixed-horizon decode workloads, DHB-XR ships CusADi artifacts and generated
CUDA source for sample horizons `50`, `80`, `100`, `150`, and `200`. You no
longer need to clone an external CusADi checkout for those supported horizons.

CPU decode remains the default fallback. Build CUDA libraries explicitly on the
machines that need GPU decode.

**Requirements**

- NVIDIA GPU with CUDA toolkit (nvcc)
- CUDA-enabled PyTorch for GPU execution
- `dhb_xr[cusadi]` for CasADi and PyTorch dependencies

**Build the fixed-horizon libraries**

```bash
pip install "dhb_xr[cusadi]"
dhb_xr-build-cusadi-decode --horizons 50 80 100 150 200

# No-write path check
dhb_xr-build-cusadi-decode --horizons 100 --dry-run
```

The default output is `$DHB_XR_CUSADI_CACHE/build` when `DHB_XR_CUSADI_CACHE` is
set, otherwise `~/.cache/dhb_xr/cusadi/build`.

**Usage through the public API**

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
    cusadi_decode_horizon=100,
)

print(result["cusadi_decode"])
print(result.get("cusadi_decode_fallback_reason"))
```

Use `cusadi_decode="gpu_required"` or `cusadi_decode_fallback="error"` when
missing CUDA assets should fail instead of falling back to CPU. Use
`cusadi_decode_library_dir=...` to point at prebuilt libraries outside the
default cache.

See the [GPU Decode guide](docs/gpu_decode.md) and the
[CusADi paper](https://arxiv.org/abs/2408.09662) for details.

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

## VLA Integration (LIBERO-PRO / LIBERO / RoboCASA)

DHB-XR includes adapters for loading trajectory data from popular VLA benchmarks, with full support for **LIBERO-PRO** — the extended LIBERO benchmark that tests policy robustness under spatial, object, semantic, task, and environment perturbations.

**Why DHB-XR for VLA:** Current VLA models (RT-2, Octo, OpenVLA) map (vision + language) → actions, but struggle when the scene layout changes. DHB-XR provides a **structured trajectory representation** that decouples *motion shape* from *spatial context*:

| Without DHB-XR | With DHB-XR |
|---------------|-------------|
| Retrain or add data augmentation | Re-decode from new pose (~5-10ms decode, <1s Fatrop) |
| Actions tied to absolute positions | Invariants are SE(3)-invariant by construction |
| 100s of demos per spatial arrangement | **1 demo + DHB adaptation covers spatial variations** |

When LIBERO-PRO perturbs object positions or swaps objects, DHB can adapt the demonstration trajectory to the new configuration while perfectly preserving the original motion geometry (0.000 mm shape error).

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
    quaternions = episode["quaternions"]  # (N, 4) quaternions (w, x, y, z)

    # Encode to SE(3)-invariant representation
    result = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    invariants = result["linear_motion_invariants"]  # Shape: (N+2, 4)
```

### Full LIBERO / LIBERO-PRO Simulation

For running LIBERO tasks in simulation with DHB-XR trajectory adaptation and perturbation robustness testing:

```bash
# 1. Install Miniforge (if conda/mamba not available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# 2. Create and configure libero environment
~/miniforge3/bin/mamba create -n libero python=3.10 -y
~/miniforge3/bin/mamba run -n libero pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0
~/miniforge3/bin/mamba run -n libero pip install future easydict hydra-core cloudpickle 'gym==0.25.2'

# 3. Clone and install LIBERO-PRO (drop-in replacement for LIBERO with perturbation support)
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git ~/Projects/repos/LIBERO-PRO
~/miniforge3/bin/mamba run -n libero pip install -e ~/Projects/repos/LIBERO-PRO --config-settings editable_mode=compat

# 4. Configure LIBERO paths (creates ~/.libero/config.yaml)
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: ~/Projects/repos/LIBERO-PRO/libero/libero
bddl_files: ~/Projects/repos/LIBERO-PRO/libero/libero/bddl_files
init_states: ~/Projects/repos/LIBERO-PRO/libero/libero/init_files
datasets: ~/Projects/data/libero
assets: ~/Projects/repos/LIBERO-PRO/libero/libero/assets
EOF

# 5. Install dhb_xr and visualization dependencies
~/miniforge3/bin/mamba run -n libero pip install dhb_xr opencv-python imageio imageio-ffmpeg

# 6. Run simulation demo
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py
```

> **Note:** LIBERO-PRO is a drop-in replacement for LIBERO with identical core dependencies. It adds perturbation test suites (spatial swap, object replacement, language, task, environment) for evaluating policy robustness. All original LIBERO benchmarks (`libero_spatial`, `libero_goal`, etc.) work unchanged.

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

### DHB-XR vs Naive Replay — Swap Demo

The most compelling showcase of DHB-XR's value — directly comparing naive replay vs solver-adapted trajectory under spatial perturbation:

```bash
# Object positions swap (~17cm shift) — naive replay fails, DHB adapts
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_swap_demo.py

# Results:
#   Naive replay:  11.1 cm from NEW plate (wrong target)
#   DHB-adapted:    4.6 cm from NEW plate (correct target, decode ~5-10ms)
#   Improvement:    6.5 cm closer to correct target
```

### LIBERO-PRO Perturbation Robustness Demo

The `libero_pro_dhb_demo.py` script demonstrates how DHB's SE(3)-invariance enables robust trajectory adaptation under LIBERO-PRO's perturbation types:

```bash
# DHB analysis: encode demo, apply spatial perturbations, verify shape preservation
pixi run python examples/integration/libero_pro_dhb_demo.py --analysis

# Batch evaluation across multiple tasks (generates comparison plots)
pixi run python examples/integration/libero_pro_dhb_demo.py --batch

# Simulation: run original + perturbed variants, compare invariants
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_pro_dhb_demo.py --simulate

# With comparison video
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_pro_dhb_demo.py --simulate --save-video comparison.mp4
```

**Key results:**

| Metric | Value |
|--------|-------|
| Reconstruction error | 0.000 mm |
| Shape error (20mm perturbation) | 0.000 mm |
| Shape error (50mm perturbation) | 0.000 mm |
| Shape error (100mm perturbation) | 0.000 mm |
| Invariant correlation (with_mug variant) | 0.990 |
| Invariant correlation (with_milk variant) | 0.975 |

DHB invariants are perfectly frame-independent: adapting a trajectory to any perturbed starting pose preserves the original motion shape with zero error. Even under LIBERO-PRO's object replacement perturbations, the invariant representation of the same motion achieves >0.97 correlation.

**LIBERO-PRO perturbation types:**

| Type | Description | LIBERO-PRO Benchmark |
|------|-------------|---------------------|
| Position/Swap | Objects swap positions on the table | `libero_spatial_swap` |
| Object | Replace objects with visually different ones | `libero_spatial_object` |
| Semantic | Change language instructions | `libero_spatial_lan` |
| Task | Change goal/task entirely | `libero_spatial_task` |
| Environment | Change table/scene environment | `libero_spatial_env` |

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

### CusADi implementation

CusADi tests cover **batched_decode_dhb_dr**, **CusadiTrajectoryOptimizer**, exact
fixed-horizon artifact selection, CPU fallback, and the dhb_xr-owned library
builder:

```bash
pixi run test -- tests/test_cusadi.py -v
```

- **batched_decode_dhb_dr**: batch decode; test compares with single `decode_dhb_dr` for consistency.
- **CusadiTrajectoryOptimizer.forward**: same batch decode via the optimizer interface, with decode metadata.
- **build_cusadi_decode**: compiles supported fixed horizons into `$DHB_XR_CUSADI_CACHE` or `~/.cache/dhb_xr/cusadi`.

To test the build command explicitly without writing libraries:

```bash
python -m dhb_xr.optimization.build_cusadi_decode --horizons 50 100 --dry-run
```

## References

- D. Lee, R. Soloperto, M. Saveriano, "Bidirectional invariant representation of rigid body motions and its application to gesture recognition and reproduction", *Autonomous Robots*, 2018.
- R. Soloperto, M. Saveriano, D. Lee, "A Bidirectional Invariant Representation of Motion for Gesture Recognition and Reproduction", *ICRA*, 2015.
- W. Wang et al., "Computation of rotation minimizing frames", *ACM TOG*, 2008.

## License

MIT
