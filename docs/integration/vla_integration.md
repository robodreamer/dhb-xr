# VLA Integration (LIBERO-PRO / LIBERO / RoboCASA)

This page shows how to use DHB-XR with popular Vision-Language-Action (VLA) benchmarks for trajectory encoding, retrieval, simulation, and **perturbation robustness testing**.

## Overview

DHB-XR provides thin adapter layers for:

- **LIBERO-PRO**: Extended LIBERO benchmark with perturbation testing (spatial swap, object replacement, environment change, semantic, task)
- **LIBERO**: Lifelong robot learning benchmark (130 tasks)
- **RoboCASA**: Large-scale manipulation benchmark

The pipeline:

1. **Load** dataset episodes (HDF5 format)
2. **Extract** end-effector poses (positions, quaternions)
3. **Encode** to SE(3)-invariant DHB representations
4. **Adapt** trajectories to new initial poses (perturbation robustness)
5. **Retrieve** similar motions or **tokenize** for VLA training

### Why DHB-XR for VLA?

Current VLA models (RT-2, Octo, OpenVLA) learn a mapping from **(vision + language) → actions**. When the scene changes — even slightly — they either fail, require massive data augmentation, or need fine-tuning. DHB-XR addresses this by providing a **structured trajectory representation layer** that decouples *motion shape* from *spatial context*:

```
Traditional VLA:  Vision → Policy → Actions (tied to absolute positions)

With DHB-XR:      Vision → Object Pose Detection → DHB Adaptation → Actions
                            ↑                         ↑
                     "Where is the plate now?"   "Same motion shape,
                                                  new target pose"
```

DHB invariants are **SE(3)-invariant** — the encoded motion shape is independent of the global reference frame. This makes DHB a natural fit for VLA spatial generalization:

| VLA Scenario | Without DHB-XR | With DHB-XR |
|-------------|---------------|-------------|
| Object shifts on table | Retrain or fail | Re-decode from new pose (~5-10ms) |
| Different robot starting config | Action replay fails | Adapt with same invariants |
| Transfer between robots | Need new training data | Same invariants, different kinematics |
| Few-shot learning | Need 100s of demos per arrangement | 1 demo + DHB adaptation covers spatial variations |

**Key advantage — data efficiency:** Instead of collecting 1000s of demonstrations across different spatial arrangements, collect a few demos and use DHB to *analytically* generate the spatial variations. This is especially valuable for LIBERO-PRO's perturbation benchmarks, where each perturbation type would otherwise require separate training data.

**Demonstrated results (LIBERO-PRO spatial swap):**

| Method | EE → OLD plate | EE → NEW plate |
|--------|---------------|---------------|
| Naive replay (same actions) | 5.2 cm | 11.1 cm |
| DHB-adapted (Fatrop solver) | 20.8 cm | **4.6 cm** |

When the plate moves ~17cm (via LIBERO-PRO swap perturbation), naive replay reaches for the old position and fails. DHB-adapted trajectory correctly targets the new plate position — 6.5cm closer to the correct target, decoded in ~5-10ms.

### LIBERO-PRO perturbation compatibility

- **Position perturbations** (objects swap): DHB adapts trajectory to new target locations while preserving motion shape (0.000 mm shape error)
- **Object replacement**: Same behavioral motion with different objects yields >0.97 invariant correlation
- **Environment changes**: Motion essence is captured regardless of scene layout

## LIBERO Integration

### Dataset Download

LIBERO provides four task suites. Start with LIBERO-Spatial (smallest):

```bash
# Create data directory
mkdir -p ~/Projects/data/libero && cd ~/Projects/data/libero

# Download LIBERO-Spatial (10 tasks, ~6GB)
wget -O libero_spatial.zip "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
unzip libero_spatial.zip

# Other available datasets:
# LIBERO-Object: https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip
# LIBERO-Goal: https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip
# LIBERO-100: https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip
```

### Using the LiberoAdapter

```python
from dhb_xr.integration.vla.libero import LiberoAdapter

# Create adapter
adapter = LiberoAdapter()

# Load episodes from HDF5 file
for episode in adapter.load_dataset("/path/to/libero_task.hdf5"):
    positions = episode["positions"]      # (N, 3) end-effector positions
    quaternions = episode["quaternions"]  # (N, 4) quaternions (w, x, y, z)
    metadata = episode["metadata"]        # demo_id, task info, etc.

    print(f"Episode {metadata['demo_id']}: {len(positions)} frames")
```

**Quaternion convention:** DHB-XR public encoders/decoders use scalar-first `wxyz` quaternions. LIBERO stores `robot_states[:, 5:9]` and direct `robot0_eef_quat` observations in `wxyz` on the tested setup, so `LiberoAdapter` returns `episode["quaternions"]` in `wxyz`. Convert to scipy `xyzw` only at scipy/robosuite boundaries.

### DHB Encoding

```python
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import EncodingMethod, DHBMethod
import numpy as np

# Encode trajectory to invariants
result = encode_dhb_dr(
    positions, quaternions,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)

linear_inv = result["linear_motion_invariants"]   # (N+2, 4)
angular_inv = result["angular_motion_invariants"] # (N+2, 4)
invariants = np.concatenate([linear_inv, angular_inv], axis=1)  # (N+2, 8)
```

### Motion Retrieval

Build a database of trajectories and query for similar motions:

```python
from dhb_xr.database.motion_db import MotionDatabase

# Build database (encoding happens automatically)
db = MotionDatabase(dhb_method="double_reflection")

for episode in adapter.load_dataset(dataset_path):
    db.add(
        positions=episode["positions"],
        quaternions=episode["quaternions"],
        metadata=episode["metadata"],
    )

# Query similar motions (DTW for variable-length trajectories)
results = db.retrieve(
    query_positions=query_pos,
    query_quaternions=query_quat,
    k=5,
    use_dtw=True,
)

for invariants, metadata, distance in results:
    print(f"Match: {metadata['demo_id']}, distance={distance:.4f}")
```

### Run Integration Tests

```bash
# Test adapter loading
pixi run python examples/integration/test_libero_adapter.py

# Test DHB encoding
pixi run python examples/integration/test_libero_encoding.py

# Full retrieval demo
pixi run python examples/integration/test_libero_retrieval.py
```

## LIBERO / LIBERO-PRO Simulation

To run LIBERO tasks in simulation with DHB-XR trajectory adaptation and perturbation robustness testing:

### Installation

LIBERO (and LIBERO-PRO) require specific package versions. We recommend using a separate conda environment:

```bash
# 1. Install Miniforge (if conda/mamba not available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# 2. Create libero environment
~/miniforge3/bin/mamba create -n libero python=3.10 -y

# 3. Install LIBERO dependencies (specific versions required!)
~/miniforge3/bin/mamba run -n libero pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0
~/miniforge3/bin/mamba run -n libero pip install future easydict hydra-core cloudpickle 'gym==0.25.2'

# 4. Clone and install LIBERO-PRO (recommended; drop-in replacement for LIBERO)
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git ~/Projects/repos/LIBERO-PRO
~/miniforge3/bin/mamba run -n libero pip install -e ~/Projects/repos/LIBERO-PRO --config-settings editable_mode=compat

# 5. Configure LIBERO paths
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /home/$USER/Projects/repos/LIBERO-PRO/libero/libero
bddl_files: /home/$USER/Projects/repos/LIBERO-PRO/libero/libero/bddl_files
init_states: /home/$USER/Projects/repos/LIBERO-PRO/libero/libero/init_files
datasets: /home/$USER/Projects/data/libero
assets: /home/$USER/Projects/repos/LIBERO-PRO/libero/libero/assets
EOF

# 6. Install dhb_xr
~/miniforge3/bin/mamba run -n libero pip install dhb_xr[optimization]
```

> **LIBERO-PRO vs LIBERO:** LIBERO-PRO is a drop-in replacement with identical core dependencies. It adds 50+ perturbed task suites for evaluating policy robustness under spatial swap, object replacement, semantic instruction changes, task changes, and environment changes. All original LIBERO benchmarks work unchanged.

**Critical version notes:**
- `robosuite==1.4.0` is required (LIBERO is incompatible with robosuite 1.5+)
- `gym==0.25.2` (older gym API required by LIBERO)
- `bddl==1.0.1` and `robomimic==0.2.0` for dataset compatibility

### Running the Demo

```bash
# Full simulation demo (no visualization)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py

# DHB-only mode (no simulation required, generates plot)
pixi run python examples/integration/libero_full_demo.py --dhb-only
# View the generated plot: xdg-open /tmp/dhb_demo_plot.png

# Specific task
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --task_id 3
```

### LIBERO-PRO Perturbation Robustness Demo

The dedicated `libero_pro_dhb_demo.py` script demonstrates DHB's SE(3)-invariance under LIBERO-PRO perturbations:

```bash
# Perturbation analysis (no simulation, fast)
# Encodes demo, applies spatial perturbations, verifies shape preservation
pixi run python examples/integration/libero_pro_dhb_demo.py --analysis

# Batch evaluation across multiple tasks (generates comparison plots)
pixi run python examples/integration/libero_pro_dhb_demo.py --batch

# Full simulation: run original + LIBERO-PRO variants, compare invariants
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_pro_dhb_demo.py --simulate

# With side-by-side comparison video
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_pro_dhb_demo.py --simulate --save-video comparison.mp4
```

**Perturbation analysis results (LIBERO-Spatial, 5 tasks):**

| Perturbation Level | Shape Preservation Error |
|--------------------|------------------------|
| 20mm offset | 0.000 mm |
| 50mm offset | 0.000 mm |
| 100mm offset | 0.000 mm |

DHB invariants are **perfectly frame-independent**: adapting a trajectory to any perturbed starting pose preserves the original motion shape with zero error.

**Cross-perturbation invariant consistency:**

| LIBERO-PRO Variant | Invariant Correlation |
|-------------------|----------------------|
| with_mug (object change) | 0.990 |
| with_milk (object change) | 0.975 |

Even when LIBERO-PRO replaces objects in the scene, running the same demo actions produces trajectories whose DHB invariants are >0.97 correlated with the original — confirming that DHB captures the motion essence rather than scene-specific details.

**Available LIBERO-PRO perturbation types:**

| Type | Description | Example Benchmark |
|------|-------------|-------------------|
| Position/Swap | Objects swap positions | `libero_spatial_swap` |
| Object | Replace with different objects | `libero_spatial_object` |
| Semantic | Paraphrase language instructions | `libero_spatial_lan` |
| Task | Change goal/task | `libero_spatial_task` |
| Environment | Change table/scene | `libero_spatial_env` |

### DHB-XR vs Naive Replay — Swap Demo

The most compelling demonstration of DHB-XR's value is the `libero_swap_demo.py` script. It runs three scenarios back-to-back:

1. **Original**: Run the demo in the unperturbed environment (baseline, verifies SUCCESS)
2. **Naive replay in swapped env**: Same actions, but plate and cookies have swapped positions (~17cm shift) — robot reaches for old plate position (FAILS)
3. **DHB-adapted in swapped env**: Trajectory adapted via Fatrop solver to target the NEW plate position — robot correctly moves towards new target

```bash
# Run the swap comparison demo
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_swap_demo.py

# Outputs:
#   /tmp/dhb_swap_comparison.png  — 8-panel plot with trajectory comparison
#   /tmp/dhb_swap_comparison.mp4  — side-by-side video
```

**Note on LIBERO's action space:** LIBERO uses Operational Space Control (OSC) velocity commands. Converting an adapted SE(3) trajectory to OSC actions requires controller inversion, which is non-trivial. The demo uses a simplified action bias approach to illustrate the *geometric intent* of the DHB adaptation. For a production VLA system, DHB-XR would be integrated at the trajectory planning level (before the controller), not as a post-hoc action modifier.

### Viewing Simulations

LIBERO's `OffScreenRenderEnv` captures camera frames that can be displayed or saved:

```bash
# Option 1: Real-time display with OpenCV (requires X11 display)
# Press 'q' to quit the display window
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render

# Option 2: Save video for later viewing (works headless)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --save-video demo.mp4

# Option 3: Both display and save
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render --save-video demo.mp4

# Play saved video
vlc demo.mp4  # or: ffplay demo.mp4
```

**Visualization dependencies:**
```bash
~/miniforge3/bin/mamba run -n libero pip install opencv-python imageio imageio-ffmpeg
```

**Remote server tips:**
- Use `--save-video` to record, then download and play locally
- For SSH with X11 forwarding (`ssh -X`), `--render` should work
- The displayed view is the `agentview` camera (same as policy input)

### Basic Simulation Example

```python
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import EncodingMethod, DHBMethod

# Load benchmark and create environment
benchmark = get_benchmark("libero_spatial")()
task_bddl_file = benchmark.get_task_bddl_file_path(0)
env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=128, camera_widths=128)
obs = env.reset()

# Get initial EE pose from simulation
initial_ee_pos = obs["robot0_eef_pos"]
initial_ee_quat = obs["robot0_eef_quat"]

# Load and encode demo trajectory
from dhb_xr.integration.vla.libero import LiberoAdapter
adapter = LiberoAdapter()
demo = next(adapter.load_dataset("/path/to/libero_task.hdf5"))

enc = encode_dhb_dr(
    demo["positions"], demo["quaternions"],
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)

# Decode with new initial pose (trajectory adaptation)
new_initial_pose = {
    "position": initial_ee_pos,
    "quaternion": initial_ee_quat,
}
decoded = decode_dhb_dr(
    enc["linear_motion_invariants"],
    enc["angular_motion_invariants"],
    new_initial_pose,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
)

# Use adapted trajectory
adapted_positions = decoded["positions"]
adapted_quaternions = decoded["quaternions"]

env.close()
```

## RoboCASA Integration

RoboCASA uses the same HDF5 format with robomimic conventions:

```python
from dhb_xr.integration.vla.robocasa import RoboCASAAdapter

adapter = RoboCASAAdapter()
for episode in adapter.load_dataset("/path/to/robocasa.hdf5"):
    # Same interface as LiberoAdapter
    positions = episode["positions"]
    quaternions = episode["quaternions"]
```

## VLA Tokenization And Streaming Utilities

For VLA training, tokenize invariants to discrete action tokens:

```python
from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig

config = DHBVLAPipelineConfig(
    tokenizer_kind="oat",
    codebook_size=256,
    latent_dim=32,
)
pipeline = DHBVLAPipeline(config=config)

# Process dataset
outputs = pipeline.process_dataset(episodes)
for out in outputs:
    tokens = out["tokens"]        # Discrete action tokens
    invariants = out["invariants"] # Raw invariants
```

For learned invariant heads, chunk streaming, SmolVLA wrapping, and waypoint execution:

```python
from dhb_xr.integration.vla.action_head import DHBActionHead
from dhb_xr.integration.vla.chunk_rollout import ChunkRolloutConfig, TrajectoryChunkRunner
from dhb_xr.integration.vla.smolvla_adapter import SmolVLADHBAdapter
from dhb_xr.integration.vla.waypoint_controller import WaypointPDController

head = DHBActionHead(vlm_dim=576, action_horizon=16)
adapter = SmolVLADHBAdapter.from_dhb_head(head)
runner = TrajectoryChunkRunner(ChunkRolloutConfig(chunk_horizon=16, stride=4))
controller = WaypointPDController()
```

**Note:** OAT tokenization requires PyTorch and einops: `pip install "dhb_xr[tokenization]"`. Use `dhb_xr[gpu]` when you only need torch-backed heads without the tokenizer extras.

## Data Format Reference

### LIBERO HDF5 Structure

```
/data/
  demo_0/
    obs/
      ee_pos: (N, 3)           # End-effector position
      ee_ori: (N, 3)           # Euler angles (not used)
      agentview_rgb: (N, 128, 128, 3)
      eye_in_hand_rgb: (N, 128, 128, 3)
    robot_states: (N, 9)       # [gripper(2), pos(3), quat(4)]
    actions: (N, 7)            # Delta actions
  demo_1/
    ...
```

### Adapter Key Configuration

If your dataset uses different keys, customize the adapter:

```python
adapter = LiberoAdapter(
    pos_keys=("custom_pos", "ee_pos"),
    quat_keys=("custom_quat", "ee_quat"),
    robot_states_key="robot_states",
    robot_states_quat_slice=(5, 9),
)
```

## Troubleshooting

### Missing h5py

```bash
pixi add h5py
# or: pip install h5py
```

### Quaternion format mismatch

DHB-XR APIs use `wxyz`. `LiberoAdapter` returns `wxyz`. Scipy and robosuite APIs use `xyzw`, so convert with `quat_xyzw = quat_wxyz[[1, 2, 3, 0]]` only when calling those libraries.

### Variable-length trajectories

Use `use_dtw=True` in `db.retrieve()` for comparing trajectories of different lengths.

### ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'

This means you have robosuite >= 1.5 installed, but LIBERO requires 1.4.0:

```bash
pip uninstall robosuite
pip install robosuite==1.4.0
```

### LIBERO import prompts for input

On first import, LIBERO asks to configure dataset paths. Pre-create the config:

```bash
mkdir -p ~/.libero
# For LIBERO-PRO:
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /path/to/LIBERO-PRO/libero/libero
bddl_files: /path/to/LIBERO-PRO/libero/libero/bddl_files
init_states: /path/to/LIBERO-PRO/libero/libero/init_files
datasets: /path/to/your/datasets
assets: /path/to/LIBERO-PRO/libero/libero/assets
EOF

# For original LIBERO (same format):
# Replace LIBERO-PRO with LIBERO in the paths above
```

### Switching between LIBERO and LIBERO-PRO

To switch, update `~/.libero/config.yaml` to point to the desired repo and reinstall:

```bash
# Uninstall current
pip uninstall -y libero

# Install LIBERO-PRO
pip install -e /path/to/LIBERO-PRO --config-settings editable_mode=compat

# Or install original LIBERO
# pip install -e /path/to/LIBERO --config-settings editable_mode=compat
```

### LIBERO-PRO custom asset errors

Some LIBERO-PRO perturbation variants (e.g. `libero_spatial_with_red_box`) require custom 3D assets that may not be bundled. If you see `FileNotFoundError` for custom asset XMLs, use the pre-built variants that work out of the box:

- `libero_spatial_with_mug` (tested, works)
- `libero_spatial_with_milk` (tested, works)
- `libero_spatial_with_blue_stick` (tested, works)

### gym/gymnasium compatibility

LIBERO uses the older `gym` API. If you see gymnasium-related errors:

```bash
pip uninstall gymnasium
pip install 'gym==0.25.2'
```

### MuJoCo rendering issues

For headless servers without a display:

```python
# Use offscreen rendering only
env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    has_renderer=False,       # No on-screen rendering
    has_offscreen_renderer=True,
)
```

### Package version summary

For a working LIBERO / LIBERO-PRO setup, these versions are tested:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10 | Required by robosuite 1.4 |
| robosuite | 1.4.0 | LIBERO incompatible with 1.5+ |
| bddl | 1.0.1 | Task specification |
| robomimic | 0.2.0 | Dataset format |
| gym | 0.25.2 | Older API required |
| mujoco | >=2.3.0 | Simulation engine |
| future | 0.18.2 | Python 2/3 compat |
| easydict | 1.9 | Config handling |
| LIBERO-PRO | 0.1.0 | Drop-in replacement for LIBERO |
| opencv-python | >=4.0 | For `--render` display |
| imageio | >=2.0 | For `--save-video` recording |
