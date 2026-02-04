# VLA Integration (LIBERO + RoboCASA)

This page shows how to use DHB-XR with popular Vision-Language-Action (VLA) benchmarks for trajectory encoding, retrieval, and simulation.

## Overview

DHB-XR provides thin adapter layers for:

- **LIBERO**: Lifelong robot learning benchmark (130 tasks)
- **RoboCASA**: Large-scale manipulation benchmark

The pipeline:

1. **Load** dataset episodes (HDF5 format)
2. **Extract** end-effector poses (positions, quaternions)
3. **Encode** to SE(3)-invariant DHB representations
4. **Retrieve** similar motions or **tokenize** for VLA training

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
    quaternions = episode["quaternions"]  # (N, 4) quaternions (x, y, z, w)
    metadata = episode["metadata"]        # demo_id, task info, etc.

    print(f"Episode {metadata['demo_id']}: {len(positions)} frames")
```

**Note:** LIBERO stores quaternions in `robot_states[:, 5:9]` as (w,x,y,z). The adapter automatically extracts and converts to (x,y,z,w) format.

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

## LIBERO Simulation

To run LIBERO tasks in simulation with DHB-XR trajectory adaptation:

### Installation

LIBERO requires specific package versions. We recommend using a separate conda environment:

```bash
# 1. Install Miniforge (if conda/mamba not available)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p ~/miniforge3

# 2. Create libero environment
~/miniforge3/bin/mamba create -n libero python=3.10 -y

# 3. Install LIBERO dependencies (specific versions required!)
~/miniforge3/bin/mamba run -n libero pip install robosuite==1.4.0 mujoco bddl==1.0.1 robomimic==0.2.0
~/miniforge3/bin/mamba run -n libero pip install future easydict hydra-core cloudpickle 'gym==0.25.2'

# 4. Clone and install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/Projects/repos/LIBERO
~/miniforge3/bin/mamba run -n libero pip install -e ~/Projects/repos/LIBERO --config-settings editable_mode=compat

# 5. Configure LIBERO paths
mkdir -p ~/.libero
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /home/$USER/Projects/repos/LIBERO/libero/libero
bddl_files: /home/$USER/Projects/repos/LIBERO/libero/libero/bddl_files
init_states: /home/$USER/Projects/repos/LIBERO/libero/libero/init_files
datasets: /home/$USER/Projects/data/libero
assets: /home/$USER/Projects/repos/LIBERO/libero/libero/assets
EOF

# 6. Install dhb_xr
~/miniforge3/bin/mamba run -n libero pip install dhb_xr[optimization]
```

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

## VQ-VAE Tokenization

For VLA training, tokenize invariants to discrete action tokens:

```python
from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig

config = DHBVLAPipelineConfig(
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

**Note:** Tokenization requires PyTorch: `pip install dhb_xr[gpu]`

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

LIBERO uses (w,x,y,z) format internally. The adapter converts to (x,y,z,w) for compatibility with scipy and transforms3d.

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
cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /path/to/LIBERO/libero/libero
bddl_files: /path/to/LIBERO/libero/libero/bddl_files
init_states: /path/to/LIBERO/libero/libero/init_files
datasets: /path/to/your/datasets
assets: /path/to/LIBERO/libero/libero/assets
EOF
```

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

For a working LIBERO setup, these versions are tested:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| robosuite | 1.4.0 |
| bddl | 1.0.1 |
| robomimic | 0.2.0 |
| gym | 0.25.2 |
| mujoco | >=2.3.0 |
| future | 0.18.2 |
| easydict | 1.9 |
