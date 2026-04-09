# LIBERO to LeRobot Converter Guide

## Overview

The `libero_to_lerobot.py` converter transforms LIBERO HDF5 datasets into LeRobot v2.0 format, which is required by StarVLA's dataloader for training vision-language-action models.

**Key Achievement**: Unblocks StarVLA training with LIBERO datasets by providing the correct data format.

## What is LeRobot?

LeRobot is a framework for robot learning developed by Hugging Face. It standardizes robot datasets with:
- Parquet-based storage for efficient columnar data access
- Standardized metadata describing modalities (vision, state, actions)
- Support for multiple embodiments and tasks in a single dataset
- Built-in statistics for normalization

## Data Format Conversions

### LIBERO HDF5 Format (Input)

LIBERO stores demonstrations in HDF5 files with the following structure:

```
pick_up_the_black_bowl_from_table_center.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions              [T, 7]        joint velocity targets
│   │   ├── robot_states         [T, 9]        [gripper(2), ee_pos(3), ee_quat(4)]
│   │   ├── agentview_rgb        [T, H, W, 3]  primary camera
│   │   └── wrist_rgb            [T, H, W, 3]  wrist camera
│   ├── demo_1/
│   └── ...
```

**Key Fields:**
- `actions`: [T, 7] - Joint velocity targets for Franka arm (6 DoF) + gripper
- `robot_states`: [T, 9] - Robot state vector:
  - [0:2]   - Gripper state (normalized position)
  - [2:5]   - End-effector XYZ position (meters)
  - [5:9]   - End-effector quaternion in (w, x, y, z) format
- `agentview_rgb`: [T, H, W, 3] - Agent-view RGB frames (uint8)
- `wrist_rgb`: [T, H, W, 3] - Wrist-mounted camera (optional)

### LeRobot v2.0 Format (Output)

The converter produces the following structure:

```
libero_lerobot/
├── meta/
│   ├── modality.json           State/action/video definitions
│   ├── info.json               Dataset metadata
│   ├── episodes.jsonl          Episode indices and lengths (JSONL)
│   ├── tasks.jsonl             Task descriptions (JSONL)
│   └── stats_gr00t.json        Dataset statistics (min/max/mean/std)
├── data/
│   ├── chunk_000000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── chunk_000001/
│   └── ...
└── videos/
    ├── chunk_000000/
    │   ├── episode_000000_primary_image.mp4
    │   ├── episode_000000_wrist_image.mp4
    │   └── ...
    └── ...
```

## Modality Definitions

### State Vector (Observation)

The state vector includes the end-effector pose:

```python
state = [
    x, y, z,                # EE position (m)
    roll, pitch, yaw,       # EE orientation (Euler angles, rad)
    gripper                 # Gripper opening (0-1)
]  # Total: 7 dimensions
```

**Conversion Details:**
- EE position: Directly from `robot_states[:, 2:5]`
- EE orientation: Quaternion → Euler angles (XYZ convention)
  - Input: (w, x, y, z) from LIBERO
  - Conversion: `scipy.spatial.transform.Rotation.from_quat().as_euler("xyz")`
  - Output: (roll, pitch, yaw) in radians
- Gripper: First component of `robot_states[:, 0:2]`

### Action Vector

For absolute control (ACTION_MODE = "abs"):

```python
action = [
    x, y, z,                # Target EE position (m)
    roll, pitch, yaw,       # Target EE orientation (rad)
    gripper                 # Target gripper (0-1)
]  # Total: 7 dimensions
```

For LIBERO, actions are the same as states (absolute end-effector positions).

### Video Modalities

Two camera views are captured:
1. **primary_image** (agentview_rgb): Main agent camera
2. **wrist_image** (wrist_rgb): Wrist-mounted camera

Default resolution: 512×384 pixels (adjustable)

## Parquet File Format

Each episode is stored in a single parquet file with the following columns:

| Column | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `observation.state` | [T, 7] | float32 | State trajectory |
| `action` | [T, 7] | float32 | Action trajectory |
| `observation.images.image` | [T] | bytes | Primary camera frames |
| `observation.images.wrist_image` | [T] | bytes | Wrist camera frames |
| `timestamp` | [T] | float32 | Frame timestamps (1/fps) |
| `task_index` | [T] | int32 | Task index |

Each frame in video columns is stored as PNG-encoded bytes for compression.

## Metadata Files

### modality.json

Defines the structure of state, action, and video modalities for StarVLA:

```json
{
  "state": {
    "x": {"start": 0, "end": 1},
    "y": {"start": 1, "end": 2},
    "z": {"start": 2, "end": 3},
    "roll": {"start": 3, "end": 4},
    "pitch": {"start": 4, "end": 5},
    "yaw": {"start": 5, "end": 6},
    "gripper": {"start": 6, "end": 7}
  },
  "action": { ... same structure ... },
  "video": {
    "primary_image": {"original_key": "observation.images.image"},
    "wrist_image": {"original_key": "observation.images.wrist_image"}
  },
  "annotation": {
    "human.action.task_description": {"original_key": "task_index"}
  }
}
```

This tells StarVLA's dataloader:
- Which columns in the parquet files correspond to each modality
- How to slice concatenated state/action vectors
- How to interpret video columns

### info.json

Dataset-level metadata:

```json
{
  "name": "libero_lerobot",
  "version": "2.0",
  "data_path": "data/{episode_chunk:06d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/{episode_chunk:06d}/episode_{episode_index:06d}_{video_key}.mp4",
  "chunks_size": 16,
  "num_episodes": 1234,
  "num_frames": 567890,
  "num_tasks": 4,
  "features": { ... field schemas ... }
}
```

### episodes.jsonl

One JSON object per line, one for each episode:

```jsonl
{"episode_index": 0, "length": 250, "task_index": 0}
{"episode_index": 1, "length": 260, "task_index": 0}
```

### tasks.jsonl

One JSON object per line, one for each task:

```jsonl
{"task_index": 0, "task": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"}
{"task_index": 1, "task": "pick_up_the_black_bowl_from_table_edge_and_place_it_on_the_plate"}
```

### stats_gr00t.json

Dataset statistics used for normalization:

```json
{
  "__format_version": 2,
  "__cache_config": {"mode": "abs"},
  "statistics": {
    "observation.state": {
      "min": [-0.5, -0.5, 0.1, ...],
      "max": [0.5, 0.5, 1.0, ...],
      "mean": [0.0, 0.0, 0.5, ...],
      "std": [0.2, 0.2, 0.15, ...],
      "q01": [-0.4, -0.4, 0.15, ...],
      "q99": [0.4, 0.4, 0.95, ...]
    },
    ...
  }
}
```

## DHB Invariants Integration

The converter supports precomputed DHB invariants as optional additional features:

```python
# Load DHB features computed by dhb_xr
dhb_features = pickle.load(open("dhb_invariants.pkl", "rb"))

# Run converter with DHB features
python libero_to_lerobot.py \
    --libero-root /data/libero/spatial \
    --output-root /data/libero_lerobot \
    --dhb-invariants /data/dhb_invariants.pkl \
    --all-tasks
```

DHB features are added to the parquet files as columns prefixed with `dhb.`:
- `dhb.twist_invariant`: SE(3)-invariant twist features
- `dhb.wrench_invariant`: SE(3)-invariant force/torque features
- etc.

## Usage

### Basic Usage

Convert a single task:

```bash
python libero_to_lerobot.py \
    --libero-root /path/to/libero_spatial \
    --output-root /path/to/output/libero_lerobot \
    --task-name "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
```

Convert all tasks:

```bash
python libero_to_lerobot.py \
    --libero-root /path/to/libero_spatial \
    --output-root /path/to/output/libero_lerobot \
    --all-tasks
```

With DHB invariants:

```bash
python libero_to_lerobot.py \
    --libero-root /path/to/libero_spatial \
    --output-root /path/to/output/libero_lerobot \
    --all-tasks \
    --dhb-invariants /path/to/dhb_invariants.pkl
```

### Command-line Options

```
--libero-root           Root directory with LIBERO HDF5 files
--output-root           Output directory for LeRobot dataset
--task-name             Specific task to convert (optional)
--all-tasks             Convert all tasks in libero-root
--dhb-invariants        Path to precomputed DHB invariants (optional)
--max-demos             Maximum demos to convert per task (testing)
--compute-stats         Compute stats after conversion (default: True)
```

## Integration with StarVLA

After conversion, update your StarVLA configuration:

### 1. Register the Dataset

In `vendor/starVLA/examples/LIBERO/train_files/data_registry/data_config.py`:

```python
# Update the dataset registry
DATASET_NAMED_MIXTURES = {
    "libero_spatial_lerobot": [
        ("libero_spatial_lerobot", 1.0, "libero_franka"),
    ],
    # ... existing entries ...
}
```

### 2. Update Training Configuration

In your training YAML (e.g., `starvla_cotrain_libero.yaml`):

```yaml
dataset_config:
  dataset_names:
    - "libero_spatial_lerobot"  # Your converted dataset
  data_dir: "/path/to/libero_lerobot"
```

### 3. Train StarVLA

```bash
python train.py --config-name starvla_cotrain_libero \
    dataset_config.data_dir=/path/to/libero_lerobot
```

## Verification

After conversion, verify the dataset:

```python
from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

# Load the converted dataset
dataset = LeRobotSingleDataset(
    dataset_path="/path/to/libero_lerobot",
    modality_configs={...},  # from data_config.py
    embodiment_tag=EmbodimentTag.FRANKA,
)

# Check dataset size
print(f"Total steps: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Action shape: {sample['action'].shape}")
print(f"Image shape: {sample['image'][0].shape}")
```

## Performance Considerations

### File Organization

The converter uses chunking (16 episodes per chunk) to:
- Keep parquet files manageable in size
- Enable parallel loading of different chunks
- Balance memory usage and I/O efficiency

### Storage

Typical storage requirements:
- Video frames: ~5-10 MB per episode (depending on resolution and compression)
- Parquet data: ~100-200 KB per episode (state/action)
- Total for 1000 episodes: ~6-12 GB

### Loading Speed

LeRobot's parquet format provides:
- Fast columnar data access (only load needed columns)
- Efficient memory usage with lazy loading
- Native support for parallel episode sampling

## Troubleshooting

### Issue: "Demo X not found"

**Solution**: Check the LIBERO HDF5 structure. Use:
```bash
python3 -c "
import h5py
with h5py.File('your_file.hdf5', 'r') as f:
    print(list(f['data'].keys())[:10])
"
```

### Issue: Memory error during statistics computation

**Solution**: The converter loads all parquet files into memory. For large datasets:
1. Reduce episodes: Use `--max-demos 100` for testing
2. Split conversion: Convert by task separately
3. Increase RAM or reduce video resolution

### Issue: StarVLA dataloader fails to load dataset

**Solution**: Ensure all metadata files exist:
```bash
ls -la /path/to/libero_lerobot/meta/
# Should have: modality.json, info.json, episodes.jsonl, tasks.jsonl, stats_gr00t.json
```

Check modality.json matches StarVLA's expectations in data_config.py.

## Advanced Features

### Custom Action Modes

To support delta or relative actions instead of absolute:

```python
# In converter, modify ACTION_MODE:
ACTION_MODE = "delta"  # or "rel"

# Rerun conversion
python libero_to_lerobot.py --all-tasks
```

Then update StarVLA's data_config:
```python
"action_mode": "delta",  # Match the converter mode
```

### Multiple Embodiments

To support different robot morphologies:

```python
# Convert with custom embodiment tag
# (Currently hardcoded to Franka, can be extended)
```

## References

- **LeRobot**: https://github.com/huggingface/lerobot
- **StarVLA**: Provided in `vendor/starVLA/`
- **LIBERO**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **LIBERO-PRO**: https://github.com/Zxy-MLlab/LIBERO-PRO

## Citation

If you use this converter in your research, please cite:

```bibtex
@inproceedings{libero,
  title={LIBERO: A Benchmark for Language-Instructed Embodied Robot Manipulation},
  author={Bisk, Yonatan and others},
  booktitle={CoRL},
  year={2023}
}

@article{lerobot,
  title={LeRobot: A Platform for Scalable End-to-End Robotic Learning},
  author={Cadene, Romain and others},
  year={2024}
}
```

---

**Last Updated**: 2026-04-09
**Author**: Generated for DHB-XR + StarVLA integration
