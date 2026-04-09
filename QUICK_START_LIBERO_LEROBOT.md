# Quick Start: LIBERO to LeRobot Conversion

## TL;DR

Convert LIBERO datasets to LeRobot format for StarVLA training in 3 steps:

### Step 1: Install Dependencies

```bash
# Ensure h5py, pandas, and scipy are installed
pip install h5py pandas scipy pillow numpy
```

### Step 2: Run Converter

```bash
python examples/integration/libero_to_lerobot.py \
    --libero-root /path/to/libero_spatial \
    --output-root ./libero_lerobot \
    --all-tasks
```

### Step 3: Use with StarVLA

```python
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

dataset = LeRobotSingleDataset(
    dataset_path="./libero_lerobot",
    modality_configs={
        "video": ModalityConfig(delta_indices=[0], modality_keys=["video.primary_image", "video.wrist_image"]),
        "state": ModalityConfig(delta_indices=list(range(-16, 0)), modality_keys=[...]),
        "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=[...]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    },
    embodiment_tag="libero_franka",
)
```

## Output Format

```
libero_lerobot/
├── meta/
│   ├── modality.json      <- State/action/video structure
│   ├── info.json          <- Dataset metadata
│   ├── episodes.jsonl     <- Episode metadata
│   ├── tasks.jsonl        <- Task descriptions
│   └── stats_gr00t.json   <- Statistics for normalization
└── data/
    └── chunk_000000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

## Key Data Conversions

| Aspect | LIBERO | LeRobot |
|--------|--------|---------|
| Format | HDF5 files | Parquet files |
| State | robot_states[t, 2:5] for EE pos, [5:9] for quat | observation.state [x,y,z,roll,pitch,yaw,gripper] |
| Action | actions[t] | action [x,y,z,roll,pitch,yaw,gripper] |
| Video | agentview_rgb, wrist_rgb (raw numpy) | PNG-compressed bytes in parquet |
| Organization | One file per task | Chunked by episode_index//16 |

## Common Commands

### Convert single task
```bash
python examples/integration/libero_to_lerobot.py \
    --libero-root /data/libero_spatial \
    --output-root ./output \
    --task-name "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
```

### Test with limited demos
```bash
python examples/integration/libero_to_lerobot.py \
    --libero-root /data/libero_spatial \
    --output-root ./output_test \
    --all-tasks \
    --max-demos 5
```

### With DHB invariants
```bash
python examples/integration/libero_to_lerobot.py \
    --libero-root /data/libero_spatial \
    --output-root ./output \
    --all-tasks \
    --dhb-invariants ./dhb_features.pkl
```

## State Vector Details

```
state = [ee_x, ee_y, ee_z, roll, pitch, yaw, gripper]
         (m)   (m)   (m)   (rad) (rad) (rad)  (0-1)
```

- **Position**: From robot_states[:, 2:5] (meters)
- **Orientation**: Convert robot_states[:, 5:9] quaternion (w,x,y,z) → Euler angles
- **Gripper**: From robot_states[:, 0] (normalized 0-1)

## Verify Conversion

```bash
# Check metadata files exist
ls -la libero_lerobot/meta/

# Inspect a parquet file
python3 << 'EOF'
import pandas as pd
df = pd.read_parquet("libero_lerobot/data/chunk_000000/episode_000000.parquet")
print(df.columns)
print(df.shape)
print(df['observation.state'].iloc[0].shape)
EOF
```

## Integration with StarVLA

The converter outputs match StarVLA's expected structure. To train:

1. Update dataset registry in `vendor/starVLA/examples/LIBERO/train_files/data_registry/data_config.py`
2. Point data_dir to the converted dataset in training config
3. Run StarVLA training normally

## Troubleshooting

**Q: "Demo X not found" error**
A: Check HDF5 structure with `h5py`. LIBERO files may have different structure than expected.

**Q: Out of memory during stats computation**
A: Use `--max-demos` to test with fewer episodes, or increase RAM.

**Q: StarVLA dataloader fails**
A: Verify all 5 metadata files exist in `meta/` directory.

---

See `LIBERO_TO_LEROBOT_GUIDE.md` for detailed documentation.
