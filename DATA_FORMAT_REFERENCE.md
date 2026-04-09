# Data Format Reference: LIBERO → LeRobot Conversion

## Complete Data Flow Diagram

```
LIBERO HDF5 File
  demo_0/
    ├── actions[T, 7]              robot_states[T, 9]          agentview_rgb[T, H, W, 3]
    │   └─ Joint velocities        │                           │
    │                              ├─ [0:2] gripper_state      └─ Uint8 RGB frames
    │                              ├─ [2:5] ee_pos             
    │                              └─ [5:9] ee_quat_wxyz       wrist_rgb[T, H, W, 3]
    │                                       (w, x, y, z)        └─ Uint8 RGB frames
    │
    ↓ CONVERSION
    
    ├─ Parse EE Position           Convert Quaternion          Convert Images
    │  ee_pos = states[:, 2:5]    quat = states[:, 5:9]       frames = uint8 arrays
    │  Shape: [T, 3]              Convert (w,x,y,z) →         Shape: [T, H, W, 3]
    │                             (x,y,z,w) format
    │                             ↓
    │                             Apply scipy.spatial.transform
    │                             .Rotation.from_quat()
    │                             .as_euler("xyz")
    │                             ↓
    │                             euler_angles [T, 3]
    │                             (roll, pitch, yaw) in radians
    │
    ├─ CONCATENATE STATE VECTOR [T, 7]
    │  [ee_x, ee_y, ee_z, roll, pitch, yaw, gripper]
    │
    └─ GENERATE PARQUET DATAFRAME
       observation.state: [T, 7] float32
       action:           [T, 7] float32
       observation.images.image:       [T] bytes (PNG)
       observation.images.wrist_image: [T] bytes (PNG)
       timestamp:        [T] float32
       task_index:       [T] int32
       
       ↓
       
       SAVE TO PARQUET
       data/chunk_000000/episode_000000.parquet


LeRobot v2.0 Dataset Structure
  ├── meta/
  │   ├── modality.json          ← Defines the structure
  │   ├── info.json              ← Dataset metadata
  │   ├── episodes.jsonl         ← Episode list
  │   ├── tasks.jsonl            ← Task descriptions
  │   └── stats_gr00t.json       ← Statistics
  └── data/
      └── chunk_000000/
          └── episode_000000.parquet  ← Converted data
```

## Field-by-Field Conversion Reference

### Position (meters)

```
LIBERO:  robot_states[:, 2:5]  →  [x, y, z]  (float64)
LeRobot: state[0:3] / action[0:3]  (float32)

Example:
  robot_states[0, 2:5] = [0.3, 0.5, 0.1]
  → state[0, 0:3] = [0.3, 0.5, 0.1]
```

### Orientation (quaternion → Euler)

```
LIBERO:  robot_states[:, 5:9]  →  [w, x, y, z]  (float64)
                                   ↓
                                   Normalize: q = q / ||q||
                                   ↓
                                   scipy.spatial.transform.Rotation.from_quat(
                                     [x, y, z, w]  # Convert format
                                   ).as_euler("xyz", degrees=False)
                                   ↓
LeRobot: state[3:6] / action[3:6]  →  [roll, pitch, yaw]  (float32, radians)

Example:
  robot_states[0, 5:9] = [1.0, 0.0, 0.0, 0.0]  (identity)
  → Normalize: [1.0, 0.0, 0.0, 0.0]
  → Reformat: [0.0, 0.0, 0.0, 1.0]
  → Euler XYZ: [0.0, 0.0, 0.0]  (roll, pitch, yaw)
  → state[0, 3:6] = [0.0, 0.0, 0.0]
```

### Gripper State

```
LIBERO:  robot_states[:, 0]  →  scalar  (float64)
LeRobot: state[6] / action[6]  →  float32

Example:
  robot_states[0, 0] = 0.8
  → state[0, 6] = 0.8
  → action[0, 6] = 0.8
```

### Video Frames

```
LIBERO:  agentview_rgb[T, H, W, 3]     wrist_rgb[T, H, W, 3]
         Shape: [250, 384, 512, 3]     Shape: [250, 384, 512, 3]
         Dtype: uint8                   Dtype: uint8
         ↓
         Iterate each frame
         Resize to [384, 512] if needed
         Convert to PIL Image
         Save to PNG bytes
         ↓
LeRobot: observation.images.image[T]    observation.images.wrist_image[T]
         Each element: bytes (PNG)
         Type: list of byte strings

Code example:
  from PIL import Image
  from io import BytesIO
  
  frame = agentview_rgb[0]  # [384, 512, 3]
  img = Image.fromarray(frame, mode='RGB')
  buffer = BytesIO()
  img.save(buffer, format='PNG')
  png_bytes = buffer.getvalue()  # bytes object
```

### Timestamps

```
LIBERO:  Implicit (frame indices)
         Assume 30 FPS
         ↓
LeRobot: timestamp[T]  →  [0.0, 1/30, 2/30, ..., (T-1)/30]
         Dtype: float32
         Units: seconds

Calculation:
  fps = 30.0
  timestamps = np.arange(T, dtype=np.float32) / fps
  
Example:
  T = 250 frames
  timestamps = [0.0, 0.0333, 0.0667, ..., 8.3]
```

### Task Index

```
LIBERO:  Task identification via filename
         "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
         ↓
LeRobot: task_index[T]  →  scalar int32
         All frames in episode have same task_index

Example:
  task_index = 0  (first task)
  task_index[:] = 0  (all T frames)
```

## Complete Parquet Schema

```python
{
  "observation.state": {
    "dtype": float32,
    "shape": [T, 7],
    "columns": [0, 1, 2, 3, 4, 5, 6],
    "meaning": ["x(m)", "y(m)", "z(m)", "roll(rad)", "pitch(rad)", "yaw(rad)", "gripper(0-1)"]
  },
  "action": {
    "dtype": float32,
    "shape": [T, 7],
    "columns": [0, 1, 2, 3, 4, 5, 6],
    "meaning": ["x(m)", "y(m)", "z(m)", "roll(rad)", "pitch(rad)", "yaw(rad)", "gripper(0-1)"]
  },
  "observation.images.image": {
    "dtype": bytes,
    "shape": [T],
    "format": "PNG-compressed uint8 RGB",
    "resolution": [512, 384]  # width, height
  },
  "observation.images.wrist_image": {
    "dtype": bytes,
    "shape": [T],
    "format": "PNG-compressed uint8 RGB",
    "resolution": [512, 384]
  },
  "timestamp": {
    "dtype": float32,
    "shape": [T],
    "units": "seconds",
    "calculation": "frame_index / 30.0"
  },
  "task_index": {
    "dtype": int32,
    "shape": [T],
    "value": "constant across episode"
  }
}
```

## Modality Metadata Slicing

The `modality.json` file defines how to slice state/action vectors:

```json
{
  "state": {
    "x": {"start": 0, "end": 1},           ← state[:, 0:1]
    "y": {"start": 1, "end": 2},           ← state[:, 1:2]
    "z": {"start": 2, "end": 3},           ← state[:, 2:3]
    "roll": {"start": 3, "end": 4},        ← state[:, 3:4]
    "pitch": {"start": 4, "end": 5},       ← state[:, 4:5]
    "yaw": {"start": 5, "end": 6},         ← state[:, 5:6]
    "gripper": {"start": 6, "end": 7}      ← state[:, 6:7]
  },
  "action": {
    "x": {"start": 0, "end": 1},
    "y": {"start": 1, "end": 2},
    "z": {"start": 2, "end": 3},
    "roll": {"start": 3, "end": 4},
    "pitch": {"start": 4, "end": 5},
    "yaw": {"start": 5, "end": 6},
    "gripper": {"start": 6, "end": 7}
  }
}
```

StarVLA uses these slices to extract individual components during training.

## Statistics Computation

```python
# Collect all values across all episodes
for each episode in dataset:
    for each column (e.g., "observation.state"):
        collect all [T, dim] arrays

# Concatenate across all episodes
all_data = np.concatenate(collected_arrays, axis=0)  # [Total, dim]

# Compute statistics
stats = {
  "min": np.min(all_data, axis=0),              # [dim]
  "max": np.max(all_data, axis=0),              # [dim]
  "mean": np.mean(all_data, axis=0),            # [dim]
  "std": np.std(all_data, axis=0),              # [dim]
  "q01": np.quantile(all_data, 0.01, axis=0),  # [dim]
  "q99": np.quantile(all_data, 0.99, axis=0)   # [dim]
}

# Output format: stats_gr00t.json
{
  "__format_version": 2,
  "__cache_config": {"mode": "abs"},
  "statistics": {
    "observation.state": {
      "min": [-0.5, -0.5, 0.1, ...],
      "max": [0.5, 0.5, 1.0, ...],
      ...
    },
    "action": { ... }
  }
}
```

## Chunking Strategy

```
Episodes are organized into chunks for efficient I/O:

CHUNK_SIZE = 16 episodes per chunk

Episode Index  →  Chunk Index
       0-15          0
      16-31          1
      32-47          2
      ...

File Organization:
  data/chunk_000000/
    episode_000000.parquet
    episode_000001.parquet
    ...
    episode_000015.parquet
  data/chunk_000001/
    episode_000016.parquet
    ...
```

## DHB Invariants Integration

If precomputed DHB features are provided:

```python
# Load DHB features
dhb_features = pickle.load(open("dhb_invariants.pkl"))
# Format: {episode_index: {"feature_name": [T, dim], ...}}

# Add to dataframe
for episode in dataset:
    if episode_index in dhb_features:
        dhb_feats = dhb_features[episode_index]
        for feature_name, values in dhb_feats.items():
            df[f"dhb.{feature_name}"] = [values]

# Parquet will have additional columns:
# dhb.twist_invariant
# dhb.wrench_invariant
# etc.
```

## Verification Checklist

When verifying a converted dataset:

```bash
# 1. Check all metadata files exist
✓ meta/modality.json
✓ meta/info.json
✓ meta/episodes.jsonl
✓ meta/tasks.jsonl
✓ meta/stats_gr00t.json

# 2. Verify modality.json structure
✓ Contains "state", "action", "video" keys
✓ Each has correct start/end indices
✓ Total dimension matches (should be 7 for LIBERO)

# 3. Check episodes.jsonl format
✓ One episode per line (JSONL)
✓ Each has episode_index, length, task_index
✓ Episode indices are sequential (0, 1, 2, ...)
✓ Lengths match actual parquet data

# 4. Verify parquet files
✓ Correct columns present
✓ observation.state shape [T, 7]
✓ action shape [T, 7]
✓ Image columns contain bytes
✓ Timestamps are sequential

# 5. Validate statistics
✓ stats_gr00t.json has __format_version: 2
✓ Statistics exist for all columns
✓ Stats contain [min, max, mean, std, q01, q99]
```

---

**Last Updated**: 2026-04-09
