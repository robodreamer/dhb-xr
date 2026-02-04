# Quickstart

This guide will get you started with DHB-XR in 5 minutes.

## Basic Usage

### 1. Import and Setup

```python
import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod
```

### 2. Create Sample Trajectory Data

```python
# Create a simple helical trajectory
n = 100
t = np.linspace(0, 4*np.pi, n)
positions = np.column_stack([
    0.3 * np.cos(t),      # x: circular motion
    0.3 * np.sin(t),      # y: circular motion
    0.01 * t             # z: linear motion
])

# Identity quaternions (no rotation)
quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
```

### 3. Encode to Invariants

```python
# Encode trajectory to DHB-DR invariants
result = encode_dhb_dr(
    positions,
    quaternions,
    method=EncodingMethod.POSITION,
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION
)

print("Invariant shapes:")
print(f"Linear: {result['linear_motion_invariants'].shape}")
print(f"Angular: {result['angular_motion_invariants'].shape}")
```

### 4. Decode Back to Trajectory

```python
# Decode invariants back to trajectory
decoded = decode_dhb_dr(
    result["linear_motion_invariants"],
    result["angular_motion_invariants"],
    result["initial_pose"],
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    drop_padded=True
)

print(f"Original positions shape: {positions.shape}")
print(f"Decoded positions shape: {decoded['positions'].shape}")
```

### 5. Verify Round-trip Accuracy

```python
# Check reconstruction accuracy
pos_error = np.linalg.norm(positions - decoded['positions'], axis=1)
rmse = np.sqrt(np.mean(pos_error**2))
print(f"RMSE: {rmse * 1e6:.2f} μm")

# Check orientation accuracy
quat_error = np.linalg.norm(quaternions - decoded['quaternions'], axis=1)
quat_rmse = np.sqrt(np.mean(quat_error**2))
print(f"Quaternion RMSE: {quat_rmse:.6f}")
```

## SE(3) Invariance Demonstration

DHB invariants are invariant to global SE(3) transformations:

```python
from dhb_xr.core import geometry as geom

# Apply random SE(3) transformation to trajectory
R = geom.axis_angle_to_rot(np.random.randn(3))  # Random rotation
t = np.random.randn(3) * 0.1  # Random translation

# Transform trajectory
positions_transformed = (R @ positions.T).T + t
quaternions_transformed = np.array([
    geom.rot_to_quat(R @ geom.quat_to_rot(q)) for q in quaternions
])

# Encode transformed trajectory
result_transformed = encode_dhb_dr(
    positions_transformed,
    quaternions_transformed,
    method=EncodingMethod.POSITION,
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION
)

# Compare invariants (should be very similar)
lin_diff = np.linalg.norm(
    result['linear_motion_invariants'] - result_transformed['linear_motion_invariants']
)
ang_diff = np.linalg.norm(
    result['angular_motion_invariants'] - result_transformed['angular_motion_invariants']
)

print(f"Linear invariant difference: {lin_diff:.6f}")
print(f"Angular invariant difference: {ang_diff:.6f}")
```

## Next Steps

- Explore [API Reference](api/index.md) for detailed function documentation
- Check out [Examples](examples_walkthrough.md) for more advanced usage
- Learn about [Time-Invariant Encoding](api/time_invariant.md) for speed-independent representations