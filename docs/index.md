# DHB-XR

**SE(3) invariant trajectory encoding for robotics and VLA applications**

DHB-XR (DHB Extended Representations) provides SE(3) invariant trajectory encoding for robotics and VLA (Vision-Language-Action) applications. The library implements various DHB (Double-Helix Based) methods for encoding trajectories into invariant representations that are independent of global position and orientation changes.

## Research Background

This library implements the double-reflection (DHB-DR) and quaternion-relative (DHB-QR) invariant representations for rigid-body motion trajectories on SE(3), as described in the manuscript **"Double-Reflection DHB Invariant Representation on SE(3)"**.

**Author:** Andy Park (andy.park@thehumanoid.ai)

**Version:** 0.1.0

## Features

- **DHB-DR** (Double-Reflection): Euler-based invariants (4 values per component)
- **DHB-QR** (Quaternion-Relative): Quaternion-based invariants (5 values per component), no gimbal lock
- **DHB-TI** (Time-Invariant): Speed-independent representations via arc-length reparameterization
- **Trajectory adaptation**: Resample and decode with new initial pose; optional batched optimization
- **Tokenization**: VQ-VAE / RVQ for DHB-Token (VLA action representation)
- **Motion database**: Storage, similarity (L2, DTW), and retrieval
- **Imitation learning**: Invariant matching, SE(3) geodesic, and hybrid losses
- **Robust encoding**: Handles reversals, zero-motion segments, and frame alignment issues

## Quick start

```bash
pip install dhb_xr
# or with extras: pip install dhb_xr[gpu,database,optimization]
```

```python
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod
import numpy as np

# Create sample trajectory data
n = 50
positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))  # identity orientation

# Encode trajectory to invariants
result = encode_dhb_dr(
    positions, quaternions,
    method=EncodingMethod.POSITION,
    use_default_initial_frames=True,
    dhb_method=DHBMethod.DOUBLE_REFLECTION
)

# Decode back to poses
decoded = decode_dhb_dr(
    result["linear_motion_invariants"],
    result["angular_motion_invariants"],
    result["initial_pose"],
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    drop_padded=True
)

print(f"Original shape: {positions.shape}")
print(f"Decoded shape: {decoded['positions'].shape}")
```

## Examples

See [Examples](examples_walkthrough.md) for a walkthrough of the example scripts and how to run them.

## API Reference

::: dhb_xr

## License

MIT License.

**Copyright (c) 2025 Andy Park <andy.park@thehumanoid.ai>**