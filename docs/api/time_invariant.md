# DHB-TI: Time-Invariant Encoding

::: dhb_xr.encoder.dhb_ti

## Overview

DHB-TI (Time-Invariant) encoding creates speed-independent representations by reparameterizing trajectories by arc-length before encoding. This ensures that the same motion executed at different speeds produces similar invariants.

## Main Functions

### encode_dhb_dr_ti

```python
def encode_dhb_dr_ti(
    positions: np.ndarray,
    quaternions: np.ndarray,
    M: int,
    progress_kind: str = "hybrid",
    alpha: float = 0.5,
    method: EncodingMethod = EncodingMethod.POSITION,
    use_default_initial_frames: bool = True,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    robust_mode: bool = False,
) -> Dict[str, Any]:
```

Encode trajectory to time-invariant DHB-DR invariants.

**Parameters:**

- `positions`: Trajectory positions (N, 3)
- `quaternions`: Trajectory quaternions (N, 4)
- `M`: Number of points in reparameterized trajectory
- `progress_kind`: Progress measure ("translation", "angular", "hybrid")
- `alpha`: Weight for hybrid progress (0=translation, 1=angular)
- Other parameters same as `encode_dhb_dr`

### compute_progress

```python
def compute_progress(
    positions: np.ndarray,
    quaternions: np.ndarray,
    kind: str = "hybrid",
    alpha: float = 0.5,
) -> np.ndarray:
```

Compute progress along trajectory for reparameterization.

### resample_by_progress

```python
def resample_by_progress(
    positions: np.ndarray,
    quaternions: np.ndarray,
    M: int,
    progress: Optional[np.ndarray] = None,
    progress_kind: str = "hybrid",
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
```

Resample trajectory to M points at uniform progress intervals.

## Time-Invariance Example

```python
import numpy as np
from dhb_xr.encoder.dhb_ti import encode_dhb_dr_ti
from dhb_xr.core.types import EncodingMethod, DHBMethod

# Create base trajectory
t = np.linspace(0, 2*np.pi, 100)
positions = np.column_stack([np.cos(t), np.sin(t), t * 0.01])
quaternions = np.tile([1, 0, 0, 0], (100, 1))

# Encode at different speeds
speeds = [50, 100, 200]  # Different numbers of points
results = []

for M in speeds:
    result = encode_dhb_dr_ti(
        positions, quaternions, M,
        progress_kind="hybrid",
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION
    )
    results.append(result)

# Compare invariants (should be similar despite different speeds)
for i, (M, result) in enumerate(zip(speeds, results)):
    print(f"Speed {i+1} (M={M}): linear shape {result['linear_motion_invariants'].shape}")

# Invariants should be very similar
diff_01 = np.linalg.norm(
    results[0]['linear_motion_invariants'][:50] -
    results[1]['linear_motion_invariants']
)
print(f"Difference between speed 1 and 2: {diff_01:.6f}")
```