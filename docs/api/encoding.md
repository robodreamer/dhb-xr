# DHB-DR Encoding

Encoding functions live in `dhb_xr.encoder.dhb_dr` and are also re-exported
from the top-level `dhb_xr` package.

## Main Functions

### encode_dhb_dr

```python
def encode_dhb_dr(
    positions: np.ndarray,
    quaternions: np.ndarray,
    init_pose: Optional[Dict[str, np.ndarray]] = None,
    method: EncodingMethod = EncodingMethod.POSITION,
    use_default_initial_frames: bool = True,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    robust_mode: bool = False,
    reversal_threshold: float = -0.9,
    zero_motion_threshold: float = 1e-6,
    validate_frames: bool = False,
    return_diagnostics: bool = False,
) -> Dict[str, Any]:
```

Encode trajectory to DHB-DR (Double-Reflection) invariants.

**Parameters:**

- `positions`: Trajectory positions (N, 3) array
- `quaternions`: Trajectory quaternions (N, 4) array in wxyz format
- `init_pose`: Optional initial pose override
- `method`: Initial frame computation method
- `use_default_initial_frames`: Use default frame initialization
- `dhb_method`: DHB variant (DOUBLE_REFLECTION or ORIGINAL)
- `robust_mode`: Enable robustness features
- `reversal_threshold`: Threshold for reversal detection
- `zero_motion_threshold`: Threshold for zero-motion detection
- `validate_frames`: Validate frame orthonormality
- `return_diagnostics`: Return diagnostic information

**Returns:**

Dictionary containing:
- `linear_motion_invariants`: Linear motion invariants (M, 4) or (M, 3)
- `angular_motion_invariants`: Angular motion invariants (M, 4) or (M, 3)
- `initial_pose`: Initial pose used for encoding
- Optional diagnostics if `return_diagnostics=True`

## Usage Example

```python
import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod

# Create trajectory data
positions = np.random.randn(100, 3) * 0.01
positions = np.cumsum(positions, axis=0)
quaternions = np.tile([1, 0, 0, 0], (100, 1))

# Encode to invariants
result = encode_dhb_dr(
    positions,
    quaternions,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    robust_mode=True
)

print(f"Linear invariants shape: {result['linear_motion_invariants'].shape}")
print(f"Angular invariants shape: {result['angular_motion_invariants'].shape}")
```
