# DHB Decoding

Decoding functions live in `dhb_xr.decoder.dhb_dr` and are also re-exported
from the top-level `dhb_xr` package.

## Main Functions

### decode_dhb_dr

```python
def decode_dhb_dr(
    linear_motion_invariants: np.ndarray,
    angular_motion_invariants: np.ndarray,
    initial_pose: Dict[str, np.ndarray],
    method: EncodingMethod = EncodingMethod.POSITION,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
    robust_mode: bool = False,
    validate_initial_pose: bool = False,
    return_diagnostics: bool = False,
) -> Dict[str, Any]:
```

Decode DHB-DR invariants back to trajectory.

**Parameters:**

- `linear_motion_invariants`: Linear motion invariants (M, 4) or (M, 3)
- `angular_motion_invariants`: Angular motion invariants (M, 4) or (M, 3)
- `initial_pose`: Initial pose dictionary with 'position' and 'quaternion' keys
- `method`: Decoding method (must match encoding method)
- `dhb_method`: DHB variant (must match encoding)
- `drop_padded`: Remove padding frames from output
- `robust_mode`: Enable robustness features
- `validate_initial_pose`: Validate initial pose format
- `return_diagnostics`: Return diagnostic information

**Returns:**

Dictionary containing:
- `positions`: Reconstructed positions (N, 3)
- `quaternions`: Reconstructed quaternions (N, 4)
- Optional diagnostics if `return_diagnostics=True`

## Round-trip Example

```python
import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod

# Original trajectory
positions = np.random.randn(50, 3) * 0.01
positions = np.cumsum(positions, axis=0)
quaternions = np.tile([1, 0, 0, 0], (50, 1))

# Encode
result = encode_dhb_dr(
    positions, quaternions,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION
)

# Decode
decoded = decode_dhb_dr(
    result['linear_motion_invariants'],
    result['angular_motion_invariants'],
    result['initial_pose'],
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION,
    drop_padded=True
)

# Check accuracy
error = np.linalg.norm(positions - decoded['positions'], axis=1)
rmse = np.sqrt(np.mean(error**2))
print(f"RMSE: {rmse * 1e6:.2f} μm")
```
