# Core Types

Core type definitions live in `dhb_xr.core.types`.

## Key Classes and Enums

### DHBMethod

```python
from dhb_xr.core.types import DHBMethod

class DHBMethod(Enum):
    """DHB encoding method variants."""
    DOUBLE_REFLECTION = "double_reflection"  # 4 values per component (Euler-based)
    ORIGINAL = "original"                    # 3 values per component (original DHB)
```

### EncodingMethod

```python
from dhb_xr.core.types import EncodingMethod

class EncodingMethod(Enum):
    """Encoding method for initial frame computation."""
    POSITION = "pos"  # Position-based encoding: use initial position for frame origin
    VELOCITY = "vel"  # Velocity-based encoding: use first position difference for frame origin
```

### Usage

```python
from dhb_xr.core.types import DHBMethod, EncodingMethod

# Use in encoding functions
result = encode_dhb_dr(
    positions, quaternions,
    method=EncodingMethod.POSITION,
    dhb_method=DHBMethod.DOUBLE_REFLECTION
)
```
