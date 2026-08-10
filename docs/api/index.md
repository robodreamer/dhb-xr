# API Reference

This section provides comprehensive documentation for all DHB-XR modules and functions.

## Core Modules

### Core Types and Geometry
- [Core Types](core.md) - Enums, types, and constants

### Encoding and Decoding
- [DHB-DR Encoding](encoding.md) - Double-reflection Euler-based encoding
- [DHB Decoding](decoding.md) - Reconstruction from invariants

### Time-Invariant Representations
- [DHB-TI](time_invariant.md) - Speed-independent encoding via reparameterization

### Advanced Features
- [Optimization](optimization.md) - Trajectory adaptation and optimization
- [Motion Database](database.md) - Storage and similarity search
- [Primitive Programs](primitives.md) - Motion, gripper/tool actions, evidence, execution checks, recovery, and memory
- [Tokenization](tokenization.md) - VQ-VAE for discrete representations
- [Loss Functions](losses.md) - Imitation learning losses
- [Visualization](visualization.md) - Plotting and visualization utilities

## Usage Examples

```python
# Import main functions
from dhb_xr import encode_dhb_dr, decode_dhb_dr

# Import types
from dhb_xr.core.types import DHBMethod, EncodingMethod

# Import utilities
from dhb_xr.core import geometry as geom
from dhb_xr.utils import preprocessing, resampling
```

## Module Organization

DHB-XR is organized into the following main modules:

- `dhb_xr.core` - Core types, geometry, and utilities
- `dhb_xr.encoder` - Encoding functions (DHB-DR, DHB-QR, DHB-TI)
- `dhb_xr.decoder` - Decoding functions
- `dhb_xr.optimization` - Trajectory optimization and adaptation
- `dhb_xr.database` - Motion storage and retrieval
- `dhb_xr.primitives` - Typed manipulation primitives, sources, execution, and memory
- `dhb_xr.tokenization` - VQ-VAE tokenization
- `dhb_xr.losses` - Loss functions for imitation learning
- `dhb_xr.visualization` - Plotting utilities
- `dhb_xr.utils` - Additional utilities
