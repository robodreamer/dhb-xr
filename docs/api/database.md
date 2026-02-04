# Motion Database

::: dhb_xr.database

## Overview

Store and retrieve trajectories using invariant-based similarity search.

## Main Classes

### MotionDatabase

::: dhb_xr.database.motion_db.MotionDatabase

## Usage Example

```python
from dhb_xr.database.motion_db import MotionDatabase
from dhb_xr.encoder.dhb_dr import encode_dhb_dr

# Create database
db = MotionDatabase()

# Add trajectories
for trajectory in trajectories:
    invariants = encode_dhb_dr(trajectory['positions'], trajectory['quaternions'])
    db.add_trajectory(invariants, metadata=trajectory['metadata'])

# Search similar trajectories
query_invariants = encode_dhb_dr(query_positions, query_quaternions)
similar = db.find_similar(query_invariants, k=5)
```