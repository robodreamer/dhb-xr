# Trajectory Optimization

## Overview

DHB-XR provides optimization-based trajectory adaptation using CasADi and Fatrop solvers.

## Main Functions

### batched_decode_dhb_dr

::: dhb_xr.optimization.cusadi_solver.batched_decode_dhb_dr

## Usage Example

```python
from dhb_xr.optimization.casadi_solver import CasadiTrajectoryOptimizer
import numpy as np

# Create optimizer
optimizer = CasadiTrajectoryOptimizer(batch_size=1)

# Sample data
lin_inv = np.random.randn(1, 20, 4)
ang_inv = np.random.randn(1, 20, 4)
init_poses = [{"position": np.zeros(3), "quaternion": np.array([1, 0, 0, 0])}]

# Optimize trajectory
positions, quaternions = optimizer.forward(lin_inv, ang_inv, init_poses)
```