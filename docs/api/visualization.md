# Visualization

Visualization utilities live under `dhb_xr.visualization`.

## Overview

Utilities for plotting and visualizing DHB trajectories and invariants.

## Main Functions

### plot_se3_trajectory

Plotting helpers live in `dhb_xr.visualization.plot`.

## Usage Example

```python
from dhb_xr.visualization.plot import plot_se3_trajectory
import matplotlib.pyplot as plt

# Plot trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

plot_se3_trajectory(
    ax, positions, quaternions,
    color='blue', label='Trajectory'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```
