# Loss Functions

::: dhb_xr.losses

## Overview

Loss functions for imitation learning with DHB invariants.

## Main Functions

### invariant_matching_loss

::: dhb_xr.losses.invariant_loss

### se3_geodesic_loss

::: dhb_xr.losses.geodesic_loss

### hybrid_invariant_pose_loss

::: dhb_xr.losses.hybrid_loss

## Usage Example

```python
from dhb_xr.losses.invariant_loss import invariant_matching_loss
from dhb_xr.encoder.dhb_dr import encode_dhb_dr

# Encode demonstrations and predictions
demo_inv = encode_dhb_dr(demo_pos, demo_quat)
pred_inv = encode_dhb_dr(pred_pos, pred_quat)

# Compute invariant matching loss
loss = invariant_matching_loss(
    demo_inv['linear_motion_invariants'],
    demo_inv['angular_motion_invariants'],
    pred_inv['linear_motion_invariants'],
    pred_inv['angular_motion_invariants']
)
```