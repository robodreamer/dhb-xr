"""Hybrid invariant + pose-space loss for imitation learning."""

import numpy as np
from typing import Optional

from dhb_xr.losses.invariant_loss import invariant_matching_loss
from dhb_xr.losses.geodesic_loss import se3_geodesic_loss_np


def hybrid_invariant_pose_loss(
    pred_positions: np.ndarray,
    pred_quaternions: np.ndarray,
    demo_positions: np.ndarray,
    demo_quaternions: np.ndarray,
    pred_invariants: Optional[np.ndarray] = None,
    demo_invariants: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    beta: float = 1.0,
) -> float:
    """
    alpha * invariant_loss + (1-alpha) * pose_loss.
    If pred_invariants/demo_invariants are None, only pose loss is used (alpha ignored for invariant part).
    """
    loss_pose = 0.0
    n = len(pred_positions)
    assert n == len(demo_positions) and n == len(pred_quaternions) and n == len(demo_quaternions)
    for i in range(n):
        loss_pose += se3_geodesic_loss_np(
            pred_positions[i], pred_quaternions[i],
            demo_positions[i], demo_quaternions[i],
            beta=beta,
        )
    if pred_invariants is not None and demo_invariants is not None:
        loss_inv = invariant_matching_loss(pred_invariants, demo_invariants)
        return float(alpha * loss_inv + (1 - alpha) * loss_pose)
    return float(loss_pose)
