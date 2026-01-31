"""Imitation learning losses: invariant, geodesic, hybrid."""

from dhb_xr.losses.invariant_loss import invariant_matching_loss
from dhb_xr.losses.geodesic_loss import so3_geodesic_loss, se3_geodesic_loss

__all__ = [
    "invariant_matching_loss",
    "so3_geodesic_loss",
    "se3_geodesic_loss",
    "hybrid_invariant_pose_loss",
]

try:
    from dhb_xr.losses.hybrid_loss import hybrid_invariant_pose_loss
except ImportError:
    hybrid_invariant_pose_loss = None
