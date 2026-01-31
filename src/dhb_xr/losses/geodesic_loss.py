"""SO(3) and SE(3) geodesic losses."""

import numpy as np
from typing import Optional

from dhb_xr.core import geometry as geom

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def so3_geodesic_loss_np(R_pred: np.ndarray, R_demo: np.ndarray) -> float:
    """||Log(R_pred^T R_demo)||^2. R_pred, R_demo: (3, 3) or (N, 3, 3)."""
    R_pred = np.asarray(R_pred)
    R_demo = np.asarray(R_demo)
    if R_pred.ndim == 2:
        R_pred = R_pred.reshape(1, 3, 3)
        R_demo = R_demo.reshape(1, 3, 3)
    R_diff = np.einsum("...ji,...jk->...ik", R_pred, R_demo)
    rvec = np.array([geom.rot_to_axis_angle(R_diff[i]) for i in range(len(R_diff))])
    return float(np.sum(rvec ** 2))


def se3_geodesic_loss_np(
    pos_pred: np.ndarray,
    quat_pred: np.ndarray,
    pos_demo: np.ndarray,
    quat_demo: np.ndarray,
    beta: float = 1.0,
) -> float:
    """Position L2 + beta * SO3 geodesic. Quaternions wxyz."""
    loss_pos = np.sum((pos_pred - pos_demo) ** 2)
    R_pred = geom.quat_to_rot(quat_pred)
    R_demo = geom.quat_to_rot(quat_demo)
    if R_pred.ndim == 2:
        R_pred = R_pred.reshape(1, 3, 3)
        R_demo = R_demo.reshape(1, 3, 3)
    R_diff = np.einsum("...ji,...jk->...ik", R_pred, R_demo)
    rvec = np.array([geom.rot_to_axis_angle(R_diff[i]) for i in range(len(R_diff))])
    loss_rot = np.sum(rvec ** 2)
    return float(loss_pos + beta * loss_rot)


def so3_geodesic_loss(R_pred, R_demo):
    """Dispatch to numpy or torch."""
    if hasattr(R_pred, "numpy"):
        return so3_geodesic_loss_torch(R_pred, R_demo)
    return so3_geodesic_loss_np(R_pred, R_demo)


def se3_geodesic_loss(
    pos_pred, quat_pred, pos_demo, quat_demo, beta: float = 1.0
):
    if hasattr(pos_pred, "numpy"):
        return se3_geodesic_loss_torch(pos_pred, quat_pred, pos_demo, quat_demo, beta)
    return se3_geodesic_loss_np(pos_pred, quat_pred, pos_demo, quat_demo, beta)


if HAS_TORCH:
    try:
        from dhb_xr.core.geometry_torch import rot_to_axis_angle_torch, quat_to_rot_torch
    except ImportError:
        rot_to_axis_angle_torch = None
        quat_to_rot_torch = None

    def so3_geodesic_loss_torch(R_pred: torch.Tensor, R_demo: torch.Tensor) -> torch.Tensor:
        R_diff = R_pred.transpose(-2, -1) @ R_demo
        rvec = rot_to_axis_angle_torch(R_diff)
        return (rvec ** 2).sum()

    def se3_geodesic_loss_torch(
        pos_pred: torch.Tensor,
        quat_pred: torch.Tensor,
        pos_demo: torch.Tensor,
        quat_demo: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        loss_pos = ((pos_pred - pos_demo) ** 2).sum()
        if quat_to_rot_torch is not None:
            R_pred = quat_to_rot_torch(quat_pred)
            R_demo = quat_to_rot_torch(quat_demo)
            loss_rot = so3_geodesic_loss_torch(R_pred, R_demo)
        else:
            loss_rot = torch.tensor(0.0, device=pos_pred.device)
        return loss_pos + beta * loss_rot
else:
    so3_geodesic_loss_torch = None
    se3_geodesic_loss_torch = None
