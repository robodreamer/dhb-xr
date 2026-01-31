"""Imitation learning losses in invariant space."""

import numpy as np
from typing import Optional

from dhb_xr.core import geometry as geom

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def quaternion_geodesic_loss_np(q1: np.ndarray, q2: np.ndarray) -> float:
    """Sum of squared quaternion geodesic distances. q1, q2: (N, 4) wxyz."""
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0, 1)
    return np.sum((2 * np.arccos(dot)) ** 2)


def invariant_matching_loss(
    pred_inv: np.ndarray,
    demo_inv: np.ndarray,
    method: str = "dhb_dr",
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    pred_inv, demo_inv: (N, 2*k). method 'dhb_dr' (Euler) or 'dhb_qr' (quaternion).
    For dhb_qr, angular quaternion part uses geodesic; else L2 with optional angle wrap.
    """
    pred_inv = np.asarray(pred_inv)
    demo_inv = np.asarray(demo_inv)
    assert pred_inv.shape == demo_inv.shape
    k = pred_inv.shape[1] // 2
    if weights is None:
        weights = np.ones(pred_inv.shape[1])
    if method == "dhb_qr":
        m_lin = np.sum(weights[0] * (pred_inv[:, 0] - demo_inv[:, 0]) ** 2)
        q_lin = pred_inv[:, 1:5]
        q_lin_d = demo_inv[:, 1:5]
        m_ang = np.sum(weights[k] * (pred_inv[:, k] - demo_inv[:, k]) ** 2)
        q_ang = pred_inv[:, k + 1 : k + 5]
        q_ang_d = demo_inv[:, k + 1 : k + 5]
        loss_lin_q = quaternion_geodesic_loss_np(q_lin, q_lin_d)
        loss_ang_q = quaternion_geodesic_loss_np(q_ang, q_ang_d)
        return float(m_lin + m_ang + loss_lin_q + loss_ang_q)
    diff = pred_inv - demo_inv
    return float(np.sum(weights * (diff ** 2)))


if HAS_TORCH:

    def invariant_matching_loss_torch(
        pred_inv: torch.Tensor,
        demo_inv: torch.Tensor,
        method: str = "dhb_dr",
    ) -> torch.Tensor:
        if method == "dhb_qr":
            m_lin = F.mse_loss(pred_inv[..., 0], demo_inv[..., 0])
            q_lin = pred_inv[..., 1:5]
            q_lin_d = demo_inv[..., 1:5]
            dot = (q_lin * q_lin_d).sum(dim=-1).abs().clamp(0, 1)
            loss_lin_q = (2 * torch.acos(dot)).pow(2).sum()
            k = pred_inv.shape[-1] // 2
            m_ang = F.mse_loss(pred_inv[..., k], demo_inv[..., k])
            q_ang = pred_inv[..., k + 1 : k + 5]
            q_ang_d = demo_inv[..., k + 1 : k + 5]
            dot = (q_ang * q_ang_d).sum(dim=-1).abs().clamp(0, 1)
            loss_ang_q = (2 * torch.acos(dot)).pow(2).sum()
            return m_lin + m_ang + loss_lin_q + loss_ang_q
        return F.mse_loss(pred_inv, demo_inv)
